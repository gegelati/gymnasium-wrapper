#ifndef GYMNASIUM_WRAPPER_H
#define GYMNASIUM_WRAPPER_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <gegelati.h>

/**
* \brief Gymnasium Python Wrapper
*
* The code of the class is interfacing Gegelati with the Gymnasium reinforcement learning environment
*/
class GymnasiumWrapper : public Learn::LearningEnvironment
{
private:
	/// Randomness control
	Mutator::RNG rng;

	/// current reward after the last step
	double reward;

	/// True if the environment has reached a terminal state, either success, failure or timeout
	bool isTerminalFlag;

protected:
	/// Available actions for the LearningAgent
	const std::vector<double> availableActions;

	/// Gymnasium Python Environment
	PyObject* pGym;

public:

	/**
	* Default constructor.
	*/
	GymnasiumWrapper(const std::vector<double>& actions, const char* environment_name) :
		LearningEnvironment(actions.size()),
		availableActions{ actions }
	{
		// Aquire the GIL in this thread to call Python
		PyGILState_STATE gstate;
		gstate = PyGILState_Ensure();
		
		// Load all module level attributes as a dictionary
		PyObject *pName = PyUnicode_FromString("gymnasium");
		PyObject *pModule = PyImport_Import(pName);
		if (pModule == NULL) {
			throw std::runtime_error("Can't find gymnasium module");
		}
        PyObject *pDict = PyModule_GetDict(pModule);
        PyObject *pFunc = PyDict_GetItem(pDict, PyUnicode_FromString("make"));
        this->pGym = PyObject_CallFunction(pFunc, "(s)", environment_name);

		if (this->pGym == NULL) {
			throw std::runtime_error("Can't find " + std::string(environment_name) + " environment");
		}

		// Decrement references to Python objects
		Py_DECREF(pFunc);
		Py_DECREF(pDict);
		Py_DECREF(pModule);
		Py_DECREF(pName);

		// Release the GIL
		PyGILState_Release(gstate);
	};

	/**
	* \brief Copy constructor for the Pendulum.
	*
	* Default copy constructor since all attributes are trivially copyable.
	*/
	GymnasiumWrapper(const GymnasiumWrapper& other) = default;

	~GymnasiumWrapper();

	/// Inherited via LearningEnvironment
	virtual void reset(size_t seed = 0, Learn::LearningMode mode = Learn::LearningMode::TRAINING) override;

    /// @brief call the environment to run on the action actionID
    /// @param actionID action to run in the environment
    /// @return the result of the environment
    virtual PyObject* stepEnvironment(uint64_t actionID) = 0;

    /// @brief Copy the state of the environment locally
    /// @param pState Python state to be copied
    virtual void copyState(PyArrayObject* pState) = 0;

	/// Inherited via LearningEnvironment
	virtual void doAction(uint64_t actionID) override;

	/// Inherited via LearningEnvironment
	virtual bool isCopyable() const override;

	/// Inherited via LearningEnvironment
	virtual double getScore() const override;

	/// Inherited via LearningEnvironment
	virtual bool isTerminal() const override;
};

#endif // !GYMNASIUM_WRAPPER_H