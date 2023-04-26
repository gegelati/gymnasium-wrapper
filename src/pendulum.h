#ifndef PENDULUM_H
#define PENDULUM_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include "gymnasium_wrapper.h"

#include <gegelati.h>

/**
* \brief Inverted pendulum LearningEnvironment.
*
* The code of the class is interfacing Gegelati with the Gymnasium Pendulum reinforcement learning environment
*/
class Pendulum : public GymnasiumWrapper
{
private:
	/// State of the environment
	Data::PrimitiveTypeArray<double> currentState;

public:

	/**
	* Default constructor.
	*/
	Pendulum(const std::vector<double>& actions) :
		GymnasiumWrapper(actions, "Pendulum-v1"),
		currentState{ 3 }
	{};

	/**
	* \brief Copy constructor for the Pendulum.
	*
	* Default copy constructor since all attributes are trivially copyable.
	*/
	Pendulum(const Pendulum& other) = default;

	/**
	* \brief Get the action from its associated ID.
	*
	* \param[in] actionID the integer representing the selected action.
	* \return the torque applied to the pendulum.
	*/
	double getActionFromID(const uint64_t& actionID);

    /// @brief call the environment to run on the action actionID
    /// @param actionID action to run in the environment
    /// @return the result of the environment
    virtual PyObject* stepEnvironment(uint64_t actionID) override;

	/// Inherited via LearningEnvironment
	virtual std::vector<std::reference_wrapper<const Data::DataHandler>> getDataSources() override;

	/// @brief Copy the state of the environment locally
    /// @param pState Python state to be copied
    virtual void copyState(PyArrayObject* pState) override;

	/// Inherited via LearningEnvironment
	virtual Learn::LearningEnvironment* clone() const override;
};

#endif // !PENDULUM_H