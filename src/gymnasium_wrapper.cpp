#include "gymnasium_wrapper.h"

GymnasiumWrapper::~GymnasiumWrapper() {
    // Aquire the GIL in this thread to call Python
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

	Py_DECREF(this->pGym);

    // Release the GIL
	PyGILState_Release(gstate);
}

void GymnasiumWrapper::reset(size_t seed, Learn::LearningMode mode)
{
	// Create seed from seed and mode
	size_t hash_seed = Data::Hash<size_t>()(seed) ^ Data::Hash<Learn::LearningMode>()(mode);

	// Reset the RNG
	this->rng.setSeed(hash_seed);

	// Aquire the GIL in this thread to call Python
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();

	// Reset the learning environment
	PyObject *pMethod = PyObject_GetAttrString(this->pGym, "reset");
	PyObject *pArgs = Py_BuildValue("()");
	PyObject *pKwargs = PyDict_New();
	PyObject *pSeed = Py_BuildValue("i", this->rng.getInt32(0, INT32_MAX));
	PyDict_SetItemString(pKwargs, "seed", pSeed);
	PyObject *pRes = PyObject_Call(pMethod, pArgs, pKwargs);

	// Decrement references to Python objects
	Py_DECREF(pMethod);
	Py_DECREF(pArgs);
	Py_DECREF(pKwargs);
	Py_DECREF(pSeed);
	Py_DECREF(pRes);

	// Release the GIL
	PyGILState_Release(gstate);

	this->isTerminalFlag = false;
}

void GymnasiumWrapper::doAction(uint64_t actionID)
{
	// Aquire the GIL in this thread to call Python
	PyGILState_STATE gstate;
	gstate = PyGILState_Ensure();

	// Run a step in the environment
    PyObject* pRet = this->stepEnvironment(actionID);

	// Copy the environment state
    this->copyState(reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(pRet, 0)));

	// Copy the reward and the flag
	this->reward = PyFloat_AsDouble(PyTuple_GetItem(pRet, 1));
	this->isTerminalFlag = PyLong_AsLong(PyTuple_GetItem(pRet, 2)) + PyLong_AsLong(PyTuple_GetItem(pRet, 3));
	
	// Decrement references to Python objects
	Py_DECREF(pRet);

	// Release the GIL
	PyGILState_Release(gstate);}

bool GymnasiumWrapper::isCopyable() const
{
    // All Python calls should be fenced to not be ran in parallel, so no interest in creating copies
	return false;
}

double GymnasiumWrapper::getScore() const
{
	return this->reward;
}

bool GymnasiumWrapper::isTerminal() const
{
	return this->isTerminalFlag;
}
