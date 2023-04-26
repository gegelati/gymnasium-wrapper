#include "pendulum.h"

double Pendulum::getActionFromID(const uint64_t& actionID)
{
	double result = this->availableActions.at((actionID) % availableActions.size());
	return result;
}

PyObject* Pendulum::stepEnvironment(uint64_t actionID)
{
	float currentAction = this->getActionFromID(actionID);
	return PyObject_CallMethod(this->pGym, "step", "([f])", currentAction);
}

std::vector<std::reference_wrapper<const Data::DataHandler>> Pendulum::getDataSources()
{
	auto result = std::vector<std::reference_wrapper<const Data::DataHandler>>();
	result.push_back(this->currentState);
	return result;
}

void Pendulum::copyState(PyArrayObject* pState) {
	// Copy the environment state
	for (int i = 0; i < 3; i++) {
		this->currentState.setDataAt(typeid(double), i, *(float*)(PyArray_GETPTR1(pState, i)));
	}
}

Learn::LearningEnvironment* Pendulum::clone() const {
	return new Pendulum(*this);
}
