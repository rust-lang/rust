/** This file contains functionality related to exploration, such as scheduling.  */

#include "MiriInterface.hpp"

// CXX.rs generated headers:
#include "genmc-sys/src/lib.rs.h"

// GenMC headers:
#include "Support/Error.hpp"
#include "Support/Verbosity.hpp"

// C++ headers:
#include <cmath>
#include <cstdint>
#include <limits>

auto MiriGenmcShim::schedule_next(
    const int curr_thread_id,
    const ActionKind curr_thread_next_instr_kind
) -> SchedulingResult {
    // The current thread is the only one where the `kind` could have changed since we last made
    // a scheduling decision.
    threads_action_[curr_thread_id].kind = curr_thread_next_instr_kind;

    if (const auto result = GenMCDriver::scheduleNext(threads_action_))
        return SchedulingResult { ExecutionState::Ok, static_cast<int32_t>(result.value()) };
    if (getExec().getGraph().isBlocked())
        return SchedulingResult { ExecutionState::Blocked, 0 };
    if (getResult().status.has_value()) // the "value" here is a `VerificationError`
        return SchedulingResult { ExecutionState::Error, 0 };
    return SchedulingResult { ExecutionState::Finished, 0 };
}

/**** Execution start/end handling ****/

void MiriGenmcShim::handle_execution_start() {
    threads_action_.clear();
    threads_action_.push_back(Action(ActionKind::Load, Event::getInit()));
    GenMCDriver::handleExecutionStart();
}

auto MiriGenmcShim::handle_execution_end() -> std::unique_ptr<std::string> {
    // FIXME(genmc): add error handling once GenMC returns an error here.
    GenMCDriver::handleExecutionEnd();
    return {};
}

/**** Estimation mode result ****/

auto MiriGenmcShim::get_estimation_results() const -> EstimationResult {
    const auto& res = getResult();
    return EstimationResult {
        .mean = static_cast<double>(res.estimationMean),
        .sd = static_cast<double>(std::sqrt(res.estimationVariance)),
        .explored_execs = static_cast<uint64_t>(res.explored),
        .blocked_execs = static_cast<uint64_t>(res.exploredBlocked),
    };
}
