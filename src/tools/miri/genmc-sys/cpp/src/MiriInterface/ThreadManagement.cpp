
/** This file contains functionality related thread management (creation, finishing, join, etc.)  */

#include "MiriInterface.hpp"

// CXX.rs generated headers:
#include "genmc-sys/src/lib.rs.h"

// GenMC headers:
#include "Support/Error.hpp"
#include "Support/Verbosity.hpp"

// C++ headers:
#include <cstdint>

void MiriGenmcShim::handle_thread_create(ThreadId thread_id, ThreadId parent_id) {
    // NOTE: The threadCreate event happens in the parent:
    const auto pos = inc_pos(parent_id);
    // FIXME(genmc): for supporting symmetry reduction, these will need to be properly set:
    const unsigned fun_id = 0;
    const SVal arg = SVal(0);
    const ThreadInfo child_info = ThreadInfo { thread_id, parent_id, fun_id, arg };

    // NOTE: Default memory ordering (`Release`) used here.
    const auto child_tid = GenMCDriver::handleThreadCreate(pos, child_info, EventDeps());
    // Sanity check the thread id, which is the index in the `threads_action_` array.
    BUG_ON(child_tid != thread_id || child_tid <= 0 || child_tid != threads_action_.size());
    threads_action_.push_back(Action(ActionKind::Load, Event(child_tid, 0)));
}

void MiriGenmcShim::handle_thread_join(ThreadId thread_id, ThreadId child_id) {
    // The thread join event happens in the parent.
    const auto pos = inc_pos(thread_id);

    // NOTE: Default memory ordering (`Acquire`) used here.
    const auto ret = GenMCDriver::handleThreadJoin(pos, child_id, EventDeps());
    // If the join failed, decrease the event index again:
    if (!std::holds_alternative<SVal>(ret)) {
        dec_pos(thread_id);
    }
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset, VerificationError}` return values.

    // NOTE: Thread return value is ignored, since Miri doesn't need it.
}

void MiriGenmcShim::handle_thread_finish(ThreadId thread_id, uint64_t ret_val) {
    const auto pos = inc_pos(thread_id);
    // NOTE: Default memory ordering (`Release`) used here.
    GenMCDriver::handleThreadFinish(pos, SVal(ret_val));
}

void MiriGenmcShim::handle_thread_kill(ThreadId thread_id) {
    const auto pos = inc_pos(thread_id);
    GenMCDriver::handleThreadKill(pos);
}
