/** This file contains functionality related to exploration events
 * such as loads, stores and memory (de)allocation. */

#include "MiriInterface.hpp"

// CXX.rs generated headers:
#include "genmc-sys/src/lib.rs.h"

// GenMC headers:
#include "ADT/value_ptr.hpp"
#include "ExecutionGraph/EventLabel.hpp"
#include "ExecutionGraph/LoadAnnotation.hpp"
#include "Runtime/InterpreterEnumAPI.hpp"
#include "Static/ModuleID.hpp"
#include "Support/ASize.hpp"
#include "Support/Error.hpp"
#include "Support/Logger.hpp"
#include "Support/MemAccess.hpp"
#include "Support/RMWOps.hpp"
#include "Support/SAddr.hpp"
#include "Support/SVal.hpp"
#include "Support/ThreadInfo.hpp"
#include "Support/Verbosity.hpp"
#include "Verification/GenMCDriver.hpp"
#include "Verification/MemoryModel.hpp"

// C++ headers:
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

/** Scheduling */

auto MiriGenmcShim::schedule_next(
    const int curr_thread_id,
    const ActionKind curr_thread_next_instr_kind
) -> SchedulingResult {
    // The current thread is the only one where the `kind` could have changed since we last made
    // a scheduling decision.
    threads_action_[curr_thread_id].kind = curr_thread_next_instr_kind;

    auto result = GenMCDriver::scheduleNext(threads_action_);
    return std::visit(
        [](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, int>)
                return SchedulingResult { ExecutionState::Ok, static_cast<int32_t>(arg) };
            else if constexpr (std::is_same_v<T, Blocked>)
                return SchedulingResult { ExecutionState::Blocked, 0 };
            else if constexpr (std::is_same_v<T, Error>)
                return SchedulingResult { ExecutionState::Error, 0 };
            else if constexpr (std::is_same_v<T, Finished>)
                return SchedulingResult { ExecutionState::Finished, 0 };
            else
                static_assert(false, "non-exhaustive visitor!");
        },
        result
    );
}

void MiriGenmcShim::handle_execution_start() {
    threads_action_.clear();
    threads_action_.push_back(Action(ActionKind::Load, Event::getInit()));
    GenMCDriver::handleExecutionStart();
}

auto MiriGenmcShim::handle_execution_end() -> std::unique_ptr<std::string> {
    auto ret = GenMCDriver::handleExecutionEnd();
    return ret.has_value() ? format_error(*ret) : nullptr;
}

/**** Blocking instructions ****/

void MiriGenmcShim::handle_assume_block(ThreadId thread_id, AssumeType assume_type) {
    BUG_ON(getExec().getGraph().isThreadBlocked(thread_id));
    GenMCDriver::handleAssume(nullptr, inc_pos(thread_id), assume_type);
}

/**** Memory access handling ****/

[[nodiscard]] auto MiriGenmcShim::handle_load(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    MemOrdering ord,
    GenmcScalar old_val
) -> LoadResult {
    // `type` is only used for printing.
    const auto type = AType::Unsigned;
    const auto ret = handle_load_reset_if_none<EventLabel::EventLabelKind::Read>(
        thread_id,
        GenmcScalarExt::try_to_sval(old_val),
        ord,
        SAddr(address),
        ASize(size),
        type
    );

    if (const auto* err = std::get_if<VerificationError>(&ret))
        return LoadResultExt::from_error(format_error(*err));
    const auto* ret_val = std::get_if<SVal>(&ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(!ret_val, "Unimplemented: load returned unexpected result.");
    return LoadResultExt::from_value(*ret_val);
}

[[nodiscard]] auto MiriGenmcShim::handle_store(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    GenmcScalar value,
    GenmcScalar old_val,
    MemOrdering ord
) -> StoreResult {
    const auto pos = inc_pos(thread_id);
    const auto ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::Write>(
        nullptr,
        pos,
        GenmcScalarExt::try_to_sval(old_val),
        ord,
        SAddr(address),
        ASize(size),
        /* type */ AType::Unsigned, // `type` is only used for printing.
        GenmcScalarExt::to_sval(value),
        EventDeps()
    );

    if (const auto* err = std::get_if<VerificationError>(&ret))
        return StoreResultExt::from_error(format_error(*err));

    const auto* is_co_max = std::get_if<bool>(&ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(!is_co_max, "Unimplemented: Store returned unexpected result.");
    return StoreResultExt::ok(*is_co_max);
}

void MiriGenmcShim::handle_fence(ThreadId thread_id, MemOrdering ord) {
    const auto pos = inc_pos(thread_id);
    GenMCDriver::handleFence(nullptr, pos, ord, EventDeps());
}

[[nodiscard]] auto MiriGenmcShim::handle_read_modify_write(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    RMWBinOp rmw_op,
    MemOrdering ordering,
    GenmcScalar rhs_value,
    GenmcScalar old_val
) -> ReadModifyWriteResult {
    // NOTE: Both the store and load events should get the same `ordering`, it should not be split
    // into a load and a store component. This means we can have for example `AcqRel` loads and
    // stores, but this is intended for RMW operations.

    // Somewhat confusingly, the GenMC term for RMW read/write labels is
    // `FaiRead` and `FaiWrite`.
    const auto load_ret = handle_load_reset_if_none<EventLabel::EventLabelKind::FaiRead>(
        thread_id,
        GenmcScalarExt::try_to_sval(old_val),
        ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        rmw_op,
        GenmcScalarExt::to_sval(rhs_value),
        EventDeps()
    );
    if (const auto* err = std::get_if<VerificationError>(&load_ret))
        return ReadModifyWriteResultExt::from_error(format_error(*err));

    const auto* ret_val = std::get_if<SVal>(&load_ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(!ret_val, "Unimplemented: read-modify-write returned unexpected result.");
    const auto read_old_val = *ret_val;
    const auto new_value =
        executeRMWBinOp(read_old_val, GenmcScalarExt::to_sval(rhs_value), size, rmw_op);

    const auto storePos = inc_pos(thread_id);
    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::FaiWrite>(
        nullptr,
        storePos,
        GenmcScalarExt::try_to_sval(old_val),
        ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        new_value
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return ReadModifyWriteResultExt::from_error(format_error(*err));

    const auto* is_co_max = std::get_if<bool>(&store_ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(!is_co_max, "Unimplemented: RMW store returned unexpected result.");
    return ReadModifyWriteResultExt::ok(
        /* old_value: */ read_old_val,
        new_value,
        *is_co_max
    );
}

[[nodiscard]] auto MiriGenmcShim::handle_compare_exchange(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    GenmcScalar expected_value,
    GenmcScalar new_value,
    GenmcScalar old_val,
    MemOrdering success_ordering,
    MemOrdering fail_load_ordering,
    bool can_fail_spuriously
) -> CompareExchangeResult {
    // NOTE: Both the store and load events should get the same `ordering`, it should not be split
    // into a load and a store component. This means we can have for example `AcqRel` loads and
    // stores, but this is intended for CAS operations.

    // FIXME(GenMC): properly handle failure memory ordering.

    auto expectedVal = GenmcScalarExt::to_sval(expected_value);
    auto new_val = GenmcScalarExt::to_sval(new_value);

    const auto load_ret = handle_load_reset_if_none<EventLabel::EventLabelKind::CasRead>(
        thread_id,
        GenmcScalarExt::try_to_sval(old_val),
        success_ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        expectedVal,
        new_val
    );
    if (const auto* err = std::get_if<VerificationError>(&load_ret))
        return CompareExchangeResultExt::from_error(format_error(*err));
    const auto* ret_val = std::get_if<SVal>(&load_ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(nullptr == ret_val, "Unimplemented: load returned unexpected result.");
    const auto read_old_val = *ret_val;
    if (read_old_val != expectedVal)
        return CompareExchangeResultExt::failure(read_old_val);

    // FIXME(GenMC): Add support for modelling spurious failures.

    const auto storePos = inc_pos(thread_id);
    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::CasWrite>(
        nullptr,
        storePos,
        GenmcScalarExt::try_to_sval(old_val),
        success_ordering,
        SAddr(address),
        ASize(size),
        AType::Unsigned, // The type is only used for printing.
        new_val
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return CompareExchangeResultExt::from_error(format_error(*err));
    const auto* is_co_max = std::get_if<bool>(&store_ret);
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset}` return values.
    ERROR_ON(!is_co_max, "Unimplemented: compare-exchange store returned unexpected result.");
    return CompareExchangeResultExt::success(read_old_val, *is_co_max);
}

/**** Memory (de)allocation ****/

auto MiriGenmcShim::handle_malloc(ThreadId thread_id, uint64_t size, uint64_t alignment)
    -> uint64_t {
    const auto pos = inc_pos(thread_id);

    // These are only used for printing and features Miri-GenMC doesn't support (yet).
    const auto storage_duration = StorageDuration::SD_Heap;
    // Volatile, as opposed to "persistent" (i.e., non-volatile memory that persists over reboots)
    const auto storage_type = StorageType::ST_Volatile;
    const auto address_space = AddressSpace::AS_User;

    const SVal ret_val = GenMCDriver::handleMalloc(
        nullptr,
        pos,
        size,
        alignment,
        storage_duration,
        storage_type,
        address_space,
        EventDeps()
    );
    return ret_val.get();
}

auto MiriGenmcShim::handle_free(ThreadId thread_id, uint64_t address)
    -> std::unique_ptr<std::string> {
    auto pos = inc_pos(thread_id);
    auto ret = GenMCDriver::handleFree(nullptr, pos, SAddr(address), EventDeps());
    return ret.has_value() ? format_error(*ret) : nullptr;
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

/** Mutexes */

struct MutexState {
    static constexpr SVal UNLOCKED { 0 };
    static constexpr SVal LOCKED { 1 };

    static constexpr bool isValid(SVal v) {
        return v == UNLOCKED || v == LOCKED;
    }
};

auto MiriGenmcShim::handle_mutex_lock(ThreadId thread_id, uint64_t address, uint64_t size)
    -> MutexLockResult {
    // This annotation informs GenMC about the condition required to make this lock call succeed.
    // It stands for `value_read_by_load != MUTEX_LOCKED`.
    const auto size_bits = size * 8;
    const auto annot = std::move(Annotation(
        AssumeType::Spinloop,
        Annotation::ExprVP(
            NeExpr<ModuleID::ID>::create(
                // `RegisterExpr` marks the value of the current expression, i.e., the loaded value.
                // The `id` is ignored by GenMC; it is only used by the LLI frontend to substitute
                // other variables from previous expressions that may be used here.
                RegisterExpr<ModuleID::ID>::create(size_bits, /* id */ 0),
                ConcreteExpr<ModuleID::ID>::create(size_bits, MutexState::LOCKED)
            )
                .release()
        )
    ));

    // As usual, we need to tell GenMC which value was stored at this location before this atomic
    // access, if there previously was a non-atomic initializing access. We set the initial state of
    // a mutex to be "unlocked".
    const auto old_val = MutexState::UNLOCKED;
    const auto load_ret = handle_load_reset_if_none<EventLabel::EventLabelKind::LockCasRead>(
        thread_id,
        old_val,
        address,
        size,
        annot,
        EventDeps()
    );
    if (const auto* err = std::get_if<VerificationError>(&load_ret))
        return MutexLockResultExt::from_error(format_error(*err));
    // If we get a `Reset`, GenMC decided that this lock operation should not yet run, since it
    // would not acquire the mutex. Like the handling of the case further down where we read a `1`
    // ("Mutex already locked"), Miri should call the handle function again once the current thread
    // is scheduled by GenMC the next time.
    if (std::holds_alternative<Reset>(load_ret))
        return MutexLockResultExt::reset();

    const auto* ret_val = std::get_if<SVal>(&load_ret);
    ERROR_ON(!ret_val, "Unimplemented: mutex lock returned unexpected result.");
    ERROR_ON(!MutexState::isValid(*ret_val), "Mutex read value was neither 0 nor 1");
    if (*ret_val == MutexState::LOCKED) {
        // We did not acquire the mutex, so we tell GenMC to block the thread until we can acquire
        // it. GenMC determines this based on the annotation we pass with the load further up in
        // this function, namely when that load will read a value other than `MutexState::LOCKED`.
        this->handle_assume_block(thread_id, AssumeType::Spinloop);
        return MutexLockResultExt::ok(false);
    }

    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::LockCasWrite>(
        nullptr,
        inc_pos(thread_id),
        old_val,
        address,
        size,
        EventDeps()
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return MutexLockResultExt::from_error(format_error(*err));
    // We don't update Miri's memory for this operation so we don't need to know if the store
    // was the co-maximal store, but we still check that we at least get a boolean as the result
    // of the store.
    const auto* is_co_max = std::get_if<bool>(&store_ret);
    ERROR_ON(!is_co_max, "Unimplemented: mutex_try_lock store returned unexpected result.");
    return MutexLockResultExt::ok(true);
}

auto MiriGenmcShim::handle_mutex_try_lock(ThreadId thread_id, uint64_t address, uint64_t size)
    -> MutexLockResult {
    auto& currPos = threads_action_[thread_id].event;
    // As usual, we need to tell GenMC which value was stored at this location before this atomic
    // access, if there previously was a non-atomic initializing access. We set the initial state of
    // a mutex to be "unlocked".
    const auto old_val = MutexState::UNLOCKED;
    const auto load_ret = GenMCDriver::handleLoad<EventLabel::EventLabelKind::TrylockCasRead>(
        nullptr,
        ++currPos,
        old_val,
        SAddr(address),
        ASize(size)
    );
    if (const auto* err = std::get_if<VerificationError>(&load_ret))
        return MutexLockResultExt::from_error(format_error(*err));
    const auto* ret_val = std::get_if<SVal>(&load_ret);
    ERROR_ON(!ret_val, "Unimplemented: mutex trylock load returned unexpected result.");

    ERROR_ON(!MutexState::isValid(*ret_val), "Mutex read value was neither 0 nor 1");
    if (*ret_val == MutexState::LOCKED)
        return MutexLockResultExt::ok(false); /* Lock already held. */

    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::TrylockCasWrite>(
        nullptr,
        ++currPos,
        old_val,
        SAddr(address),
        ASize(size)
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return MutexLockResultExt::from_error(format_error(*err));
    // We don't update Miri's memory for this operation so we don't need to know if the store was
    // co-maximal, but we still check that we get a boolean result.
    const auto* is_co_max = std::get_if<bool>(&store_ret);
    ERROR_ON(!is_co_max, "Unimplemented: store part of mutex try_lock returned unexpected result.");
    return MutexLockResultExt::ok(true);
}

auto MiriGenmcShim::handle_mutex_unlock(ThreadId thread_id, uint64_t address, uint64_t size)
    -> StoreResult {
    const auto pos = inc_pos(thread_id);
    const auto ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::UnlockWrite>(
        nullptr,
        pos,
        // As usual, we need to tell GenMC which value was stored at this location before this
        // atomic access, if there previously was a non-atomic initializing access. We set the
        // initial state of a mutex to be "unlocked".
        /* old_val */ MutexState::UNLOCKED,
        MemOrdering::Release,
        SAddr(address),
        ASize(size),
        AType::Signed,
        /* store_value */ MutexState::UNLOCKED,
        EventDeps()
    );
    if (const auto* err = std::get_if<VerificationError>(&ret))
        return StoreResultExt::from_error(format_error(*err));
    const auto* is_co_max = std::get_if<bool>(&ret);
    ERROR_ON(!is_co_max, "Unimplemented: store part of mutex unlock returned unexpected result.");
    return StoreResultExt::ok(*is_co_max);
}

/** Thread creation/joining */

void MiriGenmcShim::handle_thread_create(ThreadId thread_id, ThreadId parent_id) {
    // NOTE: The threadCreate event happens in the parent:
    const auto pos = inc_pos(parent_id);
    // FIXME(genmc): for supporting symmetry reduction, these will need to be properly set:
    const unsigned fun_id = 0;
    const SVal arg = SVal(0);
    const ThreadInfo child_info =
        ThreadInfo { thread_id, parent_id, fun_id, arg, "unknown thread" };

    const auto child_tid = GenMCDriver::handleThreadCreate(nullptr, pos, child_info, EventDeps());
    // Sanity check the thread id, which is the index in the `threads_action_` array.
    BUG_ON(child_tid != thread_id || child_tid <= 0 || child_tid != threads_action_.size());
    threads_action_.push_back(Action(ActionKind::Load, Event(child_tid, 0)));
}

void MiriGenmcShim::handle_thread_join(ThreadId thread_id, ThreadId child_id) {
    // The thread join event happens in the parent.
    const auto pos = inc_pos(thread_id);

    const auto ret = GenMCDriver::handleThreadJoin(nullptr, pos, child_id, EventDeps());
    // If the join failed, decrease the event index again:
    if (!std::holds_alternative<SVal>(ret)) {
        dec_pos(thread_id);
    }
    // FIXME(genmc): handle `HandleResult::{Invalid, Reset, VerificationError}` return values.

    // NOTE: Thread return value is ignored, since Miri doesn't need it.
}

void MiriGenmcShim::handle_thread_finish(ThreadId thread_id, uint64_t ret_val) {
    const auto pos = inc_pos(thread_id);
    GenMCDriver::handleThreadFinish(nullptr, pos, SVal(ret_val));
}

void MiriGenmcShim::handle_thread_kill(ThreadId thread_id) {
    const auto pos = inc_pos(thread_id);
    GenMCDriver::handleThreadKill(nullptr, pos);
}
