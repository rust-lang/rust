/** This file contains functionality related to exploration events
 * such as loads, stores and memory (de)allocation. */

#include "MiriInterface.hpp"

// CXX.rs generated headers:
#include "genmc-sys/src/lib.rs.h"

// GenMC headers:
#include "genmc/ADT/value_ptr.hpp"
#include "genmc/Execution/EventLabel.hpp"
#include "genmc/Execution/LoadAnnotation.hpp"
#include "genmc/Support/ASize.hpp"
#include "genmc/Support/ActionEnums.hpp"
#include "genmc/Support/Error.hpp"
#include "genmc/Support/Logger.hpp"
#include "genmc/Support/MemAccess.hpp"
#include "genmc/Support/ModuleVarID.hpp"
#include "genmc/Support/RMWOps.hpp"
#include "genmc/Support/SAddr.hpp"
#include "genmc/Support/SVal.hpp"
#include "genmc/Support/ThreadInfo.hpp"
#include "genmc/Support/Verbosity.hpp"
#include "genmc/Verification/GenMCDriver.hpp"
#include "genmc/Verification/MemoryModel.hpp"

// C++ headers:
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

/** Scheduling */

auto MiriGenmcInterface::schedule_next(
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
                return SchedulingResult { ExecutionStatus::Ok, static_cast<int32_t>(arg) };
            else if constexpr (std::is_same_v<T, Blocked>)
                return SchedulingResult { ExecutionStatus::Blocked, 0 };
            else if constexpr (std::is_same_v<T, Error>)
                return SchedulingResult { ExecutionStatus::Error, 0 };
            else if constexpr (std::is_same_v<T, Finished>)
                return SchedulingResult { ExecutionStatus::Finished, 0 };
            else
                static_assert(false, "non-exhaustive visitor!");
        },
        result
    );
}

void MiriGenmcInterface::handle_execution_start() {
    threads_action_.clear();
    threads_action_.push_back(Action(ActionKind::Load, Event::getInit()));
    GenMCDriver::handleExecutionStart();
}

auto MiriGenmcInterface::handle_execution_end() -> std::unique_ptr<std::string> {
    auto ret = GenMCDriver::handleExecutionEnd();
    return ret.has_value() ? format_error(*ret) : nullptr;
}

/**** Blocking instructions ****/

void MiriGenmcInterface::handle_assume_block(ThreadId thread_id, AssumeType assume_type) {
    auto ret = GenMCDriver::handleAssume(nullptr, curr_pos(thread_id), assume_type);
    inc_pos(thread_id, ret.count);
}

/**** Memory access handling ****/

[[nodiscard]] auto MiriGenmcInterface::handle_atomic_load(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    MemOrdering ord,
    GenmcScalar old_val
) -> LoadResult {
    const auto ret = GenMCDriver::handleRead(
        nullptr,
        curr_pos(thread_id),
        GenmcScalarExt::try_to_sval(old_val),
        ord,
        SAddr(address),
        ASize(size),
        nullptr,
        std::nullopt,
        EventDeps()
    );
    inc_pos(thread_id, ret.count);
    // FIXME(genmc): handle `HandleResult::Reset` return value.
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return LoadResultExt::from_error(format_error(err));
            },
            [](const Invalid&) { return LoadResultExt::from_invalid(); },
            [](const SVal& v) { return LoadResultExt::from_value(v); },
            [](const Reset&) {
                UNREACHABLE();
                return LoadResultExt::from_invalid();
            },
        },
        ret.result
    );
}

[[nodiscard]] auto
MiriGenmcInterface::handle_non_atomic_load(ThreadId thread_id, uint64_t address, uint64_t size)
    -> NonAtomicResult {
    const auto ret = GenMCDriver::handleNALoad(
        nullptr,
        curr_pos(thread_id),
        SAddr(address),
        ASize(size),
        EventDeps()
    );
    inc_pos(thread_id, ret.count);
    // FIXME(genmc): handle `HandleResult::Reset` return value.
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return NonAtomicResultExt::from_error(format_error(err));
            },
            [](const Invalid&) { return NonAtomicResultExt::from_invalid(); },
            [](const std::monostate&) { return NonAtomicResultExt::ok(); },
            [](const Reset&) {
                UNREACHABLE();
                return NonAtomicResultExt::from_invalid();
            },
        },
        ret.result
    );
}

[[nodiscard]] auto MiriGenmcInterface::handle_atomic_store(
    ThreadId thread_id,
    uint64_t address,
    uint64_t size,
    GenmcScalar value,
    GenmcScalar old_val,
    MemOrdering ord
) -> StoreResult {
    const auto ret = GenMCDriver::handleWrite(
        nullptr,
        curr_pos(thread_id),
        GenmcScalarExt::try_to_sval(old_val),
        ord,
        SAddr(address),
        ASize(size),
        GenmcScalarExt::to_sval(value),
        WriteAttr(),
        EventDeps()
    );

    inc_pos(thread_id, ret.count);
    // FIXME(genmc): handle `HandleResult::Reset` return value.
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return StoreResultExt::from_error(format_error(err));
            },
            [](const Invalid&) { return StoreResultExt::from_invalid(); },
            [](bool is_co_max) { return StoreResultExt::ok(is_co_max); },
            [](const Reset&) {
                UNREACHABLE();
                return StoreResultExt::from_invalid();
            },
        },
        ret.result
    );
}

[[nodiscard]] auto
MiriGenmcInterface::handle_non_atomic_store(ThreadId thread_id, uint64_t address, uint64_t size)
    -> NonAtomicResult {
    const auto ret = GenMCDriver::handleNAStore(
        nullptr,
        curr_pos(thread_id),
        SAddr(address),
        ASize(size),
        EventDeps()
    );
    inc_pos(thread_id, ret.count);
    // FIXME(genmc): handle `HandleResult::Reset` return value.
    return std::visit(
        overloaded {
            [](const VerificationError& err) -> NonAtomicResult {
                return NonAtomicResultExt::from_error(format_error(err));
            },
            [](const Invalid&) { return NonAtomicResultExt::from_invalid(); },
            [](const std::monostate&) { return NonAtomicResultExt::ok(); },
            [](const Reset&) {
                UNREACHABLE();
                return NonAtomicResultExt::from_invalid();
            },
        },
        ret.result
    );
}

void MiriGenmcInterface::handle_fence(ThreadId thread_id, MemOrdering ord) {
    auto ret = GenMCDriver::handleFence(nullptr, curr_pos(thread_id), ord, EventDeps());
    inc_pos(thread_id, ret.count);
}

[[nodiscard]] auto MiriGenmcInterface::handle_read_modify_write(
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

    const auto load_ret = GenMCDriver::handleFaiRead(
        nullptr,
        curr_pos(thread_id),
        GenmcScalarExt::try_to_sval(old_val),
        ordering,
        SAddr(address),
        ASize(size),
        rmw_op,
        GenmcScalarExt::to_sval(rhs_value),
        WriteAttr(),
        nullptr,
        std::nullopt,
        EventDeps()
    );
    inc_pos(thread_id, load_ret.count);
    if (const auto* err = std::get_if<VerificationError>(&load_ret.result))
        return ReadModifyWriteResultExt::from_error(format_error(*err));
    if (std::holds_alternative<GenMCDriver::Invalid>(load_ret.result))
        return ReadModifyWriteResultExt::from_invalid();

    const auto* ret_val = std::get_if<SVal>(&load_ret.result);
    // FIXME(genmc): handle `HandleResult::Reset` return values.
    ERROR_ON(!ret_val, "Unimplemented: read-modify-write returned unexpected result.");
    const auto read_old_val = *ret_val;
    const auto new_value =
        executeRMWBinOp(read_old_val, GenmcScalarExt::to_sval(rhs_value), size, rmw_op);

    const auto store_ret = GenMCDriver::handleFaiWrite(
        nullptr,
        curr_pos(thread_id),
        GenmcScalarExt::try_to_sval(old_val),
        ordering,
        SAddr(address),
        ASize(size),
        new_value,
        WriteAttr(),
        EventDeps()
    );
    inc_pos(thread_id, store_ret.count);
    // FIXME(genmc): handle `HandleResult::Reset` return values.
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return ReadModifyWriteResultExt::from_error(format_error(err));
            },
            [](const GenMCDriver::Invalid&) { return ReadModifyWriteResultExt::from_invalid(); },
            [&read_old_val, &new_value](bool is_co_max) {
                return ReadModifyWriteResultExt::ok(read_old_val, new_value, is_co_max);
            },
            [](const Reset&) {
                UNREACHABLE();
                return ReadModifyWriteResultExt::from_invalid();
            },
        },
        store_ret.result
    );
}

[[nodiscard]] auto MiriGenmcInterface::handle_compare_exchange(
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

    const auto load_ret = GenMCDriver::handleCasRead(
        nullptr,
        curr_pos(thread_id),
        GenmcScalarExt::try_to_sval(old_val),
        success_ordering,
        SAddr(address),
        ASize(size),
        expectedVal,
        new_val,
        WriteAttr(),
        nullptr,
        std::nullopt,
        EventDeps()
    );
    inc_pos(thread_id, load_ret.count);
    if (const auto* err = std::get_if<VerificationError>(&load_ret.result))
        return CompareExchangeResultExt::from_error(format_error(*err));
    if (std::holds_alternative<GenMCDriver::Invalid>(load_ret.result))
        return CompareExchangeResultExt::from_invalid();

    const auto* ret_val = std::get_if<SVal>(&load_ret.result);
    // FIXME(genmc): handle `HandleResult::Reset` return values.
    ERROR_ON(nullptr == ret_val, "Unimplemented: load returned unexpected result.");
    const auto read_old_val = *ret_val;
    if (read_old_val != expectedVal)
        return CompareExchangeResultExt::failure(read_old_val);

    // FIXME(GenMC): Add support for modelling spurious failures.

    const auto store_ret = GenMCDriver::handleCasWrite(
        nullptr,
        curr_pos(thread_id),
        GenmcScalarExt::try_to_sval(old_val),
        success_ordering,
        SAddr(address),
        ASize(size),
        new_val,
        WriteAttr(),
        EventDeps()
    );
    inc_pos(thread_id, store_ret.count);
    // FIXME(genmc): handle `HandleResult::Reset` return values.
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return CompareExchangeResultExt::from_error(format_error(err));
            },
            [](const GenMCDriver::Invalid&) { return CompareExchangeResultExt::from_invalid(); },
            [&read_old_val](bool is_co_max) {
                return CompareExchangeResultExt::success(read_old_val, is_co_max);
            },
            [](const Reset&) {
                UNREACHABLE();
                return CompareExchangeResultExt::from_invalid();
            },
        },
        store_ret.result
    );
}

/**** Memory (de)allocation ****/

auto MiriGenmcInterface::handle_malloc(ThreadId thread_id, uint64_t size, uint64_t alignment)
    -> MallocResult {
    // These are only used for printing and features Miri-GenMC doesn't support (yet).
    const auto storage_duration = StorageDuration::SD_Heap;
    // Volatile, as opposed to "persistent" (i.e., non-volatile memory that persists over reboots)
    const auto storage_type = StorageType::ST_Volatile;
    const auto address_space = AddressSpace::AS_User;

    const auto ret = GenMCDriver::handleMalloc(
        nullptr,
        curr_pos(thread_id),
        size,
        alignment,
        storage_duration,
        storage_type,
        address_space,
        nullptr,
        "",
        EventDeps()
    );
    inc_pos(thread_id, ret.count);
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return MallocResultExt::from_error(format_error(err));
            },
            [](const SVal& addr) { return MallocResultExt::ok(addr); },
            [](const Invalid&) { return MallocResultExt::from_invalid(); },
            [](const Reset&) {
                UNREACHABLE();
                return MallocResultExt::from_invalid();
            },
        },
        ret.result
    );
}

auto MiriGenmcInterface::handle_free(ThreadId thread_id, uint64_t address) -> FreeResult {
    auto ret = GenMCDriver::handleFree(nullptr, curr_pos(thread_id), SAddr(address), EventDeps());
    inc_pos(thread_id, ret.count);
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return FreeResultExt::from_error(format_error(err));
            },
            [](const std::monostate&) { return FreeResultExt::ok(); },
            [](const Invalid&) { return FreeResultExt::from_invalid(); },
            [](const Reset&) {
                UNREACHABLE();
                return FreeResultExt::from_invalid();
            },
        },
        ret.result
    );
}

/**** Estimation mode result ****/

auto MiriGenmcInterface::get_estimation_results() const -> EstimationResult {
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

auto MiriGenmcInterface::handle_mutex_lock(ThreadId thread_id, uint64_t address, uint64_t size)
    -> MutexLockResult {
    // This annotation informs GenMC about the condition required to make this lock call succeed.
    // It stands for `value_read_by_load != MUTEX_LOCKED`.
    const auto size_bits = size * 8;
    const auto annot = std::move(Annotation(
        AssumeType::Spinloop,
        Annotation::ExprVP(
            NeExpr<ModuleVarID>::create(
                // `RegisterExpr` marks the value of the current expression, i.e., the loaded value.
                // The `id` is ignored by GenMC; it is only used by the LLI frontend to substitute
                // other variables from previous expressions that may be used here.
                RegisterExpr<ModuleVarID>::create(size_bits, /* id */ 0),
                ConcreteExpr<ModuleVarID>::create(size_bits, MutexState::LOCKED)
            )
                .release()
        )
    ));

    // As usual, we need to tell GenMC which value was stored at this location before this atomic
    // access, if there previously was a non-atomic initializing access. We set the initial state of
    // a mutex to be "unlocked".
    const auto old_val = MutexState::UNLOCKED;
    const auto load_ret = GenMCDriver::handleLockCasRead(
        nullptr,
        curr_pos(thread_id),
        old_val,
        address,
        size,
        annot,
        EventDeps()
    );
    inc_pos(thread_id, load_ret.count);
    if (const auto* err = std::get_if<VerificationError>(&load_ret.result))
        return MutexLockResultExt::from_error(format_error(*err));
    if (std::holds_alternative<GenMCDriver::Invalid>(load_ret.result))
        return MutexLockResultExt::from_invalid();
    // If we get a `Reset`, GenMC decided that this lock operation should not yet run, since it
    // would not acquire the mutex. Like the handling of the case further down where we read a `1`
    // ("Mutex already locked"), Miri should call the handle function again once the current thread
    // is scheduled by GenMC the next time.
    if (std::holds_alternative<Reset>(load_ret.result))
        return MutexLockResultExt::reset();

    const auto* ret_val = std::get_if<SVal>(&load_ret.result);
    ERROR_ON(!ret_val, "Unimplemented: mutex lock load returned unexpected result.");
    ERROR_ON(
        !MutexState::isValid(*ret_val),
        "Mutex read value was neither 0 nor 1 ({})",
        std::to_string(ret_val->get())
    );
    if (*ret_val == MutexState::LOCKED) {
        // We did not acquire the mutex, so we tell GenMC to block the thread until we can acquire
        // it. GenMC determines this based on the annotation we pass with the load further up in
        // this function, namely when that load will read a value other than `MutexState::LOCKED`.
        this->handle_assume_block(thread_id, AssumeType::Spinloop);
        return MutexLockResultExt::not_acquired();
    }

    const auto store_ret =
        GenMCDriver::handleLockCasWrite(nullptr, curr_pos(thread_id), address, size, EventDeps());
    inc_pos(thread_id, store_ret.count);
    // We don't update Miri's memory for this operation so we don't need to know if the store
    // was the co-maximal store.
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return MutexLockResultExt::from_error(format_error(err));
            },
            [](const GenMCDriver::Invalid&) { return MutexLockResultExt::from_invalid(); },
            [](bool is_co_max) {
                VERIFY(is_co_max);
                return MutexLockResultExt::acquired();
            },
            [](const Reset&) {
                UNREACHABLE();
                return MutexLockResultExt::from_invalid();
            },
        },
        store_ret.result
    );
}

auto MiriGenmcInterface::handle_mutex_try_lock(ThreadId thread_id, uint64_t address, uint64_t size)
    -> MutexLockResult {
    // As usual, we need to tell GenMC which value was stored at this location before this atomic
    // access, if there previously was a non-atomic initializing access. We set the initial state of
    // a mutex to be "unlocked".
    const auto old_val = MutexState::UNLOCKED;
    const auto load_ret = GenMCDriver::handleTrylockCasRead(
        nullptr,
        curr_pos(thread_id),
        old_val,
        SAddr(address),
        ASize(size),
        std::nullopt,
        EventDeps()
    );
    inc_pos(thread_id, load_ret.count);
    if (const auto* err = std::get_if<VerificationError>(&load_ret.result))
        return MutexLockResultExt::from_error(format_error(*err));
    if (std::holds_alternative<GenMCDriver::Invalid>(load_ret.result))
        return MutexLockResultExt::from_invalid();
    const auto* ret_val = std::get_if<SVal>(&load_ret.result);
    ERROR_ON(!ret_val, "Unimplemented: mutex trylock load returned unexpected result.");

    ERROR_ON(!MutexState::isValid(*ret_val), "Mutex read value was neither 0 nor 1");
    if (*ret_val == MutexState::LOCKED)
        return MutexLockResultExt::not_acquired(); /* Lock already held. */

    const auto store_ret = GenMCDriver::handleTrylockCasWrite(
        nullptr,
        curr_pos(thread_id),
        SAddr(address),
        ASize(size),
        EventDeps()
    );
    inc_pos(thread_id, store_ret.count);
    // We don't update Miri's memory for this operation so we don't need to know if the store was
    // co-maximal.
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return MutexLockResultExt::from_error(format_error(err));
            },
            [](const GenMCDriver::Invalid&) { return MutexLockResultExt::from_invalid(); },
            [](bool is_co_max) {
                VERIFY(is_co_max);
                return MutexLockResultExt::acquired();
            },
            [](const Reset&) {
                UNREACHABLE();
                return MutexLockResultExt::from_invalid();
            },
        },
        store_ret.result
    );
}

auto MiriGenmcInterface::handle_mutex_unlock(ThreadId thread_id, uint64_t address, uint64_t size)
    -> StoreResult {
    const auto ret = GenMCDriver::handleUnlockWrite(
        nullptr,
        curr_pos(thread_id),
        // As usual, we need to tell GenMC which value was stored at this location before this
        // atomic access, if there previously was a non-atomic initializing access. We set the
        // initial state of a mutex to be "unlocked".
        /* old_val */ MutexState::UNLOCKED,
        MemOrdering::Release,
        SAddr(address),
        ASize(size),
        /* store_value */ MutexState::UNLOCKED,
        WriteAttr(),
        EventDeps()
    );
    inc_pos(thread_id, ret.count);
    return std::visit(
        overloaded {
            [](const VerificationError& err) {
                return StoreResultExt::from_error(format_error(err));
            },
            [](const GenMCDriver::Invalid&) { return StoreResultExt::from_invalid(); },
            [](bool is_co_max) { return StoreResultExt::ok(is_co_max); },
            [](const Reset&) {
                UNREACHABLE();
                return StoreResultExt::from_invalid();
            },
        },
        ret.result
    );
}

/** Thread creation/joining */

void MiriGenmcInterface::handle_thread_create(ThreadId thread_id, ThreadId parent_id) {
    // FIXME(genmc): for supporting symmetry reduction, these will need to be properly set:
    const unsigned fun_id = 0;
    const SVal arg = SVal(0);
    const ThreadInfo child_info =
        ThreadInfo { thread_id, parent_id, fun_id, arg, "unknown thread" };

    // NOTE: The threadCreate event happens in the parent:
    const auto ret =
        GenMCDriver::handleThreadCreate(nullptr, curr_pos(parent_id), child_info, EventDeps());
    inc_pos(parent_id, ret.count);
    ERROR_ON(
        !std::holds_alternative<int>(ret.result),
        "Unimplemented: unexpected return value for thread create"
    );
    auto child_tid = std::get<int>(ret.result);

    // Sanity check the thread id, which is the index in the `threads_action_` array.
    VERIFY(child_tid == thread_id && child_tid > 0 && child_tid == threads_action_.size());
    threads_action_.push_back(Action(ActionKind::Load, Event(child_tid, 0)));
}

void MiriGenmcInterface::handle_thread_join(ThreadId thread_id, ThreadId child_id) {
    // The thread join event happens in the parent.
    const auto ret =
        GenMCDriver::handleThreadJoin(nullptr, curr_pos(thread_id), child_id, EventDeps());
    inc_pos(thread_id, ret.count);
    // FIXME(genmc): handle `HandleResult::{Invalid, VerificationError}` return values.
    ERROR_ON(
        !std::holds_alternative<SVal>(ret.result) && !std::holds_alternative<Reset>(ret.result),
        "Unimplemented: unexpected return value for thread join"
    );
    // FIXME(genmc): Here Reset{} is silently accepted. Double-check why that is.
    // The reason is likely that, although GenMC wants to re-run the join instruction,
    // when GenMC deems that the join has executed, it will also deem it successful,
    // i.e., the return value is guaranteed to be 0 (or at least we assume that).
    // In this case, it doesn't matter that we don't re-run the instruction, since
    // Miri sets the correct return value, and GenMC will only schedule this thread
    // when it knows the child has terminated.

    // NOTE: Thread return value is ignored, since Miri doesn't need it.
}

void MiriGenmcInterface::handle_thread_finish(ThreadId thread_id, uint64_t ret_val) {
    auto ret = GenMCDriver::handleThreadFinish(nullptr, curr_pos(thread_id), SVal(ret_val));
    inc_pos(thread_id, ret.count);
    ERROR_ON(
        !std::holds_alternative<std::monostate>(ret.result),
        "Unimplemented: unexpected return value for thread finish"
    );
}

void MiriGenmcInterface::handle_thread_kill(ThreadId thread_id) {
    auto ret = GenMCDriver::handleThreadKill(nullptr, curr_pos(thread_id));
    inc_pos(thread_id, ret.count);
    ERROR_ON(
        !std::holds_alternative<std::monostate>(ret.result),
        "Unimplemented: unexpected return value for thread kill"
    );
}
