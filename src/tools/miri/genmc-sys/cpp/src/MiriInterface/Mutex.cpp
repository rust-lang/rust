/** This file contains functionality related to handling mutexes.  */

#include "MiriInterface.hpp"

// GenMC headers:
#include "Static/ModuleID.hpp"

// CXX.rs generated headers:
#include "genmc-sys/src/lib.rs.h"

#define MUTEX_UNLOCKED SVal(0)
#define MUTEX_LOCKED SVal(1)

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
                ConcreteExpr<ModuleID::ID>::create(size_bits, MUTEX_LOCKED)
            )
                .release()
        )
    ));

    // As usual, we need to tell GenMC which value was stored at this location before this atomic
    // access, if there previously was a non-atomic initializing access. We set the initial state of
    // a mutex to be "unlocked".
    const auto old_val = MUTEX_UNLOCKED;
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
    ERROR_ON(
        *ret_val != MUTEX_UNLOCKED && *ret_val != MUTEX_LOCKED,
        "Mutex read value was neither 0 nor 1"
    );
    const bool is_lock_acquired = *ret_val == MUTEX_UNLOCKED;
    if (is_lock_acquired) {
        const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::LockCasWrite>(
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
        const bool* is_coherence_order_maximal_write = std::get_if<bool>(&store_ret);
        ERROR_ON(
            nullptr == is_coherence_order_maximal_write,
            "Unimplemented: store part of mutex try_lock returned unexpected result."
        );
    } else {
        // We did not acquire the mutex, so we tell GenMC to block the thread until we can acquire
        // it. GenMC determines this based on the annotation we pass with the load further up in
        // this function, namely when that load will read a value other than `MUTEX_LOCKED`.
        this->handle_assume_block(thread_id, AssumeType::Spinloop);
    }
    return MutexLockResultExt::ok(is_lock_acquired);
}

auto MiriGenmcShim::handle_mutex_try_lock(ThreadId thread_id, uint64_t address, uint64_t size)
    -> MutexLockResult {
    auto& currPos = threads_action_[thread_id].event;
    // As usual, we need to tell GenMC which value was stored at this location before this atomic
    // access, if there previously was a non-atomic initializing access. We set the initial state of
    // a mutex to be "unlocked".
    const auto old_val = MUTEX_UNLOCKED;
    const auto load_ret = GenMCDriver::handleLoad<EventLabel::EventLabelKind::TrylockCasRead>(
        ++currPos,
        old_val,
        SAddr(address),
        ASize(size)
    );
    if (const auto* err = std::get_if<VerificationError>(&load_ret))
        return MutexLockResultExt::from_error(format_error(*err));
    const auto* ret_val = std::get_if<SVal>(&load_ret);
    if (nullptr == ret_val) {
        ERROR("Unimplemented: mutex trylock load returned unexpected result.");
    }

    ERROR_ON(
        *ret_val != MUTEX_UNLOCKED && *ret_val != MUTEX_LOCKED,
        "Mutex read value was neither 0 nor 1"
    );
    const bool is_lock_acquired = *ret_val == MUTEX_UNLOCKED;
    if (!is_lock_acquired) {
        return MutexLockResultExt::ok(false); /* Lock already held. */
    }

    const auto store_ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::TrylockCasWrite>(
        ++currPos,
        old_val,
        SAddr(address),
        ASize(size)
    );
    if (const auto* err = std::get_if<VerificationError>(&store_ret))
        return MutexLockResultExt::from_error(format_error(*err));
    // We don't update Miri's memory for this operation so we don't need to know if the store was
    // co-maximal, but we still check that we get a boolean result.
    const bool* is_coherence_order_maximal_write = std::get_if<bool>(&store_ret);
    ERROR_ON(
        nullptr == is_coherence_order_maximal_write,
        "Unimplemented: store part of mutex try_lock returned unexpected result."
    );
    return MutexLockResultExt::ok(true);
}

auto MiriGenmcShim::handle_mutex_unlock(ThreadId thread_id, uint64_t address, uint64_t size)
    -> StoreResult {
    const auto pos = inc_pos(thread_id);
    const auto ret = GenMCDriver::handleStore<EventLabel::EventLabelKind::UnlockWrite>(
        pos,
        // As usual, we need to tell GenMC which value was stored at this location before this
        // atomic access, if there previously was a non-atomic initializing access. We set the
        // initial state of a mutex to be "unlocked".
        /* old_val */ MUTEX_UNLOCKED,
        MemOrdering::Release,
        SAddr(address),
        ASize(size),
        AType::Signed,
        /* store_value */ MUTEX_UNLOCKED,
        EventDeps()
    );
    if (const auto* err = std::get_if<VerificationError>(&ret))
        return StoreResultExt::from_error(format_error(*err));
    const bool* is_coherence_order_maximal_write = std::get_if<bool>(&ret);
    ERROR_ON(
        nullptr == is_coherence_order_maximal_write,
        "Unimplemented: store part of mutex unlock returned unexpected result."
    );
    return StoreResultExt::ok(*is_coherence_order_maximal_write);
}
