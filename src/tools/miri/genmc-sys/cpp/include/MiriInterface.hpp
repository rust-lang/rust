#ifndef GENMC_MIRI_INTERFACE_HPP
#define GENMC_MIRI_INTERFACE_HPP

// CXX.rs generated headers:
#include "rust/cxx.h"

// GenMC generated headers:
#include "config.h"

// Miri `genmc-sys/src_cpp` headers:
#include "ResultHandling.hpp"

// GenMC headers:
#include "ExecutionGraph/EventLabel.hpp"
#include "Support/MemOrdering.hpp"
#include "Support/RMWOps.hpp"
#include "Verification/Config.hpp"
#include "Verification/GenMCDriver.hpp"

// C++ headers:
#include <cstdint>
#include <format>
#include <iomanip>
#include <memory>

/**** Types available to both Rust and C++ ****/

struct GenmcParams;
enum class LogLevel : std::uint8_t;

struct GenmcScalar;
struct SchedulingResult;
struct EstimationResult;
struct LoadResult;
struct StoreResult;
struct ReadModifyWriteResult;
struct CompareExchangeResult;
struct MutexLockResult;

// GenMC uses `int` for its thread IDs.
using ThreadId = int;

/// Set the log level for GenMC.
///
/// # Safety
///
/// This function is not thread safe, since it writes to the global, mutable, non-atomic `logLevel`
/// variable. Any GenMC function may read from `logLevel` unsynchronized.
/// The safest way to use this function is to set the log level exactly once before first calling
/// `create_handle`.
/// Never calling this function is safe, GenMC will fall back to its default log level.
/* unsafe */ void set_log_level_raw(LogLevel log_level);

struct MiriGenmcShim : private GenMCDriver {

  public:
    MiriGenmcShim(std::shared_ptr<const Config> conf, Mode mode /* = VerificationMode{} */)
        : GenMCDriver(std::move(conf), nullptr, mode) {}

    /// Create a new `MiriGenmcShim`, which wraps a `GenMCDriver`.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe since the `logLevel` global variable is non-atomic.
    /// This function should not be called in an unsynchronized way with `set_log_level_raw`,
    /// since this function and any methods on the returned `MiriGenmcShim` may read the
    /// `logLevel`, causing a data race. The safest way to use these functions is to call
    /// `set_log_level_raw` once, and only then start creating handles. There should not be any
    /// other (safe) way to create a `MiriGenmcShim`.
    /* unsafe */ static auto create_handle(const GenmcParams& params, bool estimation_mode)
        -> std::unique_ptr<MiriGenmcShim>;

    virtual ~MiriGenmcShim() {}

    /**** Execution start/end handling ****/

    // This function must be called at the start of any execution, before any events are
    // reported to GenMC.
    void handle_execution_start();
    // This function must be called at the end of any execution, even if an error was found
    // during the execution.
    // Returns `null`, or a string containing an error message if an error occured.
    std::unique_ptr<std::string> handle_execution_end();

    /***** Functions for handling events encountered during program execution. *****/

    /**** Memory access handling ****/

    [[nodiscard]] LoadResult handle_load(
        ThreadId thread_id,
        uint64_t address,
        uint64_t size,
        MemOrdering ord,
        GenmcScalar old_val
    );
    [[nodiscard]] ReadModifyWriteResult handle_read_modify_write(
        ThreadId thread_id,
        uint64_t address,
        uint64_t size,
        RMWBinOp rmw_op,
        MemOrdering ordering,
        GenmcScalar rhs_value,
        GenmcScalar old_val
    );
    [[nodiscard]] CompareExchangeResult handle_compare_exchange(
        ThreadId thread_id,
        uint64_t address,
        uint64_t size,
        GenmcScalar expected_value,
        GenmcScalar new_value,
        GenmcScalar old_val,
        MemOrdering success_ordering,
        MemOrdering fail_load_ordering,
        bool can_fail_spuriously
    );
    [[nodiscard]] StoreResult handle_store(
        ThreadId thread_id,
        uint64_t address,
        uint64_t size,
        GenmcScalar value,
        GenmcScalar old_val,
        MemOrdering ord
    );

    void handle_fence(ThreadId thread_id, MemOrdering ord);

    /**** Memory (de)allocation ****/
    auto handle_malloc(ThreadId thread_id, uint64_t size, uint64_t alignment) -> uint64_t;
    auto handle_free(ThreadId thread_id, uint64_t address) -> bool;

    /**** Thread management ****/
    void handle_thread_create(ThreadId thread_id, ThreadId parent_id);
    void handle_thread_join(ThreadId thread_id, ThreadId child_id);
    void handle_thread_finish(ThreadId thread_id, uint64_t ret_val);
    void handle_thread_kill(ThreadId thread_id);

    /**** Blocking instructions ****/
    /// Inform GenMC that the thread should be blocked.
    void handle_assume_block(ThreadId thread_id, AssumeType assume_type);

    /**** Mutex handling ****/
    auto handle_mutex_lock(ThreadId thread_id, uint64_t address, uint64_t size) -> MutexLockResult;
    auto handle_mutex_try_lock(ThreadId thread_id, uint64_t address, uint64_t size)
        -> MutexLockResult;
    auto handle_mutex_unlock(ThreadId thread_id, uint64_t address, uint64_t size) -> StoreResult;

    /***** Exploration related functionality *****/

    /** Ask the GenMC scheduler for a new thread to schedule and return whether the execution is
     * finished, blocked, or can continue.
     * Updates the next instruction kind for the given thread id. */
    auto schedule_next(const int curr_thread_id, const ActionKind curr_thread_next_instr_kind)
        -> SchedulingResult;

    /**
     * Check whether there are more executions to explore.
     * If there are more executions, this method prepares for the next execution and returns
     * `true`. Returns true if there are no more executions to explore. */
    auto is_exploration_done() -> bool {
        return GenMCDriver::done();
    }

    /**** Result querying functionality. ****/

    // NOTE: We don't want to share the `VerificationResult` type with the Rust side, since it
    // is very large, uses features that CXX.rs doesn't support and may change as GenMC changes.
    // Instead, we only use the result on the C++ side, and only expose these getter function to
    // the Rust side.

    // Note that CXX.rs doesn't support returning a C++ string to Rust by value,
    // it must be behind an indirection like a `unique_ptr` (tested with CXX 1.0.170).

    /// Get the number of blocked executions encountered by GenMC (cast into a fixed with
    /// integer)
    auto get_blocked_execution_count() const -> uint64_t {
        return static_cast<uint64_t>(getResult().exploredBlocked);
    }

    /// Get the number of executions explored by GenMC (cast into a fixed with integer)
    auto get_explored_execution_count() const -> uint64_t {
        return static_cast<uint64_t>(getResult().explored);
    }

    /// Get all messages that GenMC produced (errors, warnings), combined into one string.
    auto get_result_message() const -> std::unique_ptr<std::string> {
        return std::make_unique<std::string>(getResult().message);
    }

    /// If an error occurred, return a string describing the error, otherwise, return `nullptr`.
    auto get_error_string() const -> std::unique_ptr<std::string> {
        const auto& result = GenMCDriver::getResult();
        if (result.status.has_value())
            return format_error(result.status.value());
        return nullptr;
    }

    /**** Printing and estimation mode functionality. ****/

    /// Get the results of a run in estimation mode.
    auto get_estimation_results() const -> EstimationResult;

  private:
    /** Increment the event index in the given thread by 1 and return the new event. */
    [[nodiscard]] inline auto inc_pos(ThreadId tid) -> Event {
        ERROR_ON(tid >= threads_action_.size(), "ThreadId out of bounds");
        return ++threads_action_[tid].event;
    }
    /** Decrement the event index in the given thread by 1 and return the new event. */
    inline auto dec_pos(ThreadId tid) -> Event {
        ERROR_ON(tid >= threads_action_.size(), "ThreadId out of bounds");
        return --threads_action_[tid].event;
    }

    /**
     * Helper function for loads that need to reset the event counter when no value is returned.
     * Same syntax as `GenMCDriver::handleLoad`, but this takes a thread id instead of an Event.
     * Automatically calls `inc_pos` and `dec_pos` where needed for the given thread.
     */
    template <EventLabel::EventLabelKind k, typename... Ts>
    auto handle_load_reset_if_none(ThreadId tid, std::optional<SVal> old_val, Ts&&... params)
        -> HandleResult<SVal> {
        const auto pos = inc_pos(tid);
        const auto ret = GenMCDriver::handleLoad<k>(pos, old_val, std::forward<Ts>(params)...);
        // If we didn't get a value, we have to reset the index of the current thread.
        if (!std::holds_alternative<SVal>(ret)) {
            dec_pos(tid);
        }
        return ret;
    }

    /**
     * GenMC uses the term `Action` to refer to a struct of:
     * - `ActionKind`, storing whether the next instruction in a thread may be a load
     * - `Event`, storing the most recent event index added for a thread
     *
     * Here we store the "action" for each thread. In particular we use this to assign event
     * indices, since GenMC expects us to do that.
     */
    std::vector<Action> threads_action_;
};

/// Get the bit mask that GenMC expects for global memory allocations.
/// FIXME(genmc): currently we use `get_global_alloc_static_mask()` to ensure the
/// `SAddr::staticMask` constant is consistent between Miri and GenMC, but if
/// https://github.com/dtolnay/cxx/issues/1051 is fixed we could share the constant
///   directly.
constexpr auto get_global_alloc_static_mask() -> uint64_t {
    return SAddr::staticMask;
}

// CXX.rs generated headers:
// NOTE: this must be included *after* `MiriGenmcShim` and all the other types we use are defined,
// otherwise there will be compilation errors due to missing definitions.
#include "genmc-sys/src/lib.rs.h"

/**** Result handling ****/
// NOTE: these must come after the cxx_bridge generated code, since that contains the actual
// definitions of these types.

namespace GenmcScalarExt {
inline GenmcScalar uninit() {
    return GenmcScalar {
        .value = 0,
        .provenance = 0,
        .is_init = false,
    };
}

inline GenmcScalar from_sval(SVal sval) {
    return GenmcScalar {
        .value = sval.get(),
        .provenance = sval.getProvenance(),
        .is_init = true,
    };
}

inline SVal to_sval(GenmcScalar scalar) {
    ERROR_ON(!scalar.is_init, "Cannot convert an uninitialized `GenmcScalar` into an `SVal`\n");
    return SVal(scalar.value, scalar.provenance);
}

inline std::optional<SVal> try_to_sval(GenmcScalar scalar) {
    if (scalar.is_init)
        return { SVal(scalar.value, scalar.provenance) };
    return std::nullopt;
}
} // namespace GenmcScalarExt

namespace LoadResultExt {
inline LoadResult no_value() {
    return LoadResult {
        .error = std::unique_ptr<std::string>(nullptr),
        .has_value = false,
        .read_value = GenmcScalarExt::uninit(),
    };
}

inline LoadResult from_value(SVal read_value) {
    return LoadResult { .error = std::unique_ptr<std::string>(nullptr),
                        .has_value = true,
                        .read_value = GenmcScalarExt::from_sval(read_value) };
}

inline LoadResult from_error(std::unique_ptr<std::string> error) {
    return LoadResult { .error = std::move(error),
                        .has_value = false,
                        .read_value = GenmcScalarExt::uninit() };
}
} // namespace LoadResultExt

namespace StoreResultExt {
inline StoreResult ok(bool is_coherence_order_maximal_write) {
    return StoreResult { /* error: */ std::unique_ptr<std::string>(nullptr),
                         is_coherence_order_maximal_write };
}

inline StoreResult from_error(std::unique_ptr<std::string> error) {
    return StoreResult { .error = std::move(error), .is_coherence_order_maximal_write = false };
}
} // namespace StoreResultExt

namespace ReadModifyWriteResultExt {
inline ReadModifyWriteResult
ok(SVal old_value, SVal new_value, bool is_coherence_order_maximal_write) {
    return ReadModifyWriteResult { .error = std::unique_ptr<std::string>(nullptr),
                                   .old_value = GenmcScalarExt::from_sval(old_value),
                                   .new_value = GenmcScalarExt::from_sval(new_value),
                                   .is_coherence_order_maximal_write =
                                       is_coherence_order_maximal_write };
}

inline ReadModifyWriteResult from_error(std::unique_ptr<std::string> error) {
    return ReadModifyWriteResult { .error = std::move(error),
                                   .old_value = GenmcScalarExt::uninit(),
                                   .new_value = GenmcScalarExt::uninit(),
                                   .is_coherence_order_maximal_write = false };
}
} // namespace ReadModifyWriteResultExt

namespace CompareExchangeResultExt {
inline CompareExchangeResult success(SVal old_value, bool is_coherence_order_maximal_write) {
    return CompareExchangeResult { .error = nullptr,
                                   .old_value = GenmcScalarExt::from_sval(old_value),
                                   .is_success = true,
                                   .is_coherence_order_maximal_write =
                                       is_coherence_order_maximal_write };
}

inline CompareExchangeResult failure(SVal old_value) {
    return CompareExchangeResult { .error = nullptr,
                                   .old_value = GenmcScalarExt::from_sval(old_value),
                                   .is_success = false,
                                   .is_coherence_order_maximal_write = false };
}

inline CompareExchangeResult from_error(std::unique_ptr<std::string> error) {
    return CompareExchangeResult { .error = std::move(error),
                                   .old_value = GenmcScalarExt::uninit(),
                                   .is_success = false,
                                   .is_coherence_order_maximal_write = false };
}
} // namespace CompareExchangeResultExt

namespace MutexLockResultExt {
inline MutexLockResult ok(bool is_lock_acquired) {
    return MutexLockResult { /* error: */ nullptr, /* is_reset: */ false, is_lock_acquired };
}

inline MutexLockResult reset() {
    return MutexLockResult { /* error: */ nullptr,
                             /* is_reset: */ true,
                             /* is_lock_acquired: */ false };
}

inline MutexLockResult from_error(std::unique_ptr<std::string> error) {
    return MutexLockResult { /* error: */ std::move(error),
                             /* is_reset: */ false,
                             /* is_lock_acquired: */ false };
}
} // namespace MutexLockResultExt

#endif /* GENMC_MIRI_INTERFACE_HPP */
