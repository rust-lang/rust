use std::str::FromStr;
use std::sync::OnceLock;

pub use cxx::UniquePtr;

pub use self::ffi::*;

/// Defined in "genmc/src/Support/SAddr.hpp".
/// The first bit of all global addresses must be set to `1`.
/// This means the mask, interpreted as an address, is the lower bound of where the global address space starts.
///
/// FIXME(genmc): rework this if non-64bit support is added to GenMC (the current allocation scheme only allows for 64bit addresses).
/// FIXME(genmc): currently we use `get_global_alloc_static_mask()` to ensure the constant is consistent between Miri and GenMC,
///   but if https://github.com/dtolnay/cxx/issues/1051 is fixed we could share the constant directly.
pub const GENMC_GLOBAL_ADDRESSES_MASK: u64 = 1 << 63;

/// GenMC thread ids are C++ type `int`, which is equivalent to Rust's `i32` on most platforms.
/// The main thread always has thread id 0.
pub const GENMC_MAIN_THREAD_ID: i32 = 0;

/// Changing GenMC's log level is not thread safe, so we limit it to only be set once to prevent any data races.
/// This value will be initialized when the first `MiriGenmcShim` is created.
static GENMC_LOG_LEVEL: OnceLock<LogLevel> = OnceLock::new();

// Create a new handle to the GenMC model checker.
// The first call to this function determines the log level of GenMC, any future call with a different log level will panic.
pub fn create_genmc_driver_handle(
    params: &GenmcParams,
    genmc_log_level: LogLevel,
    do_estimation: bool,
) -> UniquePtr<MiriGenmcShim> {
    // SAFETY: Only setting the GenMC log level once is guaranteed by the `OnceLock`.
    // No other place calls `set_log_level_raw`, so the `logLevel` value in GenMC will not change once we initialize it once.
    // All functions that use GenMC's `logLevel` can only be accessed in safe Rust through a `MiriGenmcShim`.
    // There is no way to get `MiriGenmcShim` other than through `create_handle`, and we only call it *after* setting the log level, preventing any possible data races.
    assert_eq!(
        &genmc_log_level,
        GENMC_LOG_LEVEL.get_or_init(|| {
            unsafe { set_log_level_raw(genmc_log_level) };
            genmc_log_level
        }),
        "Attempt to change the GenMC log level after it was already set"
    );
    unsafe { MiriGenmcShim::create_handle(params, do_estimation) }
}

impl GenmcScalar {
    pub const UNINIT: Self = Self { value: 0, provenance: 0, is_init: false };

    pub const fn from_u64(value: u64) -> Self {
        Self { value, provenance: 0, is_init: true }
    }

    pub const fn has_provenance(&self) -> bool {
        self.provenance != 0
    }
}

impl Default for GenmcParams {
    fn default() -> Self {
        Self {
            estimation_max: 1000, // default taken from GenMC
            print_random_schedule_seed: false,
            do_symmetry_reduction: false,
            // GenMC graphs can be quite large since Miri produces a lot of (non-atomic) events.
            print_execution_graphs: ExecutiongraphPrinting::None,
            disable_weak_memory_emulation: false,
        }
    }
}

impl Default for LogLevel {
    fn default() -> Self {
        // FIXME(genmc): set `Tip` by default once the GenMC tips are relevant to Miri.
        Self::Warning
    }
}

impl FromStr for SchedulePolicy {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "wf" => SchedulePolicy::WF,
            "wfr" => SchedulePolicy::WFR,
            "arbitrary" | "random" => SchedulePolicy::Arbitrary,
            "ltr" => SchedulePolicy::LTR,
            _ => return Err("invalid scheduling policy"),
        })
    }
}

impl FromStr for LogLevel {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "quiet" => LogLevel::Quiet,
            "error" => LogLevel::Error,
            "warning" => LogLevel::Warning,
            "tip" => LogLevel::Tip,
            "debug1" => LogLevel::Debug1Revisits,
            "debug2" => LogLevel::Debug2MemoryAccesses,
            "debug3" => LogLevel::Debug3ReadsFrom,
            _ => return Err("invalid log level"),
        })
    }
}

#[cxx::bridge]
mod ffi {
    /**** Types shared between Miri/Rust and Miri/C++ through cxx_bridge: ****/

    /// Parameters that will be given to GenMC for setting up the model checker.
    /// The fields of this struct are visible to both Rust and C++.
    /// Note that this struct is #[repr(C)], so the order of fields matters.
    #[derive(Clone, Debug)]
    struct GenmcParams {
        /// Maximum number of executions explored in estimation mode.
        pub estimation_max: u32,
        pub print_random_schedule_seed: bool,
        pub do_symmetry_reduction: bool,
        pub print_execution_graphs: ExecutiongraphPrinting,
        /// Enabling this will set the memory model used by GenMC to "Sequential Consistency" (SC).
        /// This will disable any weak memory effects, which reduces the number of program executions that will be explored.
        pub disable_weak_memory_emulation: bool,
    }

    /// This is mostly equivalent to GenMC `VerbosityLevel`, but the debug log levels are always present (not conditionally compiled based on `ENABLE_GENMC_DEBUG`).
    /// We add this intermediate type to prevent changes to the GenMC log-level from breaking the Miri
    /// build, and to have a stable type for the C++-Rust interface, independent of `ENABLE_GENMC_DEBUG`.
    #[derive(Debug)]
    enum LogLevel {
        /// Disable *all* logging (including error messages on a crash).
        Quiet,
        /// Log errors.
        Error,
        /// Log errors and warnings.
        Warning,
        /// Log errors, warnings and tips.
        Tip,
        /// Debug print considered revisits.
        /// Downgraded to `Tip` if `GENMC_DEBUG` is not enabled.
        Debug1Revisits,
        /// Print the execution graph after every memory access.
        /// Also includes the previous debug log level.
        /// Downgraded to `Tip` if `GENMC_DEBUG` is not enabled.
        Debug2MemoryAccesses,
        /// Print reads-from values considered by GenMC.
        /// Also includes the previous debug log level.
        /// Downgraded to `Tip` if `GENMC_DEBUG` is not enabled.
        Debug3ReadsFrom,
    }

    #[derive(Debug)]
    /// Setting to control which execution graphs GenMC prints after every execution.
    enum ExecutiongraphPrinting {
        /// Print no graphs.
        None,
        /// Print graphs of all fully explored executions.
        Explored,
        /// Print graphs of all blocked executions.
        Blocked,
        /// Print graphs of all executions.
        ExploredAndBlocked,
    }

    /// This type corresponds to `Option<SVal>` (or `std::optional<SVal>`), where `SVal` is the type that GenMC uses for storing values.
    #[derive(Debug, Clone, Copy)]
    struct GenmcScalar {
        /// The raw byte-level value (discarding provenance, if any) of this scalar.
        value: u64,
        /// This is zero for integer values. For pointers, this encodes the provenance by
        /// storing the base address of the allocation that this pointer belongs to.
        /// Operations on `SVal` in GenMC (e.g., `fetch_add`) preserve the `provenance` of the left
        /// argument (`left.fetch_add(right, ...)`).
        provenance: u64,
        /// Indicates whether this value is initialized. If this is `false`, the other fields do not matter.
        /// (Ideally we'd use `std::optional` but CXX does not support that.)
        is_init: bool,
    }

    #[must_use]
    #[derive(Debug, Clone, Copy)]
    enum ExecutionState {
        Ok,
        Error,
        Blocked,
        Finished,
    }

    #[must_use]
    #[derive(Debug)]
    struct SchedulingResult {
        exec_state: ExecutionState,
        next_thread: i32,
    }

    #[must_use]
    #[derive(Debug)]
    struct EstimationResult {
        /// Expected number of total executions.
        mean: f64,
        /// Standard deviation of the total executions estimate.
        sd: f64,
        /// Number of explored executions during the estimation.
        explored_execs: u64,
        /// Number of encounteded blocked executions during the estimation.
        blocked_execs: u64,
    }

    #[must_use]
    #[derive(Debug)]
    struct LoadResult {
        /// If not null, contains the error encountered during the handling of the load.
        error: UniquePtr<CxxString>,
        /// Indicates whether a value was read or not.
        has_value: bool,
        /// The value that was read. Should not be used if `has_value` is `false`.
        read_value: GenmcScalar,
    }

    #[must_use]
    #[derive(Debug)]
    struct StoreResult {
        /// If not null, contains the error encountered during the handling of the store.
        error: UniquePtr<CxxString>,
        /// `true` if the write should also be reflected in Miri's memory representation.
        is_coherence_order_maximal_write: bool,
    }

    #[must_use]
    #[derive(Debug)]
    struct ReadModifyWriteResult {
        /// If there was an error, it will be stored in `error`, otherwise it is `None`.
        error: UniquePtr<CxxString>,
        /// The value that was read by the RMW operation as the left operand.
        old_value: GenmcScalar,
        /// The value that was produced by the RMW operation.
        new_value: GenmcScalar,
        /// `true` if the write should also be reflected in Miri's memory representation.
        is_coherence_order_maximal_write: bool,
    }

    #[must_use]
    #[derive(Debug)]
    struct CompareExchangeResult {
        /// If there was an error, it will be stored in `error`, otherwise it is `None`.
        error: UniquePtr<CxxString>,
        /// The value that was read by the compare-exchange.
        old_value: GenmcScalar,
        /// `true` if compare_exchange op was successful.
        is_success: bool,
        /// `true` if the write should also be reflected in Miri's memory representation.
        is_coherence_order_maximal_write: bool,
    }

    #[must_use]
    #[derive(Debug)]
    struct MutexLockResult {
        /// If there was an error, it will be stored in `error`, otherwise it is `None`.
        error: UniquePtr<CxxString>,
        /// If true, GenMC determined that we should retry the mutex lock operation once the thread attempting to lock is scheduled again.
        is_reset: bool,
        /// Indicate whether the lock was acquired by this thread.
        is_lock_acquired: bool,
    }

    /**** These are GenMC types that we have to copy-paste here since cxx does not support
    "importing" externally defined C++ types. ****/

    #[derive(Clone, Copy, Debug)]
    enum SchedulePolicy {
        LTR,
        WF,
        WFR,
        Arbitrary,
    }

    #[derive(Debug)]
    /// Corresponds to GenMC's type with the same name.
    /// Should only be modified if changed by GenMC.
    enum ActionKind {
        /// Any MIR terminator that's atomic and that may have load semantics.
        /// This includes functions with atomic properties, such as `pthread_create`.
        /// If the exact type of the terminator cannot be determined, load is a safe default `Load`.
        Load,
        /// Anything that's definitely not a `Load`.
        NonLoad,
    }

    #[derive(Debug)]
    /// Corresponds to GenMC's type with the same name.
    /// Should only be modified if changed by GenMC.
    enum MemOrdering {
        NotAtomic = 0,
        Relaxed = 1,
        // We skip 2 in case we support consume.
        Acquire = 3,
        Release = 4,
        AcquireRelease = 5,
        SequentiallyConsistent = 6,
    }

    #[derive(Debug)]
    enum RMWBinOp {
        Xchg = 0,
        Add = 1,
        Sub = 2,
        And = 3,
        Nand = 4,
        Or = 5,
        Xor = 6,
        Max = 7,
        Min = 8,
        UMax = 9,
        UMin = 10,
    }

    #[derive(Debug)]
    enum AssumeType {
        User = 0,
        Barrier = 1,
        Spinloop = 2,
    }

    // # Safety
    //
    // This block is unsafe to allow defining safe methods inside.
    //
    // `get_global_alloc_static_mask` is safe since it just returns a constant.
    // All methods on `MiriGenmcShim` are safe by the correct usage of the two unsafe functions
    // `set_log_level_raw` and `MiriGenmcShim::create_handle`.
    // See the doc comment on those two functions for their safety requirements.
    unsafe extern "C++" {
        include!("MiriInterface.hpp");

        /**** Types shared between Miri/Rust and Miri/C++: ****/
        type MiriGenmcShim;

        /**** Types shared between Miri/Rust and GenMC/C++:
        (This tells cxx that the enums defined above are already defined on the C++ side;
        it will emit assertions to ensure that the two definitions agree.) ****/
        type ActionKind;
        type AssumeType;
        type MemOrdering;
        type RMWBinOp;
        type SchedulePolicy;

        /// Set the log level for GenMC.
        ///
        /// # Safety
        ///
        /// This function is not thread safe, since it writes to the global, mutable, non-atomic `logLevel` variable.
        /// Any GenMC function may read from `logLevel` unsynchronized.
        /// The safest way to use this function is to set the log level exactly once before first calling `create_handle`.
        /// Never calling this function is safe, GenMC will fall back to its default log level.
        unsafe fn set_log_level_raw(log_level: LogLevel);

        /// Create a new `MiriGenmcShim`, which wraps a `GenMCDriver`.
        ///
        /// # Safety
        ///
        /// This function is marked as unsafe since the `logLevel` global variable is non-atomic.
        /// This function should not be called in an unsynchronized way with `set_log_level_raw`, since
        /// this function and any methods on the returned `MiriGenmcShim` may read the `logLevel`,
        /// causing a data race.
        /// The safest way to use these functions is to call `set_log_level_raw` once, and only then
        /// start creating handles.
        /// There should not be any other (safe) way to create a `MiriGenmcShim`.
        #[Self = "MiriGenmcShim"]
        unsafe fn create_handle(
            params: &GenmcParams,
            estimation_mode: bool,
        ) -> UniquePtr<MiriGenmcShim>;
        /// Get the bit mask that GenMC expects for global memory allocations.
        fn get_global_alloc_static_mask() -> u64;

        /// This function must be called at the start of any execution, before any events are reported to GenMC.
        fn handle_execution_start(self: Pin<&mut MiriGenmcShim>);
        /// This function must be called at the end of any execution, even if an error was found during the execution.
        /// Returns `null`, or a string containing an error message if an error occured.
        fn handle_execution_end(self: Pin<&mut MiriGenmcShim>) -> UniquePtr<CxxString>;

        /***** Functions for handling events encountered during program execution. *****/

        /**** Memory access handling ****/
        fn handle_load(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            address: u64,
            size: u64,
            memory_ordering: MemOrdering,
            old_value: GenmcScalar,
        ) -> LoadResult;
        fn handle_read_modify_write(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            address: u64,
            size: u64,
            rmw_op: RMWBinOp,
            ordering: MemOrdering,
            rhs_value: GenmcScalar,
            old_value: GenmcScalar,
        ) -> ReadModifyWriteResult;
        fn handle_compare_exchange(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            address: u64,
            size: u64,
            expected_value: GenmcScalar,
            new_value: GenmcScalar,
            old_value: GenmcScalar,
            success_ordering: MemOrdering,
            fail_load_ordering: MemOrdering,
            can_fail_spuriously: bool,
        ) -> CompareExchangeResult;
        fn handle_store(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            address: u64,
            size: u64,
            value: GenmcScalar,
            old_value: GenmcScalar,
            memory_ordering: MemOrdering,
        ) -> StoreResult;
        fn handle_fence(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            memory_ordering: MemOrdering,
        );

        /**** Memory (de)allocation ****/
        fn handle_malloc(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            size: u64,
            alignment: u64,
        ) -> u64;
        /// Returns true if an error was found.
        fn handle_free(self: Pin<&mut MiriGenmcShim>, thread_id: i32, address: u64) -> bool;

        /**** Thread management ****/
        fn handle_thread_create(self: Pin<&mut MiriGenmcShim>, thread_id: i32, parent_id: i32);
        fn handle_thread_join(self: Pin<&mut MiriGenmcShim>, thread_id: i32, child_id: i32);
        fn handle_thread_finish(self: Pin<&mut MiriGenmcShim>, thread_id: i32, ret_val: u64);
        fn handle_thread_kill(self: Pin<&mut MiriGenmcShim>, thread_id: i32);

        /**** Blocking instructions ****/
        /// Inform GenMC that the thread should be blocked.
        /// Note: this function is currently hardcoded for `AssumeType::User`, corresponding to user supplied assume statements.
        /// This can become a parameter once more types of assumes are added.
        fn handle_assume_block(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            assume_type: AssumeType,
        );

        /**** Mutex handling ****/
        fn handle_mutex_lock(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            address: u64,
            size: u64,
        ) -> MutexLockResult;
        fn handle_mutex_try_lock(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            address: u64,
            size: u64,
        ) -> MutexLockResult;
        fn handle_mutex_unlock(
            self: Pin<&mut MiriGenmcShim>,
            thread_id: i32,
            address: u64,
            size: u64,
        ) -> StoreResult;

        /***** Exploration related functionality *****/

        /// Ask the GenMC scheduler for a new thread to schedule and
        /// return whether the execution is finished, blocked, or can continue.
        /// Updates the next instruction kind for the given thread id.
        fn schedule_next(
            self: Pin<&mut MiriGenmcShim>,
            curr_thread_id: i32,
            curr_thread_next_instr_kind: ActionKind,
        ) -> SchedulingResult;

        /// Check whether there are more executions to explore.
        /// If there are more executions, this method prepares for the next execution and returns `true`.
        fn is_exploration_done(self: Pin<&mut MiriGenmcShim>) -> bool;

        /**** Result querying functionality. ****/

        // NOTE: We don't want to share the `VerificationResult` type with the Rust side, since it
        // is very large, uses features that CXX.rs doesn't support and may change as GenMC changes.
        // Instead, we only use the result on the C++ side, and only expose these getter function to
        // the Rust side.
        // Each `GenMCDriver` contains one `VerificationResult`, and each `MiriGenmcShim` contains on `GenMCDriver`.
        // GenMC builds up the content of the `struct VerificationResult` over the course of an exploration,
        // but it's safe to look at it at any point, since it is only accessible through exactly one `MiriGenmcShim`.
        // All these functions for querying the result can be safely called repeatedly and at any time,
        // though the results may be incomplete if called before `handle_execution_end`.

        /// Get the number of blocked executions encountered by GenMC (cast into a fixed with integer)
        fn get_blocked_execution_count(self: &MiriGenmcShim) -> u64;
        /// Get the number of executions explored by GenMC (cast into a fixed with integer)
        fn get_explored_execution_count(self: &MiriGenmcShim) -> u64;
        /// Get all messages that GenMC produced (errors, warnings), combined into one string.
        fn get_result_message(self: &MiriGenmcShim) -> UniquePtr<CxxString>;
        /// If an error occurred, return a string describing the error, otherwise, return `nullptr`.
        fn get_error_string(self: &MiriGenmcShim) -> UniquePtr<CxxString>;

        /**** Printing functionality. ****/

        /// Get the results of a run in estimation mode.
        fn get_estimation_results(self: &MiriGenmcShim) -> EstimationResult;
    }
}
