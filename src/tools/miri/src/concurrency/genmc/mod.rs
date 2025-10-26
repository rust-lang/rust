use std::cell::{Cell, RefCell};
use std::sync::Arc;

use genmc_sys::{
    EstimationResult, GENMC_GLOBAL_ADDRESSES_MASK, GenmcScalar, MemOrdering, MiriGenmcShim,
    RMWBinOp, UniquePtr, create_genmc_driver_handle,
};
use rustc_abi::{Align, Size};
use rustc_const_eval::interpret::{AllocId, InterpCx, InterpResult, interp_ok};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::{throw_machine_stop, throw_ub_format, throw_unsup_format};
// FIXME(genmc,tracing): Implement some work-around for enabling debug/trace level logging (currently disabled statically in rustc).
use tracing::{debug, info};

use self::global_allocations::{EvalContextExt as _, GlobalAllocationHandler};
use self::helper::{
    MAX_ACCESS_SIZE, genmc_scalar_to_scalar, maybe_upgrade_compare_exchange_success_orderings,
    scalar_to_genmc_scalar, to_genmc_rmw_op,
};
use self::run::GenmcMode;
use self::thread_id_map::ThreadIdMap;
use crate::concurrency::genmc::helper::split_access;
use crate::diagnostics::SpanDedupDiagnostic;
use crate::intrinsics::AtomicRmwOp;
use crate::*;

mod config;
mod global_allocations;
mod helper;
mod run;
pub(crate) mod scheduling;
mod shims;
mod thread_id_map;

pub use genmc_sys::GenmcParams;

pub use self::config::GenmcConfig;
pub use self::run::run_genmc_mode;
pub use self::shims::EvalContextExt as GenmcEvalContextExt;

#[derive(Debug)]
pub enum ExecutionEndResult {
    /// An error occurred at the end of the execution.
    Error(String),
    /// No errors occurred, and there are more executions to explore.
    Continue,
    /// No errors occurred and we are finished.
    Stop,
}

#[derive(Clone, Copy, Debug)]
pub enum ExitType {
    MainThreadFinish,
    ExitCalled,
}

/// The exit status of a program.
/// GenMC must store this if a thread exits while any others can still run.
/// The other threads must also be explored before the program is terminated.
#[derive(Clone, Copy, Debug)]
struct ExitStatus {
    exit_code: i32,
    exit_type: ExitType,
}

impl ExitStatus {
    fn do_leak_check(self) -> bool {
        matches!(self.exit_type, ExitType::MainThreadFinish)
    }
}

/// State that is reset at the start of every execution.
#[derive(Debug, Default)]
struct PerExecutionState {
    /// Thread id management, such as mapping between Miri `ThreadId` and GenMC's thread ids, or selecting GenMC thread ids.
    thread_id_manager: RefCell<ThreadIdMap>,

    /// A flag to indicate that we should not forward non-atomic accesses to genmc, e.g. because we
    /// are executing an atomic operation.
    allow_data_races: Cell<bool>,

    /// The exit status of the program. We keep running other threads even after `exit` to ensure
    /// we cover all possible executions.
    /// `None` if no thread has called `exit` and the main thread isn't finished yet.
    exit_status: Cell<Option<ExitStatus>>,

    /// Allocations in this map have been sent to GenMC, and should thus be kept around, since future loads from GenMC may return this allocation again.
    genmc_shared_allocs_map: RefCell<FxHashMap<u64, AllocId>>,
}

impl PerExecutionState {
    fn reset(&self) {
        self.allow_data_races.replace(false);
        self.thread_id_manager.borrow_mut().reset();
        self.exit_status.set(None);
        self.genmc_shared_allocs_map.borrow_mut().clear();
    }
}

struct GlobalState {
    /// Keep track of global allocations, to ensure they keep the same address across different executions, even if the order of allocations changes.
    /// The `AllocId` for globals is stable across executions, so we can use it as an identifier.
    global_allocations: GlobalAllocationHandler,
}

impl GlobalState {
    fn new(target_usize_max: u64) -> Self {
        Self { global_allocations: GlobalAllocationHandler::new(target_usize_max) }
    }
}

/// The main interface with GenMC.
/// Each `GenmcCtx` owns one `MiriGenmcShim`, which owns one `GenMCDriver` (the GenMC model checker).
/// For each GenMC run (estimation or verification), one or more `GenmcCtx` can be created (one per Miri thread).
/// However, for now, we only ever have one `GenmcCtx` per run.
///
/// In multithreading, each worker thread has its own `GenmcCtx`, which will have their results combined in the end.
/// FIXME(genmc): implement multithreading.
///
/// Some data is shared across all `GenmcCtx` in the same run, namely data for global allocation handling.
/// Globals must be allocated in a consistent manner, i.e., each global allocation must have the same address in each execution.
///
/// Some state is reset between each execution in the same run.
pub struct GenmcCtx {
    /// Handle to the GenMC model checker.
    handle: RefCell<UniquePtr<MiriGenmcShim>>,

    /// State that is reset at the start of every execution.
    exec_state: PerExecutionState,

    /// State that persists across executions.
    /// All `GenmcCtx` in one verification step share this state.
    global_state: Arc<GlobalState>,
}

/// GenMC Context creation and administrative / query actions
impl GenmcCtx {
    /// Create a new `GenmcCtx` from a given config.
    fn new(miri_config: &MiriConfig, global_state: Arc<GlobalState>, mode: GenmcMode) -> Self {
        let genmc_config = miri_config.genmc_config.as_ref().unwrap();
        let handle = RefCell::new(create_genmc_driver_handle(
            &genmc_config.params,
            genmc_config.log_level,
            /* do_estimation: */ mode == GenmcMode::Estimation,
        ));
        Self { handle, exec_state: Default::default(), global_state }
    }

    fn get_estimation_results(&self) -> EstimationResult {
        self.handle.borrow().get_estimation_results()
    }

    /// Get the number of blocked executions encountered by GenMC.
    fn get_blocked_execution_count(&self) -> u64 {
        self.handle.borrow().get_blocked_execution_count()
    }

    /// Get the number of explored executions encountered by GenMC.
    fn get_explored_execution_count(&self) -> u64 {
        self.handle.borrow().get_explored_execution_count()
    }

    /// Check if GenMC encountered an error that wasn't immediately returned during execution.
    /// Returns a string representation of the error if one occurred.
    fn try_get_error(&self) -> Option<String> {
        self.handle
            .borrow()
            .get_error_string()
            .as_ref()
            .map(|error| error.to_string_lossy().to_string())
    }

    /// Check if GenMC encountered an error that wasn't immediately returned during execution.
    /// Returns a string representation of the error if one occurred.
    fn get_result_message(&self) -> String {
        self.handle
            .borrow()
            .get_result_message()
            .as_ref()
            .map(|error| error.to_string_lossy().to_string())
            .expect("there should always be a message")
    }

    /// Select whether data race free actions should be allowed. This function should be used carefully!
    ///
    /// If `true` is passed, allow for data races to happen without triggering an error, until this function is called again with argument `false`.
    /// This allows for racy non-atomic memory accesses to be ignored (GenMC is not informed about them at all).
    ///
    /// Certain operations are not permitted in GenMC mode with data races disabled and will cause a panic, e.g., atomic accesses or asking for scheduling decisions.
    ///
    /// # Panics
    /// If data race free is attempted to be set more than once (i.e., no nesting allowed).
    pub(super) fn set_ongoing_action_data_race_free(&self, enable: bool) {
        debug!("GenMC: set_ongoing_action_data_race_free ({enable})");
        let old = self.exec_state.allow_data_races.replace(enable);
        assert_ne!(old, enable, "cannot nest allow_data_races");
    }

    /// Check whether data races are currently allowed (e.g., for loading values for validation which are not actually loaded by the program).
    fn get_alloc_data_races(&self) -> bool {
        self.exec_state.allow_data_races.get()
    }

    /// Get the GenMC id of the currently active thread.
    #[must_use]
    fn active_thread_genmc_tid<'tcx>(&self, machine: &MiriMachine<'tcx>) -> i32 {
        let thread_infos = self.exec_state.thread_id_manager.borrow();
        let curr_thread = machine.threads.active_thread();
        thread_infos.get_genmc_tid(curr_thread)
    }
}

/// GenMC event handling. These methods are used to inform GenMC about events happening in the program, and to handle scheduling decisions.
impl GenmcCtx {
    /// Prepare for the next execution and inform GenMC about it.
    /// Must be called before at the start of every execution.
    fn prepare_next_execution(&self) {
        // Reset per-execution state.
        self.exec_state.reset();
        // Inform GenMC about the new execution.
        self.handle.borrow_mut().pin_mut().handle_execution_start();
    }

    /// Inform GenMC that the program's execution has ended.
    ///
    /// This function must be called even when the execution is blocked
    /// (i.e., it returned a `InterpErrorKind::MachineStop` with error kind `TerminationInfo::GenmcBlockedExecution`).
    /// Don't call this function if an error was found.
    ///
    /// GenMC detects certain errors only when the execution ends.
    /// If an error occured, a string containing a short error description is returned.
    ///
    /// GenMC currently doesn't return an error in all cases immediately when one happens.
    /// This function will also check for those, and return their error description.
    ///
    /// To get the all messages (warnings, errors) that GenMC produces, use the `get_result_message` method.
    fn handle_execution_end(&self) -> ExecutionEndResult {
        let result = self.handle.borrow_mut().pin_mut().handle_execution_end();
        if let Some(error) = result.as_ref() {
            return ExecutionEndResult::Error(error.to_string_lossy().to_string());
        }

        // GenMC decides if there is more to explore:
        let exploration_done = self.handle.borrow_mut().pin_mut().is_exploration_done();

        // GenMC currently does not return an error value immediately in all cases.
        // Both `handle_execution_end` and `is_exploration_done` can produce such errors.
        // We manually query for any errors here to ensure we don't miss any.
        if let Some(error) = self.try_get_error() {
            ExecutionEndResult::Error(error)
        } else if exploration_done {
            ExecutionEndResult::Stop
        } else {
            ExecutionEndResult::Continue
        }
    }

    /**** Memory access handling ****/

    /// Inform GenMC about an atomic load.
    /// Returns that value that the load should read.
    ///
    /// `old_value` is the value that a non-atomic load would read here, or `None` if the memory is uninitalized.
    pub(crate) fn atomic_load<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        ordering: AtomicReadOrd,
        old_val: Option<Scalar>,
    ) -> InterpResult<'tcx, Scalar> {
        assert!(!self.get_alloc_data_races(), "atomic load with data race checking disabled.");
        let genmc_old_value = if let Some(scalar) = old_val {
            scalar_to_genmc_scalar(ecx, self, scalar)?
        } else {
            GenmcScalar::UNINIT
        };
        let read_value =
            self.handle_load(&ecx.machine, address, size, ordering.to_genmc(), genmc_old_value)?;
        genmc_scalar_to_scalar(ecx, self, read_value, size)
    }

    /// Inform GenMC about an atomic store.
    /// Returns `true` if the stored value should be reflected in Miri's memory.
    ///
    /// `old_value` is the value that a non-atomic load would read here, or `None` if the memory is uninitalized.
    pub(crate) fn atomic_store<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        value: Scalar,
        old_value: Option<Scalar>,
        ordering: AtomicWriteOrd,
    ) -> InterpResult<'tcx, bool> {
        assert!(!self.get_alloc_data_races(), "atomic store with data race checking disabled.");
        let genmc_value = scalar_to_genmc_scalar(ecx, self, value)?;
        let genmc_old_value = if let Some(scalar) = old_value {
            scalar_to_genmc_scalar(ecx, self, scalar)?
        } else {
            GenmcScalar::UNINIT
        };
        self.handle_store(
            &ecx.machine,
            address,
            size,
            genmc_value,
            genmc_old_value,
            ordering.to_genmc(),
        )
    }

    /// Inform GenMC about an atomic fence.
    pub(crate) fn atomic_fence<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        ordering: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        assert!(!self.get_alloc_data_races(), "atomic fence with data race checking disabled.");
        self.handle
            .borrow_mut()
            .pin_mut()
            .handle_fence(self.active_thread_genmc_tid(machine), ordering.to_genmc());
        interp_ok(())
    }

    /// Inform GenMC about an atomic read-modify-write operation.
    ///
    /// Returns `(old_val, Option<new_val>)`. `new_val` might not be the latest write in coherence order, which is indicated by `None`.
    ///
    /// `old_value` is the value that a non-atomic load would read here, or `None` if the memory is uninitalized.
    pub(crate) fn atomic_rmw_op<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        atomic_op: AtomicRmwOp,
        is_signed: bool,
        ordering: AtomicRwOrd,
        rhs_scalar: Scalar,
        old_value: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Option<Scalar>)> {
        self.handle_atomic_rmw_op(
            ecx,
            address,
            size,
            ordering,
            to_genmc_rmw_op(atomic_op, is_signed),
            scalar_to_genmc_scalar(ecx, self, rhs_scalar)?,
            scalar_to_genmc_scalar(ecx, self, old_value)?,
        )
    }

    /// Returns `(old_val, Option<new_val>)`. `new_val` might not be the latest write in coherence order, which is indicated by `None`.
    ///
    /// `old_value` is the value that a non-atomic load would read here, or `None` if the memory is uninitalized.
    pub(crate) fn atomic_exchange<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        rhs_scalar: Scalar,
        ordering: AtomicRwOrd,
        old_value: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Option<Scalar>)> {
        self.handle_atomic_rmw_op(
            ecx,
            address,
            size,
            ordering,
            /* genmc_rmw_op */ RMWBinOp::Xchg,
            scalar_to_genmc_scalar(ecx, self, rhs_scalar)?,
            scalar_to_genmc_scalar(ecx, self, old_value)?,
        )
    }

    /// Inform GenMC about an atomic compare-exchange operation.
    ///
    /// Returns the old value read by the compare exchange, optionally the value that Miri should write back to its memory, and whether the compare-exchange was a success or not.
    ///
    /// `old_value` is the value that a non-atomic load would read here, or `None` if the memory is uninitalized.
    pub(crate) fn atomic_compare_exchange<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        expected_old_value: Scalar,
        new_value: Scalar,
        success: AtomicRwOrd,
        fail: AtomicReadOrd,
        can_fail_spuriously: bool,
        old_value: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Option<Scalar>, bool)> {
        assert!(
            !self.get_alloc_data_races(),
            "atomic compare-exchange with data race checking disabled."
        );
        assert_ne!(0, size.bytes());
        assert!(
            size.bytes() <= MAX_ACCESS_SIZE,
            "GenMC currently does not support atomic accesses larger than {} bytes (got {} bytes)",
            MAX_ACCESS_SIZE,
            size.bytes()
        );

        // Upgrade the success memory ordering to equal the failure ordering, since GenMC currently ignores the failure ordering.
        // FIXME(genmc): remove this once GenMC properly supports the failure memory ordering.
        let upgraded_success_ordering =
            maybe_upgrade_compare_exchange_success_orderings(success, fail);

        // FIXME(genmc): remove once GenMC supports failure memory ordering in `compare_exchange`.
        let (effective_failure_ordering, _) = upgraded_success_ordering.split_memory_orderings();
        // Return a warning if the actual orderings don't match the upgraded ones.
        if success != upgraded_success_ordering || effective_failure_ordering != fail {
            static DEDUP: SpanDedupDiagnostic = SpanDedupDiagnostic::new();
            ecx.dedup_diagnostic(&DEDUP, |_first| {
                NonHaltingDiagnostic::GenmcCompareExchangeOrderingMismatch {
                    success_ordering: success,
                    upgraded_success_ordering,
                    failure_ordering: fail,
                    effective_failure_ordering,
                }
            });
        }
        // FIXME(genmc): remove once GenMC implements spurious failures for `compare_exchange_weak`.
        if can_fail_spuriously {
            static DEDUP: SpanDedupDiagnostic = SpanDedupDiagnostic::new();
            ecx.dedup_diagnostic(&DEDUP, |_first| NonHaltingDiagnostic::GenmcCompareExchangeWeak);
        }

        debug!(
            "GenMC: atomic_compare_exchange, address: {address:?}, size: {size:?} (expect: {expected_old_value:?}, new: {new_value:?}, old_value: {old_value:?}, {success:?}, orderings: {fail:?}), can fail spuriously: {can_fail_spuriously}"
        );
        let cas_result = self.handle.borrow_mut().pin_mut().handle_compare_exchange(
            self.active_thread_genmc_tid(&ecx.machine),
            address.bytes(),
            size.bytes(),
            scalar_to_genmc_scalar(ecx, self, expected_old_value)?,
            scalar_to_genmc_scalar(ecx, self, new_value)?,
            scalar_to_genmc_scalar(ecx, self, old_value)?,
            upgraded_success_ordering.to_genmc(),
            fail.to_genmc(),
            can_fail_spuriously,
        );

        if let Some(error) = cas_result.error.as_ref() {
            // FIXME(genmc): error handling
            throw_ub_format!("{}", error.to_string_lossy());
        }

        let return_scalar = genmc_scalar_to_scalar(ecx, self, cas_result.old_value, size)?;
        debug!(
            "GenMC: atomic_compare_exchange: result: {cas_result:?}, returning scalar: {return_scalar:?}"
        );
        // The write can only be a co-maximal write if the CAS succeeded.
        assert!(cas_result.is_success || !cas_result.is_coherence_order_maximal_write);
        interp_ok((
            return_scalar,
            cas_result.is_coherence_order_maximal_write.then_some(new_value),
            cas_result.is_success,
        ))
    }

    /// Inform GenMC about a non-atomic memory load
    ///
    /// NOTE: Unlike for *atomic* loads, we don't return a value here. Non-atomic values are still handled by Miri.
    pub(crate) fn memory_load<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        address: Size,
        size: Size,
    ) -> InterpResult<'tcx> {
        debug!(
            "GenMC: received memory_load (non-atomic): address: {:#x}, size: {}",
            address.bytes(),
            size.bytes()
        );
        if self.get_alloc_data_races() {
            debug!("GenMC: data race checking disabled, ignoring non-atomic load.");
            return interp_ok(());
        }
        // GenMC doesn't like ZSTs, and they can't have any data races, so we skip them
        if size.bytes() == 0 {
            return interp_ok(());
        }

        let handle_load = |address, size| {
            // NOTE: Values loaded non-atomically are still handled by Miri, so we discard whatever we get from GenMC
            let _read_value = self.handle_load(
                machine,
                address,
                size,
                MemOrdering::NotAtomic,
                // This value is used to update the co-maximal store event to the same location.
                // We don't need to update that store, since if it is ever read by any atomic loads, the value will be updated then.
                // We use uninit for lack of a better value, since we don't know whether the location we currently load from is initialized or not.
                GenmcScalar::UNINIT,
            )?;
            interp_ok(())
        };

        // This load is small enough so GenMC can handle it.
        if size.bytes() <= MAX_ACCESS_SIZE {
            return handle_load(address, size);
        }

        // This load is too big to be a single GenMC access, we have to split it.
        // FIXME(genmc): This will misbehave if there are non-64bit-atomics in there.
        // Needs proper support on the GenMC side for large and mixed atomic accesses.
        for (address, size) in split_access(address, size) {
            handle_load(Size::from_bytes(address), Size::from_bytes(size))?;
        }
        interp_ok(())
    }

    /// Inform GenMC about a non-atomic memory store
    ///
    /// NOTE: Unlike for *atomic* stores, we don't provide the actual stored values to GenMC here.
    pub(crate) fn memory_store<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        address: Size,
        size: Size,
    ) -> InterpResult<'tcx> {
        debug!(
            "GenMC: received memory_store (non-atomic): address: {:#x}, size: {}",
            address.bytes(),
            size.bytes()
        );
        if self.get_alloc_data_races() {
            debug!("GenMC: data race checking disabled, ignoring non-atomic store.");
            return interp_ok(());
        }
        // GenMC doesn't like ZSTs, and they can't have any data races, so we skip them
        if size.bytes() == 0 {
            return interp_ok(());
        }

        let handle_store = |address, size| {
            // We always write the the stored values to Miri's memory, whether GenMC says the write is co-maximal or not.
            // The GenMC scheduler ensures that replaying an execution happens in porf-respecting order (po := program order, rf: reads-from order).
            // This means that for any non-atomic read Miri performs, the corresponding write has already been replayed.
            let _is_co_max_write = self.handle_store(
                machine,
                address,
                size,
                // We don't know the value that this store will write, but GenMC expects that we give it an actual value.
                // Unfortunately, there are situations where this value can actually become visible
                // to the program: when there is an atomic load reading from a non-atomic store.
                // FIXME(genmc): update once mixed atomic-non-atomic support is added. Afterwards, this value should never be readable.
                GenmcScalar::from_u64(0xDEADBEEF),
                // This value is used to update the co-maximal store event to the same location.
                // This old value cannot be read anymore by any future loads, since we are doing another non-atomic store to the same location.
                // Any future load will either see the store we are adding now, or we have a data race (there can only be one possible non-atomic value to read from at any time).
                // We use uninit for lack of a better value, since we don't know whether the location we currently write to is initialized or not.
                GenmcScalar::UNINIT,
                MemOrdering::NotAtomic,
            )?;
            interp_ok(())
        };

        // This store is small enough so GenMC can handle it.
        if size.bytes() <= MAX_ACCESS_SIZE {
            return handle_store(address, size);
        }

        // This store is too big to be a single GenMC access, we have to split it.
        // FIXME(genmc): This will misbehave if there are non-64bit-atomics in there.
        // Needs proper support on the GenMC side for large and mixed atomic accesses.
        for (address, size) in split_access(address, size) {
            handle_store(Size::from_bytes(address), Size::from_bytes(size))?;
        }
        interp_ok(())
    }

    /**** Memory (de)allocation ****/

    /// This is also responsible for determining the address of the new allocation.
    pub(crate) fn handle_alloc<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        alloc_id: AllocId,
        size: Size,
        alignment: Align,
        memory_kind: MemoryKind,
    ) -> InterpResult<'tcx, u64> {
        assert!(
            !self.get_alloc_data_races(),
            "memory allocation with data race checking disabled."
        );
        let machine = &ecx.machine;
        if memory_kind == MiriMemoryKind::Global.into() {
            return ecx
                .get_global_allocation_address(&self.global_state.global_allocations, alloc_id);
        }
        // GenMC doesn't support ZSTs, so we set the minimum size to 1 byte
        let genmc_size = size.bytes().max(1);
        let chosen_address = self.handle.borrow_mut().pin_mut().handle_malloc(
            self.active_thread_genmc_tid(machine),
            genmc_size,
            alignment.bytes(),
        );

        // Non-global addresses should not be in the global address space or null.
        assert_ne!(0, chosen_address, "GenMC malloc returned nullptr.");
        assert_eq!(0, chosen_address & GENMC_GLOBAL_ADDRESSES_MASK);
        // Sanity check the address alignment:
        assert!(
            chosen_address.is_multiple_of(alignment.bytes()),
            "GenMC returned address {chosen_address:#x} with lower alignment than requested ({}).",
            alignment.bytes()
        );

        interp_ok(chosen_address)
    }

    pub(crate) fn handle_dealloc<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        alloc_id: AllocId,
        address: Size,
        kind: MemoryKind,
    ) -> InterpResult<'tcx> {
        assert_ne!(
            kind,
            MiriMemoryKind::Global.into(),
            "we probably shouldn't try to deallocate global allocations (alloc_id: {alloc_id:?})"
        );
        assert!(
            !self.get_alloc_data_races(),
            "memory deallocation with data race checking disabled."
        );
        if self
            .handle
            .borrow_mut()
            .pin_mut()
            .handle_free(self.active_thread_genmc_tid(machine), address.bytes())
        {
            // FIXME(genmc): improve error handling.
            // An error was detected, so we get the error string from GenMC.
            throw_ub_format!("{}", self.try_get_error().unwrap());
        }

        interp_ok(())
    }

    /**** Thread management ****/

    pub(crate) fn handle_thread_create<'tcx>(
        &self,
        threads: &ThreadManager<'tcx>,
        // FIXME(genmc,symmetry reduction): pass info to GenMC
        _start_routine: crate::Pointer,
        _func_arg: &crate::ImmTy<'tcx>,
        new_thread_id: ThreadId,
    ) -> InterpResult<'tcx> {
        assert!(!self.get_alloc_data_races(), "thread creation with data race checking disabled.");
        let mut thread_infos = self.exec_state.thread_id_manager.borrow_mut();

        let curr_thread_id = threads.active_thread();
        let genmc_parent_tid = thread_infos.get_genmc_tid(curr_thread_id);
        let genmc_new_tid = thread_infos.add_thread(new_thread_id);

        self.handle.borrow_mut().pin_mut().handle_thread_create(genmc_new_tid, genmc_parent_tid);
        interp_ok(())
    }

    pub(crate) fn handle_thread_join<'tcx>(
        &self,
        active_thread_id: ThreadId,
        child_thread_id: ThreadId,
    ) -> InterpResult<'tcx> {
        assert!(!self.get_alloc_data_races(), "thread join with data race checking disabled.");
        let thread_infos = self.exec_state.thread_id_manager.borrow();

        let genmc_curr_tid = thread_infos.get_genmc_tid(active_thread_id);
        let genmc_child_tid = thread_infos.get_genmc_tid(child_thread_id);

        self.handle.borrow_mut().pin_mut().handle_thread_join(genmc_curr_tid, genmc_child_tid);

        interp_ok(())
    }

    pub(crate) fn handle_thread_finish<'tcx>(&self, threads: &ThreadManager<'tcx>) {
        assert!(!self.get_alloc_data_races(), "thread finish with data race checking disabled.");
        let curr_thread_id = threads.active_thread();

        let thread_infos = self.exec_state.thread_id_manager.borrow();
        let genmc_tid = thread_infos.get_genmc_tid(curr_thread_id);

        debug!("GenMC: thread {curr_thread_id:?} ({genmc_tid:?}) finished.");
        // NOTE: Miri doesn't support return values for threads, but GenMC expects one, so we return 0.
        self.handle.borrow_mut().pin_mut().handle_thread_finish(genmc_tid, /* ret_val */ 0);
    }

    /// Handle a call to `libc::exit` or the exit of the main thread.
    /// Unless an error is returned, the program should continue executing (in a different thread, chosen by the next scheduling call).
    pub(crate) fn handle_exit<'tcx>(
        &self,
        thread: ThreadId,
        exit_code: i32,
        exit_type: ExitType,
    ) -> InterpResult<'tcx> {
        // Calling `libc::exit` doesn't do cleanup, so we skip the leak check in that case.
        let exit_status = ExitStatus { exit_code, exit_type };

        if let Some(old_exit_status) = self.exec_state.exit_status.get() {
            throw_ub_format!(
                "`exit` called twice, first with status {old_exit_status:?}, now with status {exit_status:?}",
            );
        }

        // FIXME(genmc): Add a flag to continue exploration even when the program exits with a non-zero exit code.
        if exit_code != 0 {
            info!("GenMC: 'exit' called with non-zero argument, aborting execution.");
            let leak_check = exit_status.do_leak_check();
            throw_machine_stop!(TerminationInfo::Exit { code: exit_code, leak_check });
        }

        if matches!(exit_type, ExitType::ExitCalled) {
            let thread_infos = self.exec_state.thread_id_manager.borrow();
            let genmc_tid = thread_infos.get_genmc_tid(thread);

            self.handle.borrow_mut().pin_mut().handle_thread_kill(genmc_tid);
        } else {
            assert_eq!(thread, ThreadId::MAIN_THREAD);
        }
        // We continue executing now, so we store the exit status.
        self.exec_state.exit_status.set(Some(exit_status));
        interp_ok(())
    }
}

impl GenmcCtx {
    /// Inform GenMC about a load (atomic or non-atomic).
    /// Returns the value that GenMC wants this load to read.
    fn handle_load<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        address: Size,
        size: Size,
        memory_ordering: MemOrdering,
        genmc_old_value: GenmcScalar,
    ) -> InterpResult<'tcx, GenmcScalar> {
        assert!(
            size.bytes() != 0
                && (memory_ordering == MemOrdering::NotAtomic || size.bytes().is_power_of_two())
        );
        if size.bytes() > MAX_ACCESS_SIZE {
            throw_unsup_format!(
                "GenMC mode currently does not support atomics larger than {MAX_ACCESS_SIZE} bytes.",
            );
        }
        debug!(
            "GenMC: load, address: {addr} == {addr:#x}, size: {size:?}, ordering: {memory_ordering:?}, old_value: {genmc_old_value:x?}",
            addr = address.bytes()
        );
        let load_result = self.handle.borrow_mut().pin_mut().handle_load(
            self.active_thread_genmc_tid(machine),
            address.bytes(),
            size.bytes(),
            memory_ordering,
            genmc_old_value,
        );

        if let Some(error) = load_result.error.as_ref() {
            // FIXME(genmc): error handling
            throw_ub_format!("{}", error.to_string_lossy());
        }

        if !load_result.has_value {
            // FIXME(GenMC): Implementing certain GenMC optimizations will lead to this.
            unimplemented!("GenMC: load returned no value.");
        }

        debug!("GenMC: load returned value: {:?}", load_result.read_value);
        interp_ok(load_result.read_value)
    }

    /// Inform GenMC about a store (atomic or non-atomic).
    /// Returns true if the store is co-maximal, i.e., it should be written to Miri's memory too.
    fn handle_store<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        address: Size,
        size: Size,
        genmc_value: GenmcScalar,
        genmc_old_value: GenmcScalar,
        memory_ordering: MemOrdering,
    ) -> InterpResult<'tcx, bool> {
        assert!(
            size.bytes() != 0
                && (memory_ordering == MemOrdering::NotAtomic || size.bytes().is_power_of_two())
        );
        if size.bytes() > MAX_ACCESS_SIZE {
            throw_unsup_format!(
                "GenMC mode currently does not support atomics larger than {MAX_ACCESS_SIZE} bytes."
            );
        }
        debug!(
            "GenMC: store, address: {addr} = {addr:#x}, size: {size:?}, ordering {memory_ordering:?}, value: {genmc_value:?}",
            addr = address.bytes()
        );
        let store_result = self.handle.borrow_mut().pin_mut().handle_store(
            self.active_thread_genmc_tid(machine),
            address.bytes(),
            size.bytes(),
            genmc_value,
            genmc_old_value,
            memory_ordering,
        );

        if let Some(error) = store_result.error.as_ref() {
            // FIXME(genmc): error handling
            throw_ub_format!("{}", error.to_string_lossy());
        }

        interp_ok(store_result.is_coherence_order_maximal_write)
    }

    /// Inform GenMC about an atomic read-modify-write operation.
    /// This includes atomic swap (also often called "exchange"), but does *not*
    /// include compare-exchange (see `RMWBinOp` for full list of operations).
    /// Returns the previous value at that memory location, and optionally the value that should be written back to Miri's memory.
    fn handle_atomic_rmw_op<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        ordering: AtomicRwOrd,
        genmc_rmw_op: RMWBinOp,
        genmc_rhs_scalar: GenmcScalar,
        genmc_old_value: GenmcScalar,
    ) -> InterpResult<'tcx, (Scalar, Option<Scalar>)> {
        assert!(
            !self.get_alloc_data_races(),
            "atomic read-modify-write operation with data race checking disabled."
        );
        assert_ne!(0, size.bytes());
        assert!(
            size.bytes() <= MAX_ACCESS_SIZE,
            "GenMC currently does not support atomic accesses larger than {} bytes (got {} bytes)",
            MAX_ACCESS_SIZE,
            size.bytes()
        );
        debug!(
            "GenMC: atomic_rmw_op (op: {genmc_rmw_op:?}, rhs value: {genmc_rhs_scalar:?}), address: {address:?}, size: {size:?}, ordering: {ordering:?}",
        );
        let rmw_result = self.handle.borrow_mut().pin_mut().handle_read_modify_write(
            self.active_thread_genmc_tid(&ecx.machine),
            address.bytes(),
            size.bytes(),
            genmc_rmw_op,
            ordering.to_genmc(),
            genmc_rhs_scalar,
            genmc_old_value,
        );

        if let Some(error) = rmw_result.error.as_ref() {
            // FIXME(genmc): error handling
            throw_ub_format!("{}", error.to_string_lossy());
        }

        let old_value_scalar = genmc_scalar_to_scalar(ecx, self, rmw_result.old_value, size)?;

        let new_value_scalar = if rmw_result.is_coherence_order_maximal_write {
            Some(genmc_scalar_to_scalar(ecx, self, rmw_result.new_value, size)?)
        } else {
            None
        };
        interp_ok((old_value_scalar, new_value_scalar))
    }
}

impl VisitProvenance for GenmcCtx {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        let genmc_shared_allocs_map = self.exec_state.genmc_shared_allocs_map.borrow();
        for alloc_id in genmc_shared_allocs_map.values().copied() {
            visit(Some(alloc_id), None);
        }
    }
}
