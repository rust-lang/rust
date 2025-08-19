#![allow(unused)] // FIXME(GenMC): remove this

use std::cell::Cell;

use genmc_sys::{GenmcParams, createGenmcHandle};
use rustc_abi::{Align, Size};
use rustc_const_eval::interpret::{InterpCx, InterpResult, interp_ok};
use rustc_middle::mir;

use crate::{
    AtomicFenceOrd, AtomicReadOrd, AtomicRwOrd, AtomicWriteOrd, MemoryKind, MiriConfig,
    MiriMachine, Scalar, ThreadId, ThreadManager, VisitProvenance, VisitWith,
};

mod config;

pub use self::config::GenmcConfig;

// FIXME(GenMC): add fields
pub struct GenmcCtx {
    /// Some actions Miri does are allowed to cause data races.
    /// GenMC will not be informed about certain actions (e.g. non-atomic loads) when this flag is set.
    allow_data_races: Cell<bool>,
}

impl GenmcCtx {
    /// Create a new `GenmcCtx` from a given config.
    pub fn new(miri_config: &MiriConfig) -> Self {
        let genmc_config = miri_config.genmc_config.as_ref().unwrap();

        let handle = createGenmcHandle(&genmc_config.params);
        assert!(!handle.is_null());

        eprintln!("Miri: GenMC handle creation successful!");

        drop(handle);
        eprintln!("Miri: Dropping GenMC handle successful!");

        // FIXME(GenMC): implement
        std::process::exit(0);
    }

    pub fn get_stuck_execution_count(&self) -> usize {
        todo!()
    }

    pub fn print_genmc_graph(&self) {
        todo!()
    }

    /// This function determines if we should continue exploring executions or if we are done.
    ///
    /// In GenMC mode, the input program should be repeatedly executed until this function returns `true` or an error is found.
    pub fn is_exploration_done(&self) -> bool {
        todo!()
    }

    /// Inform GenMC that a new program execution has started.
    /// This function should be called at the start of every execution.
    pub(crate) fn handle_execution_start(&self) {
        todo!()
    }

    /// Inform GenMC that the program's execution has ended.
    ///
    /// This function must be called even when the execution got stuck (i.e., it returned a `InterpErrorKind::MachineStop` with error kind `TerminationInfo::GenmcStuckExecution`).
    pub(crate) fn handle_execution_end<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    ) -> Result<(), String> {
        todo!()
    }

    /**** Memory access handling ****/

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
        let old = self.allow_data_races.replace(enable);
        assert_ne!(old, enable, "cannot nest allow_data_races");
    }

    pub(crate) fn atomic_load<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        ordering: AtomicReadOrd,
        old_val: Option<Scalar>,
    ) -> InterpResult<'tcx, Scalar> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    pub(crate) fn atomic_store<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        value: Scalar,
        ordering: AtomicWriteOrd,
    ) -> InterpResult<'tcx, ()> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    pub(crate) fn atomic_fence<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        ordering: AtomicFenceOrd,
    ) -> InterpResult<'tcx, ()> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    /// Inform GenMC about an atomic read-modify-write operation.
    ///
    /// Returns `(old_val, new_val)`.
    pub(crate) fn atomic_rmw_op<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        ordering: AtomicRwOrd,
        (rmw_op, not): (mir::BinOp, bool),
        rhs_scalar: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Scalar)> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    /// Inform GenMC about an atomic `min` or `max` operation.
    ///
    /// Returns `(old_val, new_val)`.
    pub(crate) fn atomic_min_max_op<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        ordering: AtomicRwOrd,
        min: bool,
        is_signed: bool,
        rhs_scalar: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Scalar)> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    pub(crate) fn atomic_exchange<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        address: Size,
        size: Size,
        rhs_scalar: Scalar,
        ordering: AtomicRwOrd,
    ) -> InterpResult<'tcx, (Scalar, bool)> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

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
    ) -> InterpResult<'tcx, (Scalar, bool)> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    /// Inform GenMC about a non-atomic memory load
    ///
    /// NOTE: Unlike for *atomic* loads, we don't return a value here. Non-atomic values are still handled by Miri.
    pub(crate) fn memory_load<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        address: Size,
        size: Size,
    ) -> InterpResult<'tcx, ()> {
        todo!()
    }

    pub(crate) fn memory_store<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        address: Size,
        size: Size,
    ) -> InterpResult<'tcx, ()> {
        todo!()
    }

    /**** Memory (de)allocation ****/

    pub(crate) fn handle_alloc<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        size: Size,
        alignment: Align,
        memory_kind: MemoryKind,
    ) -> InterpResult<'tcx, u64> {
        todo!()
    }

    pub(crate) fn handle_dealloc<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        address: Size,
        size: Size,
        align: Align,
        kind: MemoryKind,
    ) -> InterpResult<'tcx, ()> {
        todo!()
    }

    /**** Thread management ****/

    pub(crate) fn handle_thread_create<'tcx>(
        &self,
        threads: &ThreadManager<'tcx>,
        new_thread_id: ThreadId,
    ) -> InterpResult<'tcx, ()> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    pub(crate) fn handle_thread_join<'tcx>(
        &self,
        active_thread_id: ThreadId,
        child_thread_id: ThreadId,
    ) -> InterpResult<'tcx, ()> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    pub(crate) fn handle_thread_stack_empty(&self, thread_id: ThreadId) {
        todo!()
    }

    pub(crate) fn handle_thread_finish<'tcx>(
        &self,
        threads: &ThreadManager<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    /**** Scheduling functionality ****/

    /// Ask for a scheduling decision. This should be called before every MIR instruction.
    ///
    /// GenMC may realize that the execution got stuck, then this function will return a `InterpErrorKind::MachineStop` with error kind `TerminationInfo::GenmcStuckExecution`).
    ///
    /// This is **not** an error by iself! Treat this as if the program ended normally: `handle_execution_end` should be called next, which will determine if were are any actual errors.
    pub(crate) fn schedule_thread<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    ) -> InterpResult<'tcx, ThreadId> {
        assert!(!self.allow_data_races.get());
        todo!()
    }

    /**** Blocking instructions ****/

    pub(crate) fn handle_verifier_assume<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        condition: bool,
    ) -> InterpResult<'tcx, ()> {
        if condition { interp_ok(()) } else { self.handle_user_block(machine) }
    }
}

impl VisitProvenance for GenmcCtx {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // We don't have any tags.
    }
}

impl GenmcCtx {
    fn handle_user_block<'tcx>(&self, machine: &MiriMachine<'tcx>) -> InterpResult<'tcx, ()> {
        todo!()
    }
}
