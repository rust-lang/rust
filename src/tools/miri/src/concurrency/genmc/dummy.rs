#![allow(unused)]

use rustc_abi::{Align, Size};
use rustc_const_eval::interpret::{InterpCx, InterpResult};
use rustc_middle::mir;

use crate::{
    AtomicFenceOrd, AtomicReadOrd, AtomicRwOrd, AtomicWriteOrd, MemoryKind, MiriConfig,
    MiriMachine, Scalar, ThreadId, ThreadManager, VisitProvenance, VisitWith,
};

#[derive(Debug)]
pub struct GenmcCtx {}

#[derive(Debug, Default, Clone)]
pub struct GenmcConfig {}

impl GenmcCtx {
    pub fn new(_miri_config: &MiriConfig) -> Self {
        unreachable!()
    }

    pub fn get_stuck_execution_count(&self) -> usize {
        unreachable!()
    }

    pub fn print_genmc_graph(&self) {
        unreachable!()
    }

    pub fn is_exploration_done(&self) -> bool {
        unreachable!()
    }

    /**** Memory access handling ****/

    pub(crate) fn handle_execution_start(&self) {
        unreachable!()
    }

    pub(crate) fn handle_execution_end<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    ) -> Result<(), String> {
        unreachable!()
    }

    pub(super) fn set_ongoing_action_data_race_free(&self, _enable: bool) {
        unreachable!()
    }

    //* might fails if there's a race, load might also not read anything (returns None) */
    pub(crate) fn atomic_load<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        _address: Size,
        _size: Size,
        _ordering: AtomicReadOrd,
        _old_val: Option<Scalar>,
    ) -> InterpResult<'tcx, Scalar> {
        unreachable!()
    }

    pub(crate) fn atomic_store<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        _address: Size,
        _size: Size,
        _value: Scalar,
        _ordering: AtomicWriteOrd,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }

    pub(crate) fn atomic_fence<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _ordering: AtomicFenceOrd,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }

    pub(crate) fn atomic_rmw_op<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        _address: Size,
        _size: Size,
        _ordering: AtomicRwOrd,
        (rmw_op, not): (mir::BinOp, bool),
        _rhs_scalar: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Scalar)> {
        unreachable!()
    }

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
        unreachable!()
    }

    pub(crate) fn atomic_exchange<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        _address: Size,
        _size: Size,
        _rhs_scalar: Scalar,
        _ordering: AtomicRwOrd,
    ) -> InterpResult<'tcx, (Scalar, bool)> {
        unreachable!()
    }

    pub(crate) fn atomic_compare_exchange<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        _address: Size,
        _size: Size,
        _expected_old_value: Scalar,
        _new_value: Scalar,
        _success: AtomicRwOrd,
        _fail: AtomicReadOrd,
        _can_fail_spuriously: bool,
    ) -> InterpResult<'tcx, (Scalar, bool)> {
        unreachable!()
    }

    pub(crate) fn memory_load<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _address: Size,
        _size: Size,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }

    pub(crate) fn memory_store<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _address: Size,
        _size: Size,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }

    /**** Memory (de)allocation ****/

    pub(crate) fn handle_alloc<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _size: Size,
        _alignment: Align,
        _memory_kind: MemoryKind,
    ) -> InterpResult<'tcx, u64> {
        unreachable!()
    }

    pub(crate) fn handle_dealloc<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _address: Size,
        _size: Size,
        _align: Align,
        _kind: MemoryKind,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }

    /**** Thread management ****/

    pub(crate) fn handle_thread_create<'tcx>(
        &self,
        _threads: &ThreadManager<'tcx>,
        _new_thread_id: ThreadId,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }

    pub(crate) fn handle_thread_join<'tcx>(
        &self,
        _active_thread_id: ThreadId,
        _child_thread_id: ThreadId,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }

    pub(crate) fn handle_thread_stack_empty(&self, _thread_id: ThreadId) {
        unreachable!()
    }

    pub(crate) fn handle_thread_finish<'tcx>(
        &self,
        _threads: &ThreadManager<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }

    /**** Scheduling functionality ****/

    pub(crate) fn schedule_thread<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    ) -> InterpResult<'tcx, ThreadId> {
        unreachable!()
    }

    /**** Blocking instructions ****/

    pub(crate) fn handle_verifier_assume<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _condition: bool,
    ) -> InterpResult<'tcx, ()> {
        unreachable!()
    }
}

impl VisitProvenance for GenmcCtx {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        unreachable!()
    }
}

impl GenmcConfig {
    pub fn parse_arg(
        _genmc_config: &mut Option<GenmcConfig>,
        trimmed_arg: &str,
    ) -> Result<(), String> {
        if cfg!(feature = "genmc") {
            Err(format!("GenMC is disabled in this build of Miri"))
        } else {
            Err(format!("GenMC is not supported on this target"))
        }
    }

    pub fn should_print_graph(&self, _rep: usize) -> bool {
        unreachable!()
    }
}
