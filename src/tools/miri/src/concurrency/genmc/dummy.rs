use rustc_abi::{Align, Size};
use rustc_const_eval::interpret::{AllocId, InterpCx, InterpResult};
use rustc_middle::ty::TyCtxt;

pub use self::intercept::EvalContextExt as GenmcEvalContextExt;
pub use self::run::run_genmc_mode;
use crate::intrinsics::AtomicRmwOp;
use crate::{
    AtomicFenceOrd, AtomicReadOrd, AtomicRwOrd, AtomicWriteOrd, MemoryKind, MiriMachine, OpTy,
    Scalar, ThreadId, ThreadManager, VisitProvenance, VisitWith,
};

#[derive(Clone, Copy, Debug)]
pub enum ExitType {
    MainThreadFinish,
    ExitCalled,
}

#[derive(Debug)]
pub struct GenmcCtx {}

#[derive(Debug, Default, Clone)]
pub struct GenmcConfig {}

mod run {
    use std::rc::Rc;

    use rustc_middle::ty::TyCtxt;

    use crate::{GenmcCtx, MiriConfig};

    pub fn run_genmc_mode<'tcx>(
        _config: &MiriConfig,
        _eval_entry: impl Fn(Rc<GenmcCtx>) -> Option<i32>,
        _tcx: TyCtxt<'tcx>,
    ) -> Option<i32> {
        unreachable!();
    }
}

mod intercept {
    use super::*;

    impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
    pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
        fn genmc_intercept_function(
            &mut self,
            _instance: rustc_middle::ty::Instance<'tcx>,
            _args: &[rustc_const_eval::interpret::FnArg<'tcx, crate::Provenance>],
            _dest: &crate::PlaceTy<'tcx>,
        ) -> InterpResult<'tcx, bool> {
            unreachable!()
        }

        fn handle_genmc_verifier_assume(&mut self, _condition: &OpTy<'tcx>) -> InterpResult<'tcx> {
            unreachable!();
        }
    }
}

impl GenmcCtx {
    // We don't provide the `new` function in the dummy module.

    pub(crate) fn schedule_thread<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    ) -> InterpResult<'tcx, Option<ThreadId>> {
        unreachable!()
    }

    /**** Memory access handling ****/

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
        _old_value: Option<Scalar>,
        _ordering: AtomicWriteOrd,
    ) -> InterpResult<'tcx, bool> {
        unreachable!()
    }

    pub(crate) fn atomic_fence<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _ordering: AtomicFenceOrd,
    ) -> InterpResult<'tcx> {
        unreachable!()
    }

    pub(crate) fn atomic_rmw_op<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        _address: Size,
        _size: Size,
        _atomic_op: AtomicRmwOp,
        _is_signed: bool,
        _ordering: AtomicRwOrd,
        _rhs_scalar: Scalar,
        _old_value: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Option<Scalar>)> {
        unreachable!()
    }

    pub(crate) fn atomic_exchange<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        _address: Size,
        _size: Size,
        _rhs_scalar: Scalar,
        _ordering: AtomicRwOrd,
        _old_value: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Option<Scalar>)> {
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
        _old_value: Scalar,
    ) -> InterpResult<'tcx, (Scalar, Option<Scalar>, bool)> {
        unreachable!()
    }

    pub(crate) fn memory_load<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _address: Size,
        _size: Size,
    ) -> InterpResult<'tcx> {
        unreachable!()
    }

    pub(crate) fn memory_store<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _address: Size,
        _size: Size,
    ) -> InterpResult<'tcx> {
        unreachable!()
    }

    /**** Memory (de)allocation ****/

    pub(crate) fn handle_alloc<'tcx>(
        &self,
        _ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
        _alloc_id: AllocId,
        _size: Size,
        _alignment: Align,
        _memory_kind: MemoryKind,
    ) -> InterpResult<'tcx, u64> {
        unreachable!()
    }

    pub(crate) fn handle_dealloc<'tcx>(
        &self,
        _machine: &MiriMachine<'tcx>,
        _alloc_id: AllocId,
        _address: Size,
        _kind: MemoryKind,
    ) -> InterpResult<'tcx> {
        unreachable!()
    }

    /**** Thread management ****/

    pub(crate) fn handle_thread_create<'tcx>(
        &self,
        _threads: &ThreadManager<'tcx>,
        _start_routine: crate::Pointer,
        _func_arg: &crate::ImmTy<'tcx>,
        _new_thread_id: ThreadId,
    ) -> InterpResult<'tcx> {
        unreachable!()
    }

    pub(crate) fn handle_thread_join<'tcx>(
        &self,
        _active_thread_id: ThreadId,
        _child_thread_id: ThreadId,
    ) -> InterpResult<'tcx> {
        unreachable!()
    }

    pub(crate) fn handle_thread_finish<'tcx>(&self, _threads: &ThreadManager<'tcx>) {
        unreachable!()
    }

    pub(crate) fn handle_exit<'tcx>(
        &self,
        _thread: ThreadId,
        _exit_code: i32,
        _exit_type: ExitType,
    ) -> InterpResult<'tcx> {
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
        _trimmed_arg: &str,
    ) -> Result<(), String> {
        if cfg!(feature = "genmc") {
            Err(format!("GenMC is disabled in this build of Miri"))
        } else {
            Err(format!("GenMC is not supported on this target"))
        }
    }

    pub fn validate(
        _miri_config: &mut crate::MiriConfig,
        _tcx: TyCtxt<'_>,
    ) -> Result<(), &'static str> {
        Ok(())
    }
}
