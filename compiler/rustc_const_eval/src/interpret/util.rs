use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{AllocInit, Allocation, InterpResult, Pointer};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{TyCtxt, TypeVisitable, TypeVisitableExt};
use tracing::debug;

use super::{InterpCx, MPlaceTy, MemoryKind, interp_ok, throw_inval};
use crate::const_eval::{CompileTimeInterpCx, CompileTimeMachine, InterpretationResult};

/// Checks whether a type contains generic parameters which must be instantiated.
///
/// In case it does, returns a `TooGeneric` const eval error.
pub(crate) fn ensure_monomorphic_enough<'tcx, T>(_tcx: TyCtxt<'tcx>, ty: T) -> InterpResult<'tcx>
where
    T: TypeVisitable<TyCtxt<'tcx>>,
{
    debug!("ensure_monomorphic_enough: ty={:?}", ty);
    if ty.has_param() {
        throw_inval!(TooGeneric);
    }
    interp_ok(())
}

impl<'tcx> InterpretationResult<'tcx> for mir::interpret::ConstAllocation<'tcx> {
    fn make_result(
        mplace: MPlaceTy<'tcx>,
        ecx: &mut InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    ) -> Self {
        let alloc_id = mplace.ptr().provenance.unwrap().alloc_id();
        let alloc = ecx.memory.alloc_map.swap_remove(&alloc_id).unwrap().1;
        ecx.tcx.mk_const_alloc(alloc)
    }
}

pub(crate) fn create_static_alloc<'tcx>(
    ecx: &mut CompileTimeInterpCx<'tcx>,
    static_def_id: LocalDefId,
    layout: TyAndLayout<'tcx>,
) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
    let alloc = Allocation::try_new(layout.size, layout.align.abi, AllocInit::Uninit, ())?;
    let alloc_id = ecx.tcx.reserve_and_set_static_alloc(static_def_id.into());
    assert_eq!(ecx.machine.static_root_ids, None);
    ecx.machine.static_root_ids = Some((alloc_id, static_def_id));
    assert!(ecx.memory.alloc_map.insert(alloc_id, (MemoryKind::Stack, alloc)).is_none());
    interp_ok(ecx.ptr_to_mplace(Pointer::from(alloc_id).into(), layout))
}

/// A marker trait returned by [crate::interpret::Machine::enter_trace_span], identifying either a
/// real [tracing::span::EnteredSpan] in case tracing is enabled, or the dummy type `()` when
/// tracing is disabled.
pub trait EnteredTraceSpan {}
impl EnteredTraceSpan for () {}
impl EnteredTraceSpan for tracing::span::EnteredSpan {}

/// Shortand for calling [crate::interpret::Machine::enter_trace_span] on a [tracing::info_span].
/// This is supposed to be compiled out when [crate::interpret::Machine::enter_trace_span] has the
/// default implementation (i.e. when it does not actually enter the span but instead returns `()`).
/// Note: the result of this macro **must be used** because the span is exited when it's dropped.
#[macro_export]
macro_rules! enter_trace_span {
    ($machine:ident, $($tt:tt)*) => {
        $machine::enter_trace_span(tracing::info_span!($($tt)*))
    }
}
