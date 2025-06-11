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

/// This struct is needed to enforce `#[must_use]` on [tracing::span::EnteredSpan]
/// while wrapping them in an `Option`.
#[must_use]
pub enum MaybeEnteredSpan {
    Some(tracing::span::EnteredSpan),
    None,
}

#[macro_export]
macro_rules! enter_trace_span {
    ($machine:ident, $($tt:tt)*) => {
        if $machine::TRACING_ENABLED {
            $crate::interpret::util::MaybeEnteredSpan::Some(tracing::info_span!($($tt)*).entered())
        } else {
            $crate::interpret::util::MaybeEnteredSpan::None
        }
    }
}
