use super::VariableLengths;
use crate::infer::InferCtxt;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{TypeSuperVisitable, TypeVisitor};
use std::ops::ControlFlow;

/// Check for leaking inference variables and placeholders
/// from snapshot. This is only used if `debug_assertions`
/// are enabled.
pub struct HasSnapshotLeaksVisitor {
    universe: ty::UniverseIndex,
    variable_lengths: VariableLengths,
}
impl HasSnapshotLeaksVisitor {
    pub fn new<'tcx>(infcx: &InferCtxt<'tcx>) -> Self {
        HasSnapshotLeaksVisitor {
            universe: infcx.universe(),
            variable_lengths: infcx.variable_lengths(),
        }
    }
}

fn continue_if(b: bool) -> ControlFlow<()> {
    if b { ControlFlow::Continue(()) } else { ControlFlow::Break(()) }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for HasSnapshotLeaksVisitor {
    type Result = ControlFlow<()>;

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> Self::Result {
        match r.kind() {
            ty::ReVar(var) => continue_if(var.as_usize() < self.variable_lengths.region_vars),
            ty::RePlaceholder(p) => continue_if(self.universe.can_name(p.universe)),
            ty::ReEarlyParam(_)
            | ty::ReBound(_, _)
            | ty::ReLateParam(_)
            | ty::ReStatic
            | ty::ReErased
            | ty::ReError(_) => ControlFlow::Continue(()),
        }
    }
    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        match t.kind() {
            ty::Infer(ty::TyVar(var)) => {
                continue_if(var.as_usize() < self.variable_lengths.type_vars)
            }
            ty::Infer(ty::IntVar(var)) => {
                continue_if(var.as_usize() < self.variable_lengths.int_vars)
            }
            ty::Infer(ty::FloatVar(var)) => {
                continue_if(var.as_usize() < self.variable_lengths.float_vars)
            }
            ty::Placeholder(p) => continue_if(self.universe.can_name(p.universe)),
            ty::Infer(ty::FreshTy(..) | ty::FreshIntTy(..) | ty::FreshFloatTy(..))
            | ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_)
            | ty::Dynamic(_, _, _)
            | ty::Closure(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(_, _)
            | ty::Never
            | ty::Tuple(_)
            | ty::Alias(_, _)
            | ty::Param(_)
            | ty::Bound(_, _)
            | ty::Error(_) => t.super_visit_with(self),
        }
    }
    fn visit_const(&mut self, c: ty::Const<'tcx>) -> Self::Result {
        match c.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(var)) => {
                continue_if(var.as_usize() < self.variable_lengths.const_vars)
            }
            // FIXME(const_trait_impl): need to handle effect vars here and in `fudge_inference_if_ok`.
            ty::ConstKind::Infer(ty::InferConst::EffectVar(_)) => ControlFlow::Continue(()),
            ty::ConstKind::Placeholder(p) => continue_if(self.universe.can_name(p.universe)),
            ty::ConstKind::Infer(ty::InferConst::Fresh(_))
            | ty::ConstKind::Param(_)
            | ty::ConstKind::Bound(_, _)
            | ty::ConstKind::Unevaluated(_)
            | ty::ConstKind::Value(_)
            | ty::ConstKind::Expr(_)
            | ty::ConstKind::Error(_) => c.super_visit_with(self),
        }
    }
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! type_foldable_verify_no_snapshot_leaks {
    ($tcx:lifetime, $t:ty) => {
        const _: () = {
            use rustc_middle::ty::TypeVisitable;
            use $crate::infer::snapshot::check_leaks::HasSnapshotLeaksVisitor;
            use $crate::infer::InferCtxt;
            impl<$tcx> $crate::infer::snapshot::NoSnapshotLeaks<$tcx> for $t {
                type StartData = HasSnapshotLeaksVisitor;
                type EndData = ($t, HasSnapshotLeaksVisitor);
                fn snapshot_start_data(infcx: &$crate::infer::InferCtxt<$tcx>) -> Self::StartData {
                    HasSnapshotLeaksVisitor::new(infcx)
                }
                fn end_of_snapshot(
                    _: &InferCtxt<'tcx>,
                    value: $t,
                    visitor: Self::StartData,
                ) -> Self::EndData {
                    (value, visitor)
                }
                fn avoid_leaks(_: &InferCtxt<$tcx>, (value, mut visitor): Self::EndData) -> Self {
                    if value.visit_with(&mut visitor).is_break() {
                        bug!("leaking vars from snapshot: {value:?}");
                    }

                    value
                }
            }
        };
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! type_foldable_verify_no_snapshot_leaks {
    ($tcx:lifetime, $t:ty) => {
        trivial_no_snapshot_leaks!($tcx, $t);
    };
}
