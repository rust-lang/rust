use rustc_data_structures::intern::Interned;
use rustc_macros::StableHash;
use rustc_type_ir as ir;
pub use rustc_type_ir::solve::*;

use crate::ty::{self, Ty, TyCtxt, TypeVisitable, TypeVisitor, try_visit};

pub type Goal<'tcx, P> = ir::solve::Goal<TyCtxt<'tcx>, P>;
pub type QueryInput<'tcx, P> = ir::solve::QueryInput<TyCtxt<'tcx>, P>;
pub type QueryResult<'tcx> = ir::solve::QueryResult<TyCtxt<'tcx>>;
pub type CandidateSource<'tcx> = ir::solve::CandidateSource<TyCtxt<'tcx>>;
pub type CanonicalInput<'tcx, P = ty::Predicate<'tcx>> = ir::solve::CanonicalInput<TyCtxt<'tcx>, P>;
pub type CanonicalResponse<'tcx> = ir::solve::CanonicalResponse<TyCtxt<'tcx>>;
pub type FetchEligibleAssocItemResponse<'tcx> =
    ir::solve::FetchEligibleAssocItemResponse<TyCtxt<'tcx>>;
pub type ComputeGoalFastPathOutcome<'tcx> = ir::solve::ComputeGoalFastPathOutcome<TyCtxt<'tcx>>;
pub type GoalStalledOn<'tcx> = ir::solve::GoalStalledOn<TyCtxt<'tcx>>;
pub type GoalStalledOnOpaques<'tcx> = ir::solve::GoalStalledOnOpaques<TyCtxt<'tcx>>;
pub type SucceededInErased<'tcx> = ir::solve::SucceededInErased<TyCtxt<'tcx>>;

pub type PredefinedOpaques<'tcx> = &'tcx ty::List<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)>;

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, StableHash)]
pub struct ExternalConstraints<'tcx>(
    pub(crate) Interned<'tcx, ExternalConstraintsData<TyCtxt<'tcx>>>,
);

impl<'tcx> std::ops::Deref for ExternalConstraints<'tcx> {
    type Target = ExternalConstraintsData<TyCtxt<'tcx>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        let ExternalConstraintsData {
            region_constraints,
            opaque_types,
            normalization_nested_goals,
        } = &**self;

        try_visit!(region_constraints.visit_with(visitor));
        try_visit!(opaque_types.visit_with(visitor));
        normalization_nested_goals.visit_with(visitor)
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(GoalStalledOn<'_>, 56);
    // tidy-alphabetical-end
}
