use rustc_ast_ir::try_visit;
use rustc_data_structures::intern::Interned;
use rustc_macros::{HashStable, TypeFoldable, TypeVisitable};
use rustc_next_trait_solver as ir;
pub use rustc_next_trait_solver::solve::*;

use crate::infer::canonical::QueryRegionConstraints;
use crate::ty::{
    self, FallibleTypeFolder, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor,
};

mod cache;

pub use cache::{CacheData, EvaluationCache};

pub type Goal<'tcx, P> = ir::solve::Goal<TyCtxt<'tcx>, P>;
pub type QueryInput<'tcx, P> = ir::solve::QueryInput<TyCtxt<'tcx>, P>;
pub type QueryResult<'tcx> = ir::solve::QueryResult<TyCtxt<'tcx>>;
pub type CandidateSource<'tcx> = ir::solve::CandidateSource<TyCtxt<'tcx>>;
pub type CanonicalInput<'tcx, P = ty::Predicate<'tcx>> = ir::solve::CanonicalInput<TyCtxt<'tcx>, P>;
pub type CanonicalResponse<'tcx> = ir::solve::CanonicalResponse<TyCtxt<'tcx>>;

/// Additional constraints returned on success.
#[derive(Debug, PartialEq, Eq, Clone, Hash, HashStable, Default)]
pub struct PredefinedOpaquesData<'tcx> {
    pub opaque_types: Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)>,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, HashStable)]
pub struct PredefinedOpaques<'tcx>(pub(crate) Interned<'tcx, PredefinedOpaquesData<'tcx>>);

impl<'tcx> std::ops::Deref for PredefinedOpaques<'tcx> {
    type Target = PredefinedOpaquesData<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, HashStable)]
pub struct ExternalConstraints<'tcx>(pub(crate) Interned<'tcx, ExternalConstraintsData<'tcx>>);

impl<'tcx> std::ops::Deref for ExternalConstraints<'tcx> {
    type Target = ExternalConstraintsData<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Additional constraints returned on success.
#[derive(Debug, PartialEq, Eq, Clone, Hash, HashStable, Default, TypeVisitable, TypeFoldable)]
pub struct ExternalConstraintsData<'tcx> {
    // FIXME: implement this.
    pub region_constraints: QueryRegionConstraints<'tcx>,
    pub opaque_types: Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)>,
    pub normalization_nested_goals: NestedNormalizationGoals<'tcx>,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, HashStable, Default, TypeVisitable, TypeFoldable)]
pub struct NestedNormalizationGoals<'tcx>(pub Vec<(GoalSource, Goal<'tcx, ty::Predicate<'tcx>>)>);
impl<'tcx> NestedNormalizationGoals<'tcx> {
    pub fn empty() -> Self {
        NestedNormalizationGoals(vec![])
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// FIXME: Having to clone `region_constraints` for folding feels bad and
// probably isn't great wrt performance.
//
// Not sure how to fix this, maybe we should also intern `opaque_types` and
// `region_constraints` here or something.
impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(FallibleTypeFolder::interner(folder).mk_external_constraints(ExternalConstraintsData {
            region_constraints: self.region_constraints.clone().try_fold_with(folder)?,
            opaque_types: self
                .opaque_types
                .iter()
                .map(|opaque| opaque.try_fold_with(folder))
                .collect::<Result<_, F::Error>>()?,
            normalization_nested_goals: self
                .normalization_nested_goals
                .clone()
                .try_fold_with(folder)?,
        }))
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        TypeFolder::interner(folder).mk_external_constraints(ExternalConstraintsData {
            region_constraints: self.region_constraints.clone().fold_with(folder),
            opaque_types: self.opaque_types.iter().map(|opaque| opaque.fold_with(folder)).collect(),
            normalization_nested_goals: self.normalization_nested_goals.clone().fold_with(folder),
        })
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        try_visit!(self.region_constraints.visit_with(visitor));
        try_visit!(self.opaque_types.visit_with(visitor));
        self.normalization_nested_goals.visit_with(visitor)
    }
}

// FIXME: Having to clone `region_constraints` for folding feels bad and
// probably isn't great wrt performance.
//
// Not sure how to fix this, maybe we should also intern `opaque_types` and
// `region_constraints` here or something.
impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for PredefinedOpaques<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(FallibleTypeFolder::interner(folder).mk_predefined_opaques_in_body(
            PredefinedOpaquesData {
                opaque_types: self
                    .opaque_types
                    .iter()
                    .map(|opaque| opaque.try_fold_with(folder))
                    .collect::<Result<_, F::Error>>()?,
            },
        ))
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        TypeFolder::interner(folder).mk_predefined_opaques_in_body(PredefinedOpaquesData {
            opaque_types: self.opaque_types.iter().map(|opaque| opaque.fold_with(folder)).collect(),
        })
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for PredefinedOpaques<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        self.opaque_types.visit_with(visitor)
    }
}
