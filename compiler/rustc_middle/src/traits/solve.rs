use rustc_data_structures::intern::Interned;
use rustc_macros::HashStable;
use rustc_type_ir as ir;
pub use rustc_type_ir::solve::*;

use crate::ty::{
    self, FallibleTypeFolder, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor,
    try_visit,
};

pub type Goal<'tcx, P> = ir::solve::Goal<TyCtxt<'tcx>, P>;
pub type QueryInput<'tcx, P> = ir::solve::QueryInput<TyCtxt<'tcx>, P>;
pub type QueryResult<'tcx> = ir::solve::QueryResult<TyCtxt<'tcx>>;
pub type CandidateSource<'tcx> = ir::solve::CandidateSource<TyCtxt<'tcx>>;
pub type CanonicalInput<'tcx, P = ty::Predicate<'tcx>> = ir::solve::CanonicalInput<TyCtxt<'tcx>, P>;
pub type CanonicalResponse<'tcx> = ir::solve::CanonicalResponse<TyCtxt<'tcx>>;

pub type PredefinedOpaques<'tcx> = &'tcx ty::List<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)>;

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, HashStable)]
pub struct ExternalConstraints<'tcx>(
    pub(crate) Interned<'tcx, ExternalConstraintsData<TyCtxt<'tcx>>>,
);

impl<'tcx> std::ops::Deref for ExternalConstraints<'tcx> {
    type Target = ExternalConstraintsData<TyCtxt<'tcx>>;

    fn deref(&self) -> &Self::Target {
        &self.0
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
        // Perf testing has found that this check is slightly faster than
        // folding and re-interning an empty `ExternalConstraintsData`.
        // See: <https://github.com/rust-lang/rust/pull/142430>.
        if self.is_empty() {
            return Ok(self);
        }

        Ok(FallibleTypeFolder::cx(folder).mk_external_constraints(ExternalConstraintsData {
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
        // Perf testing has found that this check is slightly faster than
        // folding and re-interning an empty `ExternalConstraintsData`.
        // See: <https://github.com/rust-lang/rust/pull/142430>.
        if self.is_empty() {
            return self;
        }

        TypeFolder::cx(folder).mk_external_constraints(ExternalConstraintsData {
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
