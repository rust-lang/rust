use std::cell::RefCell;
use std::ptr;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::intern::Interned;
use rustc_data_structures::stable_hash::{
    StableHash, StableHashControls, StableHashCtxt, StableHasher,
};
use rustc_macros::StableHash;
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
pub type CanonicalResponse<'tcx> = ir::solve::CanonicalResponse<TyCtxt<'tcx>>;
pub type FetchEligibleAssocItemResponse<'tcx> =
    ir::solve::FetchEligibleAssocItemResponse<TyCtxt<'tcx>>;
pub type ComputeGoalFastPathOutcome<'tcx> = ir::solve::ComputeGoalFastPathOutcome<TyCtxt<'tcx>>;
pub type GoalStalledOn<'tcx> = ir::solve::GoalStalledOn<TyCtxt<'tcx>>;
pub type GoalStalledOnOpaques<'tcx> = ir::solve::GoalStalledOnOpaques<TyCtxt<'tcx>>;
pub type SucceededInErased<'tcx> = ir::solve::SucceededInErased<TyCtxt<'tcx>>;

pub type PredefinedOpaques<'tcx> = &'tcx ty::List<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)>;
pub type TraitEvidences<'tcx> = &'tcx ty::List<TraitEvidence<'tcx>>;

/// Interned source contract rebased into one dependent binder telescope.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, StableHash)]
pub struct BoundRequiredContract<'tcx>(
    pub(crate) Interned<'tcx, ty::BoundRequiredContractData<TyCtxt<'tcx>>>,
);

impl<'tcx> std::ops::Deref for BoundRequiredContract<'tcx> {
    type Target = ty::BoundRequiredContractData<TyCtxt<'tcx>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for BoundRequiredContract<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_bound_required_contract(self)
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        folder.fold_bound_required_contract(self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for BoundRequiredContract<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        (**self).visit_with(visitor)
    }
}

/// Interned compiler-internal trait evidence value.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TraitEvidence<'tcx>(pub(crate) Interned<'tcx, TraitEvidenceData<TyCtxt<'tcx>>>);

impl StableHash for TraitEvidence<'_> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        // Like interned lists, shared proof recipes must not be expanded into a tree.
        thread_local! {
            static CACHE: RefCell<FxHashMap<(*const (), StableHashControls), Fingerprint>> =
                RefCell::new(Default::default());
        }

        let hash = CACHE.with(|cache| {
            let key = (ptr::from_ref(self.0.0).cast::<()>(), hcx.stable_hash_controls());
            if let Some(&hash) = cache.borrow().get(&key) {
                return hash;
            }

            let mut hasher = StableHasher::new();
            self.0.0.stable_hash(hcx, &mut hasher);
            let hash: Fingerprint = hasher.finish();
            cache.borrow_mut().insert(key, hash);
            hash
        });

        hash.stable_hash(hcx, hasher);
    }
}

impl<'tcx> std::ops::Deref for TraitEvidence<'tcx> {
    type Target = TraitEvidenceData<TyCtxt<'tcx>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for TraitEvidence<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_trait_evidence(self)
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        folder.fold_trait_evidence(self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for TraitEvidence<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_trait_evidence(*self)
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for TraitEvidences<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        ty::util::try_fold_list(self, folder, |tcx, values| tcx.mk_trait_evidences(values))
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        ty::util::fold_list(self, folder, |tcx, values| tcx.mk_trait_evidences(values))
    }
}

/// Interned payload shared by evidence-aware type and const projections.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, StableHash)]
pub struct EvidenceProjection<'tcx>(
    pub(crate) Interned<'tcx, ty::EvidenceProjectionData<TyCtxt<'tcx>>>,
);

impl<'tcx> std::ops::Deref for EvidenceProjection<'tcx> {
    type Target = ty::EvidenceProjectionData<TyCtxt<'tcx>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for EvidenceProjection<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_evidence_projection(self)
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        folder.fold_evidence_projection(self)
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for EvidenceProjection<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_evidence_projection(*self)
    }
}

// Interning CanonicalInput drastically reduces max memory usage when compiling a crate that has
// trait solver recursion depth overflows with next-solver deduplicating individual inputs.
// This mostly fixes #161748 where it reduced the memory usage for compiling bevy_render from
// ~14GiB to ~4GiB
// Main improved types:
//   - rustc_type_ir::search_graph::GlobalCache
//   - rustc_type_ir::search_graph::NestedGoals
#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, StableHash)]
pub struct CanonicalInput<'tcx>(pub(crate) Interned<'tcx, CanonicalInputData<TyCtxt<'tcx>>>);

impl<'tcx> std::ops::Deref for CanonicalInput<'tcx> {
    type Target = CanonicalInputData<TyCtxt<'tcx>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
