use std::fmt;
use std::ops::ControlFlow;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::intern::Interned;
use rustc_errors::ErrorGuaranteed;
use rustc_macros::StableHash;
use rustc_type_ir as ir;
pub use rustc_type_ir::solve::*;

use crate::infer::canonical::CanonicalEvidenceVarKinds;
use crate::ty::{
    self, FallibleTypeFolder, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor, try_visit,
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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, StableHash)]
pub struct TraitEvidence<'tcx>(pub(crate) Interned<'tcx, TraitEvidenceData<TyCtxt<'tcx>>>);

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

/// The boundary at which evidence-indexed projections are being validated.
///
/// Generic and runtime MIR may carry a closed selected recipe. Fully monomorphized
/// backend values may not carry a projection at all: normalization must consume it
/// before layout, symbol selection, or instruction lowering can observe the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceValidationBoundary {
    Persistence,
    RuntimeMir,
    Codegen,
}

/// Why an evidence-bearing value is not safe to persist or pass to a later compiler phase.
///
/// This intentionally contains no inference-context-local payload. Apart from making the
/// diagnostic stable, that prevents the validation failure itself from accidentally extending
/// the lifetime of an [`EvidenceVid`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceValidationError {
    UnnormalizedProjection,
    EvidenceInferenceVariable,
    EvidenceBoundVariable,
    EvidencePlaceholder,
    EvidenceError,
    TypeInferenceVariable,
    TypePlaceholder,
    EscapingBoundVariable,
    EscapingCanonicalVariable,
    CanonicalVariableOutOfRange,
    TypeError,
    EmptyRecipe,
    RecipeRootOutOfBounds,
    RecipeEdgeOutOfBounds,
    RecipeCycle,
    UnreachableRecipeNode,
    InvalidUniqueRecipe,
    UnguardedRecursiveRecipe,
    ErrorCandidate,
    NonProofCandidate,
    NestedCanonicalArityMismatch,
    InvalidCanonicalUniverse,
    CanonicalEvidencePredicateMismatch,
    RecipeCycleIndexOutOfBounds,
    DuplicateRecipeCycleKey,
    UnreferencedRecipeCycleKey,
    MalformedProofCycle,
    NonProductiveProofCycle,
    ProofCycleRootMismatch,
    RepeatedProofCycleParticipant,
    CyclicEvidenceHandles,
    InvalidDynProjectionOperation,
}

impl fmt::Display for EvidenceValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            EvidenceValidationError::UnnormalizedProjection => {
                "an evidence projection reached fully monomorphized codegen"
            }
            EvidenceValidationError::EvidenceInferenceVariable => {
                "an unresolved evidence inference variable escaped"
            }
            EvidenceValidationError::EvidenceBoundVariable => {
                "an evidence bound variable escaped its telescope"
            }
            EvidenceValidationError::EvidencePlaceholder => {
                "an evidence placeholder escaped its universe"
            }
            EvidenceValidationError::EvidenceError => {
                "error-recovery evidence reached a persistence or backend boundary"
            }
            EvidenceValidationError::TypeInferenceVariable => {
                "an inference variable occurs in a persisted evidence recipe"
            }
            EvidenceValidationError::TypePlaceholder => {
                "a placeholder occurs outside its canonical evidence scope"
            }
            EvidenceValidationError::EscapingBoundVariable => {
                "a bound variable escapes the evidence recipe containing it"
            }
            EvidenceValidationError::EscapingCanonicalVariable => {
                "a canonical variable occurs outside a canonical evidence response"
            }
            EvidenceValidationError::CanonicalVariableOutOfRange => {
                "a canonical variable is outside the response variable list"
            }
            EvidenceValidationError::TypeError => {
                "an error type, const, or region occurs in an evidence recipe"
            }
            EvidenceValidationError::EmptyRecipe => "the evidence recipe is empty",
            EvidenceValidationError::RecipeRootOutOfBounds => {
                "the evidence recipe root is out of bounds"
            }
            EvidenceValidationError::RecipeEdgeOutOfBounds => {
                "an evidence recipe edge is out of bounds"
            }
            EvidenceValidationError::RecipeCycle => {
                "the same-scope evidence recipe DAG contains a cycle"
            }
            EvidenceValidationError::UnreachableRecipeNode => {
                "the evidence recipe contains an unreachable node"
            }
            EvidenceValidationError::InvalidUniqueRecipe => {
                "a coherence-unique evidence wrapper is malformed"
            }
            EvidenceValidationError::UnguardedRecursiveRecipe => {
                "a recursive evidence leaf is not guarded by a completed proof"
            }
            EvidenceValidationError::ErrorCandidate => {
                "an error-recovery candidate was used as persistent evidence"
            }
            EvidenceValidationError::NonProofCandidate => {
                "a coherence-only candidate was used as persistent evidence"
            }
            EvidenceValidationError::NestedCanonicalArityMismatch => {
                "a nested canonical evidence mapping has the wrong arity"
            }
            EvidenceValidationError::InvalidCanonicalUniverse => {
                "a canonical evidence variable refers to an unavailable universe"
            }
            EvidenceValidationError::CanonicalEvidencePredicateMismatch => {
                "a canonical evidence variable proves a different trait predicate"
            }
            EvidenceValidationError::RecipeCycleIndexOutOfBounds => {
                "a recursive evidence node refers to a missing proof cycle"
            }
            EvidenceValidationError::DuplicateRecipeCycleKey => {
                "a proof cycle key is referenced by more than one recursive node"
            }
            EvidenceValidationError::UnreferencedRecipeCycleKey => {
                "the evidence recipe contains an unreferenced proof cycle key"
            }
            EvidenceValidationError::MalformedProofCycle => {
                "a proof cycle is empty or does not contain one edge per participant"
            }
            EvidenceValidationError::NonProductiveProofCycle => {
                "a recursive evidence cycle is not coinductively productive"
            }
            EvidenceValidationError::ProofCycleRootMismatch => {
                "a recursive evidence predicate does not match its proof cycle head"
            }
            EvidenceValidationError::RepeatedProofCycleParticipant => {
                "a proof cycle repeats a participant before its closing edge"
            }
            EvidenceValidationError::CyclicEvidenceHandles => {
                "interned evidence handles contain an unguarded reference cycle"
            }
            EvidenceValidationError::InvalidDynProjectionOperation => {
                "a dyn projection operation does not match its selected object proof"
            }
        };
        f.write_str(message)
    }
}

/// Validate all evidence projections reachable from `value`.
///
/// Unlike flag-only checks, this traversal understands binders and independently canonicalized
/// nested proof responses. It therefore rejects only *escaping* ordinary bound variables while
/// allowing variables owned by a binder inside the recipe. Evidence telescope variables are
/// conservatively rejected until the generalized `EvidenceArgs` representation can prove their
/// scope at this boundary.
pub fn validate_evidence_projections<'tcx, T>(
    value: &T,
    boundary: EvidenceValidationBoundary,
) -> Result<(), EvidenceValidationError>
where
    T: TypeVisitable<TyCtxt<'tcx>> + ?Sized,
{
    if !value.has_evidence_projections() {
        return Ok(());
    }

    let mut validator = EvidenceValidator::new(boundary, false, false);
    match value.visit_with(&mut validator) {
        ControlFlow::Continue(()) => Ok(()),
        ControlFlow::Break(error) => Err(error),
    }
}

/// Validate one interned evidence value before it is encoded on its own.
///
/// The low-level codec is invoked again for evidence nested in a canonical response and cannot
/// observe that response's outer variable list. `allow_unknown_outer_scope` keeps ordinary
/// canonical/bound type variables opaque at this fragment boundary; the containing projection's
/// full validation remains responsible for checking their scope. Evidence variables themselves
/// are never accepted here because they cannot be proven closed without `EvidenceArgs`.
pub fn validate_trait_evidence_for_persistence(
    evidence: TraitEvidence<'_>,
) -> Result<(), EvidenceValidationError> {
    let mut validator = EvidenceValidator::new(EvidenceValidationBoundary::Persistence, true, true);
    validator.validate_evidence(evidence, true)
}

/// Validate one encoded projection root. A recursive leaf is only legal below another selected
/// recipe, never as the evidence of a projection itself.
pub fn validate_evidence_projection_for_persistence(
    projection: EvidenceProjection<'_>,
) -> Result<(), EvidenceValidationError> {
    let mut validator =
        EvidenceValidator::new(EvidenceValidationBoundary::Persistence, true, false);
    validator.validate_evidence(projection.evidence, false)
}

/// Attach one call-operation projection instantiation to the selected Dyn node
/// without rebuilding or flattening its proof DAG. Only a direct Dyn proof (or
/// its same-scope Unique wrapper) is eligible here.
pub fn with_dyn_projection_operation<'tcx>(
    tcx: TyCtxt<'tcx>,
    evidence: TraitEvidence<'tcx>,
    operation: DynProjectionOperation<TyCtxt<'tcx>>,
) -> Option<TraitEvidence<'tcx>> {
    let TraitEvidenceKind::Selected(original) = &evidence.kind else { return None };
    let mut recipe = original.clone();
    let mut index = usize::try_from(recipe.root).ok()?;
    for _ in 0..=recipe.nodes.len() {
        let node = recipe.nodes.get(index)?;
        if matches!(node.source, CandidateEvidenceSource::Unique(_)) {
            index = usize::try_from(*node.nested.first()?).ok()?;
            continue;
        }
        break;
    }
    let node = recipe.nodes.get(index)?;
    let CandidateEvidenceSource::Dyn { object_bound, instantiation, vtable_slot, operation: None } =
        node.source
    else {
        return None;
    };
    if node.trait_ref != evidence.trait_ref
        || !dyn_projection_operation_matches(
            tcx,
            node.trait_ref,
            object_bound,
            instantiation,
            operation,
            evidence,
        )
    {
        return None;
    }
    recipe.nodes[index].source = CandidateEvidenceSource::Dyn {
        object_bound,
        instantiation,
        vtable_slot,
        operation: Some(operation),
    };
    recipe.assert_well_formed();
    Some(tcx.mk_trait_evidence(recipe))
}

fn dyn_projection_operation_matches<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
    object_bound: ty::Clause<'tcx>,
    object_instantiation: Option<ty::GenericArgsRef<'tcx>>,
    operation: DynProjectionOperation<TyCtxt<'tcx>>,
    evidence: TraitEvidence<'tcx>,
) -> bool {
    let Some(object_trait) = object_bound.as_trait_clause() else {
        return false;
    };
    if object_trait.evidence_bound_vars().next().is_some() {
        return false;
    }
    let object_ordinary: Vec<_> = object_trait.ordinary_bound_vars().collect();
    let object_trait_ref = if let Some(args) = object_instantiation {
        if object_ordinary.len() != args.len()
            || object_ordinary.iter().zip(args.iter()).any(|(kind, arg)| {
                !matches!(
                    (kind, arg.kind()),
                    (ty::BoundVariableKind::Region(_), ty::GenericArgKind::Lifetime(_))
                        | (ty::BoundVariableKind::Ty(_), ty::GenericArgKind::Type(_))
                        | (ty::BoundVariableKind::Const(_), ty::GenericArgKind::Const(_))
                )
            })
        {
            return false;
        }
        object_trait
            .instantiate_with_args_and_evidence_and_telescope_clauses(tcx, args, ty::List::empty())
            .0
            .trait_ref
    } else {
        let value = object_trait.skip_binder();
        if value.trait_ref.has_escaping_bound_vars() {
            return false;
        }
        value.trait_ref
    };
    if object_trait_ref != trait_ref {
        return false;
    }
    let Some(projection) = operation.projection_bound.as_projection_clause() else {
        return false;
    };
    let ordinary: Vec<_> = projection.ordinary_bound_vars().collect();
    if ordinary.len() != operation.ordinary_args.len()
        || projection.evidence_bound_vars().count() != 1
        || ordinary.iter().zip(operation.ordinary_args.iter()).any(|(kind, arg)| {
            !matches!(
                (kind, arg.kind()),
                (ty::BoundVariableKind::Region(_), ty::GenericArgKind::Lifetime(_))
                    | (ty::BoundVariableKind::Ty(_), ty::GenericArgKind::Type(_))
                    | (ty::BoundVariableKind::Const(_), ty::GenericArgKind::Const(_))
            )
        })
    {
        return false;
    }
    let ty::Dynamic(bounds, _) = trait_ref.self_ty().kind() else {
        return false;
    };
    let projection_value = projection.skip_binder();
    if projection_value.has_evidence_projections() {
        return false;
    }
    let operation_self = projection_value.projection_term.trait_ref(tcx).self_ty();
    let expected_self = ty::shift_vars(tcx, trait_ref.self_ty(), 1);
    if operation_self != expected_self {
        return false;
    }
    let existential = ty::ExistentialProjection::erase_self_ty(tcx, projection_value);
    let ordinary_vars = tcx.mk_bound_variable_kinds_from_iter(projection.ordinary_bound_vars());
    let existential = ty::Binder::bind_with_vars(existential, ordinary_vars);
    if !bounds.projection_bounds().any(|bound| bound == existential) {
        return false;
    }
    let declarations =
        projection.instantiate_telescope_clauses_with_args(tcx, operation.ordinary_args);
    let declaration = declarations
        .first()
        .and_then(|declaration| remove_empty_clause_binder(tcx, declaration.clause));
    if declarations.len() != 1
        || !matches!(declaration, Some(ty::ClauseKind::Trait(clause))
            if clause.polarity == ty::ClausePolarity::Positive
                && clause.trait_ref == evidence.trait_ref)
    {
        return false;
    }
    let evidence_args = tcx.mk_trait_evidences(&[evidence]);
    let (opened, telescope) = projection.instantiate_with_args_and_evidence_and_telescope_clauses(
        tcx,
        operation.ordinary_args,
        evidence_args,
    );
    let opened_trait_ref = opened.projection_term.trait_ref(tcx);
    let telescope_clause = telescope
        .first()
        .and_then(|declaration| remove_empty_clause_binder(tcx, declaration.clause));
    let matches = opened_trait_ref == trait_ref
        && telescope.len() == 1
        && matches!(telescope_clause, Some(ty::ClauseKind::Trait(clause))
            if clause.polarity == ty::ClausePolarity::Positive && clause.trait_ref == trait_ref);
    matches
}

/// Telescope instantiation wraps each clause in an empty binder. Removing it
/// must shift references to an enclosing carrier binder back to that scope.
fn remove_empty_clause_binder<'tcx>(
    tcx: TyCtxt<'tcx>,
    clause: ty::Clause<'tcx>,
) -> Option<ty::ClauseKind<'tcx>> {
    let clause = clause.kind();
    if !clause.bound_vars().is_empty() {
        return None;
    }
    let (value, telescope) = clause.instantiate_with_args_and_evidence_and_telescope_clauses(
        tcx,
        ty::List::empty(),
        ty::List::empty(),
    );
    telescope.is_empty().then_some(value)
}

struct EvidenceValidator<'tcx> {
    boundary: EvidenceValidationBoundary,
    checking_payload: bool,
    allow_unknown_outer_scope: bool,
    allow_unknown_evidence_scope: bool,
    outer_binder: ty::DebruijnIndex,
    canonical_bound_count: Option<usize>,
    canonical_evidence_var_kinds: Option<CanonicalEvidenceVarKinds<'tcx>>,
    active_evidence: FxHashSet<TraitEvidence<'tcx>>,
    validated_evidence: FxHashSet<(TraitEvidence<'tcx>, u32, Option<usize>, Option<usize>, bool)>,
}

impl<'tcx> EvidenceValidator<'tcx> {
    fn new(
        boundary: EvidenceValidationBoundary,
        allow_unknown_outer_scope: bool,
        allow_unknown_evidence_scope: bool,
    ) -> EvidenceValidator<'tcx> {
        EvidenceValidator {
            boundary,
            checking_payload: false,
            allow_unknown_outer_scope,
            allow_unknown_evidence_scope,
            outer_binder: ty::INNERMOST,
            canonical_bound_count: None,
            canonical_evidence_var_kinds: None,
            active_evidence: FxHashSet::default(),
            validated_evidence: FxHashSet::default(),
        }
    }

    fn validate_payload<T>(&mut self, value: &T) -> Result<(), EvidenceValidationError>
    where
        T: TypeVisitable<TyCtxt<'tcx>> + ?Sized,
    {
        let previous = self.checking_payload;
        self.checking_payload = true;
        let result = match value.visit_with(self) {
            ControlFlow::Continue(()) => Ok(()),
            ControlFlow::Break(error) => Err(error),
        };
        self.checking_payload = previous;
        result
    }

    fn validate_evidence(
        &mut self,
        evidence: TraitEvidence<'tcx>,
        allow_recursive_leaf: bool,
    ) -> Result<(), EvidenceValidationError> {
        let cache_key = (
            evidence,
            self.outer_binder.as_u32(),
            self.canonical_bound_count,
            self.canonical_evidence_var_kinds.map(|kinds| kinds.as_ptr() as usize),
            allow_recursive_leaf,
        );
        if self.validated_evidence.contains(&cache_key) {
            return Ok(());
        }
        if !self.active_evidence.insert(evidence) {
            return Err(EvidenceValidationError::CyclicEvidenceHandles);
        }

        let result = match &evidence.kind {
            TraitEvidenceKind::Selected(recipe) => {
                self.validate_recipe(evidence, recipe, allow_recursive_leaf)
            }
            TraitEvidenceKind::Infer(_) => Err(EvidenceValidationError::EvidenceInferenceVariable),
            TraitEvidenceKind::Bound(index, bound) => match index {
                ty::BoundVarIndexKind::Canonical => {
                    if let Some(kinds) = self.canonical_evidence_var_kinds {
                        if let Some(kind) = kinds.get(bound.var().as_usize()) {
                            if evidence.trait_ref != kind.trait_ref() {
                                Err(EvidenceValidationError::CanonicalEvidencePredicateMismatch)
                            } else {
                                self.validate_payload(&evidence.trait_ref)
                            }
                        } else {
                            Err(EvidenceValidationError::CanonicalVariableOutOfRange)
                        }
                    } else if self.allow_unknown_evidence_scope {
                        self.validate_payload(&evidence.trait_ref)
                    } else {
                        Err(EvidenceValidationError::EvidenceBoundVariable)
                    }
                }
                ty::BoundVarIndexKind::Bound(_) if self.allow_unknown_evidence_scope => {
                    self.validate_payload(&evidence.trait_ref)
                }
                ty::BoundVarIndexKind::Bound(_) => {
                    Err(EvidenceValidationError::EvidenceBoundVariable)
                }
            },
            TraitEvidenceKind::Placeholder(_) => Err(EvidenceValidationError::EvidencePlaceholder),
            TraitEvidenceKind::Error(_) => Err(EvidenceValidationError::EvidenceError),
        };

        self.active_evidence.remove(&evidence);
        if result.is_ok() {
            self.validated_evidence.insert(cache_key);
        }
        result
    }

    fn validate_recipe(
        &mut self,
        evidence: TraitEvidence<'tcx>,
        recipe: &CandidateEvidence<TyCtxt<'tcx>>,
        allow_recursive_leaf: bool,
    ) -> Result<(), EvidenceValidationError> {
        if recipe.nodes.is_empty() {
            return Err(EvidenceValidationError::EmptyRecipe);
        }
        let root = usize::try_from(recipe.root)
            .map_err(|_| EvidenceValidationError::RecipeRootOutOfBounds)?;
        if root >= recipe.nodes.len() {
            return Err(EvidenceValidationError::RecipeRootOutOfBounds);
        }
        if evidence.trait_ref != recipe.nodes[root].trait_ref {
            return Err(EvidenceValidationError::InvalidUniqueRecipe);
        }

        let mut used_cycle_keys = vec![false; recipe.cycle_keys.len()];
        for (index, node) in recipe.nodes.iter().enumerate() {
            self.validate_payload(&node.trait_ref)?;
            self.validate_payload(&node.source)?;

            for &nested in &node.nested {
                let nested = usize::try_from(nested)
                    .map_err(|_| EvidenceValidationError::RecipeEdgeOutOfBounds)?;
                if nested >= recipe.nodes.len() {
                    return Err(EvidenceValidationError::RecipeEdgeOutOfBounds);
                }
            }

            match node.source {
                CandidateEvidenceSource::Unique(key) => {
                    if node.trait_ref != key.trait_ref
                        || node.nested.len() != 1
                        || !node.nested_evidence.is_empty()
                    {
                        return Err(EvidenceValidationError::InvalidUniqueRecipe);
                    }
                    let selected = usize::try_from(node.nested[0])
                        .map_err(|_| EvidenceValidationError::RecipeEdgeOutOfBounds)?;
                    if selected >= recipe.nodes.len()
                        || node.trait_ref != recipe.nodes[selected].trait_ref
                    {
                        return Err(EvidenceValidationError::InvalidUniqueRecipe);
                    }
                }
                CandidateEvidenceSource::Recursive { cycle } => {
                    let exact_leaf = recipe.nodes.len() == 1
                        && root == index
                        && node.nested.is_empty()
                        && node.nested_evidence.is_empty();
                    if !allow_recursive_leaf || !exact_leaf {
                        return Err(EvidenceValidationError::UnguardedRecursiveRecipe);
                    }
                    let cycle = usize::try_from(cycle)
                        .map_err(|_| EvidenceValidationError::RecipeCycleIndexOutOfBounds)?;
                    let Some(cycle_key) = recipe.cycle_keys.get(cycle) else {
                        return Err(EvidenceValidationError::RecipeCycleIndexOutOfBounds);
                    };
                    if std::mem::replace(&mut used_cycle_keys[cycle], true) {
                        return Err(EvidenceValidationError::DuplicateRecipeCycleKey);
                    }
                    self.validate_proof_cycle(cycle_key, node.trait_ref)?;
                }
                CandidateEvidenceSource::Error => {
                    return Err(EvidenceValidationError::ErrorCandidate);
                }
                CandidateEvidenceSource::CoherenceUnknowable => {
                    return Err(EvidenceValidationError::NonProofCandidate);
                }
                CandidateEvidenceSource::Dyn { operation: Some(_), .. } => {
                    if node.trait_ref != evidence.trait_ref
                        || !ty::tls::with(|tcx| {
                            let evidence = tcx.lift(evidence);
                            let TraitEvidenceKind::Selected(recipe) = &evidence.kind else {
                                return false;
                            };
                            let node = &recipe.nodes[index];
                            let CandidateEvidenceSource::Dyn {
                                object_bound,
                                instantiation,
                                operation: Some(operation),
                                ..
                            } = node.source
                            else {
                                return false;
                            };
                            dyn_projection_operation_matches(
                                tcx,
                                node.trait_ref,
                                object_bound,
                                instantiation,
                                operation,
                                evidence,
                            )
                        })
                    {
                        return Err(EvidenceValidationError::InvalidDynProjectionOperation);
                    }
                }
                CandidateEvidenceSource::Impl { .. }
                | CandidateEvidenceSource::Builtin { .. }
                | CandidateEvidenceSource::Dyn { operation: None, .. }
                | CandidateEvidenceSource::ParamEnv { .. }
                | CandidateEvidenceSource::AliasBound(_) => {}
            }

            for nested in &node.nested_evidence {
                match nested {
                    CandidateEvidenceUse::Instantiated(evidence) => {
                        self.validate_evidence(*evidence, true)?;
                    }
                    CandidateEvidenceUse::Canonical {
                        original_values,
                        original_evidence_values,
                        response,
                    } => {
                        if original_values.len() != response.value.var_values.var_values.len() {
                            return Err(EvidenceValidationError::NestedCanonicalArityMismatch);
                        }
                        if original_evidence_values.len()
                            != response.value.evidence_var_values.var_values.len()
                        {
                            return Err(EvidenceValidationError::NestedCanonicalArityMismatch);
                        }
                        self.validate_payload(original_values)?;
                        for evidence in original_evidence_values.iter() {
                            self.validate_evidence(evidence, false)?;
                        }

                        let var_count = response.var_kinds.len();
                        for kind in response.var_kinds.iter() {
                            if kind.universe() > response.max_universe {
                                return Err(EvidenceValidationError::InvalidCanonicalUniverse);
                            }
                            if let ty::CanonicalVarKind::Ty { sub_root, .. } = kind
                                && sub_root.as_usize() >= var_count
                            {
                                return Err(EvidenceValidationError::CanonicalVariableOutOfRange);
                            }
                        }
                        for kind in response.evidence_var_kinds.iter() {
                            if kind.universe() > response.max_universe {
                                return Err(EvidenceValidationError::InvalidCanonicalUniverse);
                            }
                        }

                        let previous_binder = self.outer_binder;
                        let previous_canonical = self.canonical_bound_count;
                        let previous_evidence_canonical = self.canonical_evidence_var_kinds;
                        let previous_unknown_scope = self.allow_unknown_outer_scope;
                        self.outer_binder = ty::INNERMOST;
                        self.canonical_bound_count = Some(var_count);
                        self.canonical_evidence_var_kinds = Some(response.evidence_var_kinds);
                        self.allow_unknown_outer_scope = false;
                        let result = (|| {
                            self.validate_payload(&response.value.var_values)?;
                            for kind in response.evidence_var_kinds.iter() {
                                self.validate_payload(&kind.trait_ref())?;
                            }
                            for evidence in response.value.evidence_var_values.var_values.iter() {
                                self.validate_evidence(evidence, false)?;
                            }
                            self.validate_evidence(response.value.evidence, true)
                        })();
                        self.outer_binder = previous_binder;
                        self.canonical_bound_count = previous_canonical;
                        self.canonical_evidence_var_kinds = previous_evidence_canonical;
                        self.allow_unknown_outer_scope = previous_unknown_scope;
                        result?;
                    }
                }
            }
        }

        if used_cycle_keys.into_iter().any(|used| !used) {
            return Err(EvidenceValidationError::UnreferencedRecipeCycleKey);
        }

        fn visit_node<'tcx>(
            nodes: &[CandidateEvidenceNode<TyCtxt<'tcx>>],
            index: usize,
            state: &mut [u8],
        ) -> Result<(), EvidenceValidationError> {
            match state[index] {
                2 => return Ok(()),
                1 => return Err(EvidenceValidationError::RecipeCycle),
                0 => {}
                _ => unreachable!(),
            }
            state[index] = 1;
            for &nested in &nodes[index].nested {
                let nested = usize::try_from(nested)
                    .map_err(|_| EvidenceValidationError::RecipeEdgeOutOfBounds)?;
                if nested >= nodes.len() {
                    return Err(EvidenceValidationError::RecipeEdgeOutOfBounds);
                }
                visit_node(nodes, nested, state)?;
            }
            state[index] = 2;
            Ok(())
        }

        let mut state = vec![0; recipe.nodes.len()];
        visit_node(&recipe.nodes, root, &mut state)?;
        if state.into_iter().any(|state| state != 2) {
            return Err(EvidenceValidationError::UnreachableRecipeNode);
        }
        Ok(())
    }

    fn validate_proof_cycle(
        &mut self,
        cycle: &ProofCycleKey<TyCtxt<'tcx>>,
        trait_ref: ty::TraitRef<'tcx>,
    ) -> Result<(), EvidenceValidationError> {
        if cycle.participants.is_empty() || cycle.participants.len() != cycle.edge_kinds.len() {
            return Err(EvidenceValidationError::MalformedProofCycle);
        }
        if cycle.path_kind() != ir::search_graph::PathKind::Coinductive {
            return Err(EvidenceValidationError::NonProductiveProofCycle);
        }
        if cycle.head_trait_ref != trait_ref {
            return Err(EvidenceValidationError::ProofCycleRootMismatch);
        }
        self.validate_payload(&cycle.head_trait_ref)?;

        for (index, participant) in cycle.participants.iter().enumerate() {
            if cycle.participants[..index].contains(participant) {
                return Err(EvidenceValidationError::RepeatedProofCycleParticipant);
            }

            let canonical = &participant.canonical;
            let var_count = canonical.var_kinds.len();
            for kind in canonical.var_kinds.iter() {
                if kind.universe() > canonical.max_universe {
                    return Err(EvidenceValidationError::InvalidCanonicalUniverse);
                }
                if let ty::CanonicalVarKind::Ty { sub_root, .. } = kind
                    && sub_root.as_usize() >= var_count
                {
                    return Err(EvidenceValidationError::CanonicalVariableOutOfRange);
                }
            }
            for kind in canonical.evidence_var_kinds.iter() {
                if kind.universe() > canonical.max_universe {
                    return Err(EvidenceValidationError::InvalidCanonicalUniverse);
                }
            }

            let previous_binder = self.outer_binder;
            let previous_canonical = self.canonical_bound_count;
            let previous_evidence_canonical = self.canonical_evidence_var_kinds;
            let previous_unknown_scope = self.allow_unknown_outer_scope;
            self.outer_binder = ty::INNERMOST;
            self.canonical_bound_count = Some(var_count);
            self.canonical_evidence_var_kinds = Some(canonical.evidence_var_kinds);
            self.allow_unknown_outer_scope = false;
            let result = (|| {
                for kind in canonical.evidence_var_kinds.iter() {
                    self.validate_payload(&kind.trait_ref())?;
                }
                self.validate_payload(&canonical.value)
            })();
            self.outer_binder = previous_binder;
            self.canonical_bound_count = previous_canonical;
            self.canonical_evidence_var_kinds = previous_evidence_canonical;
            self.allow_unknown_outer_scope = previous_unknown_scope;
            result?;
        }
        Ok(())
    }

    fn validate_bound_index(
        &self,
        index: ty::BoundVarIndexKind,
        var: ty::BoundVar,
    ) -> Result<(), EvidenceValidationError> {
        match index {
            ty::BoundVarIndexKind::Bound(debruijn) => {
                if debruijn < self.outer_binder || self.allow_unknown_outer_scope {
                    Ok(())
                } else {
                    Err(EvidenceValidationError::EscapingBoundVariable)
                }
            }
            ty::BoundVarIndexKind::Canonical => match self.canonical_bound_count {
                Some(count) if var.as_usize() < count => Ok(()),
                Some(_) => Err(EvidenceValidationError::CanonicalVariableOutOfRange),
                None if self.allow_unknown_outer_scope => Ok(()),
                None => Err(EvidenceValidationError::EscapingCanonicalVariable),
            },
        }
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for EvidenceValidator<'tcx> {
    type Result = ControlFlow<EvidenceValidationError>;

    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        binder: &ty::Binder<'tcx, T>,
    ) -> Self::Result {
        self.outer_binder.shift_in(1);
        let result = binder.super_visit_with(self);
        self.outer_binder.shift_out(1);
        result
    }

    fn visit_ty(&mut self, value: Ty<'tcx>) -> Self::Result {
        if self.checking_payload {
            match *value.kind() {
                ty::Infer(_) => {
                    return ControlFlow::Break(EvidenceValidationError::TypeInferenceVariable);
                }
                ty::Placeholder(_) => {
                    return ControlFlow::Break(EvidenceValidationError::TypePlaceholder);
                }
                ty::Bound(index, bound) => {
                    if let Err(error) = self.validate_bound_index(index, bound.var()) {
                        return ControlFlow::Break(error);
                    }
                }
                ty::Error(_) => return ControlFlow::Break(EvidenceValidationError::TypeError),
                _ => {}
            }
        }
        value.super_visit_with(self)
    }

    fn visit_region(&mut self, value: ty::Region<'tcx>) -> Self::Result {
        if !self.checking_payload {
            return ControlFlow::Continue(());
        }
        match value.kind() {
            ty::ReVar(_) => ControlFlow::Break(EvidenceValidationError::TypeInferenceVariable),
            ty::RePlaceholder(_) => ControlFlow::Break(EvidenceValidationError::TypePlaceholder),
            ty::ReBound(index, bound) => match self.validate_bound_index(index, bound.var()) {
                Ok(()) => ControlFlow::Continue(()),
                Err(error) => ControlFlow::Break(error),
            },
            ty::ReError(_) => ControlFlow::Break(EvidenceValidationError::TypeError),
            ty::ReEarlyParam(_) | ty::ReLateParam(_) | ty::ReStatic | ty::ReErased => {
                ControlFlow::Continue(())
            }
        }
    }

    fn visit_const(&mut self, value: ty::Const<'tcx>) -> Self::Result {
        if self.checking_payload {
            match value.kind() {
                ty::ConstKind::Infer(_) => {
                    return ControlFlow::Break(EvidenceValidationError::TypeInferenceVariable);
                }
                ty::ConstKind::Placeholder(_) => {
                    return ControlFlow::Break(EvidenceValidationError::TypePlaceholder);
                }
                ty::ConstKind::Bound(index, bound) => {
                    if let Err(error) = self.validate_bound_index(index, bound.var()) {
                        return ControlFlow::Break(error);
                    }
                }
                ty::ConstKind::Error(_) => {
                    return ControlFlow::Break(EvidenceValidationError::TypeError);
                }
                ty::ConstKind::Param(_)
                | ty::ConstKind::Alias(..)
                | ty::ConstKind::Value(_)
                | ty::ConstKind::Expr(_) => {}
            }
        }
        value.super_visit_with(self)
    }

    fn visit_trait_evidence(&mut self, evidence: TraitEvidence<'tcx>) -> Self::Result {
        match self.validate_evidence(evidence, false) {
            Ok(()) => ControlFlow::Continue(()),
            Err(error) => ControlFlow::Break(error),
        }
    }

    fn visit_evidence_projection(&mut self, projection: EvidenceProjection<'tcx>) -> Self::Result {
        if self.boundary == EvidenceValidationBoundary::Codegen {
            return ControlFlow::Break(EvidenceValidationError::UnnormalizedProjection);
        }
        match self.validate_evidence(projection.evidence, false) {
            Ok(()) => ControlFlow::Continue(()),
            Err(error) => ControlFlow::Break(error),
        }
    }

    fn visit_error(&mut self, _guar: ErrorGuaranteed) -> Self::Result {
        if self.checking_payload {
            ControlFlow::Break(EvidenceValidationError::TypeError)
        } else {
            ControlFlow::Continue(())
        }
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

impl<'tcx, E: ty::codec::TyEncoder<'tcx>> rustc_serialize::Encodable<E> for CanonicalInput<'tcx> {
    fn encode(&self, encoder: &mut E) {
        rustc_serialize::Encodable::encode(&**self, encoder);
    }
}

impl<'tcx, D: ty::codec::TyDecoder<'tcx>> rustc_serialize::Decodable<D> for CanonicalInput<'tcx> {
    fn decode(decoder: &mut D) -> Self {
        let data = rustc_serialize::Decodable::decode(decoder);
        decoder.interner().intern_canonical_input(data)
    }
}

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
        if self.evidence.is_none() && self.is_empty() {
            return Ok(self);
        }

        Ok(FallibleTypeFolder::cx(folder).mk_external_constraints(ExternalConstraintsData {
            // Proof recipes own an independent canonical variable space and
            // must not be folded as part of the ordinary query response.
            evidence: self.evidence,
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
        if self.evidence.is_none() && self.is_empty() {
            return self;
        }

        TypeFolder::cx(folder).mk_external_constraints(ExternalConstraintsData {
            // See `try_fold_with`: this is an opaque canonical proof channel.
            evidence: self.evidence,
            region_constraints: self.region_constraints.clone().fold_with(folder),
            opaque_types: self.opaque_types.iter().map(|opaque| opaque.fold_with(folder)).collect(),
            normalization_nested_goals: self.normalization_nested_goals.clone().fold_with(folder),
        })
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        let ExternalConstraintsData {
            evidence,
            region_constraints,
            opaque_types,
            normalization_nested_goals,
        } = &**self;

        // The proof channel is independently canonicalized. Visiting it here
        // would make its bound variables look like part of the enclosing
        // response and would contaminate response flags/universe accounting.
        let _ = evidence;
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
