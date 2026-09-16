use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{ControlFlow, Deref};

use derive_where::derive_where;
use rustc_ast_ir::visit::VisitorResult;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash, StableHash_NoContext};
use rustc_type_ir_macros::{
    GenericTypeVisitable, Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic,
};
use tracing::instrument;

use crate::data_structures::SsoHashSet;
use crate::fold::{FallibleTypeFolder, TypeFoldable, TypeFolder, TypeSuperFoldable};
use crate::inherent::*;
use crate::visit::{Flags, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
use crate::{
    self as ty, DebruijnIndex, Interner, PredicateProxy, Region, UniverseIndex, Unnormalized,
    Upcast, try_visit,
};

/// `Binder` is a binder for higher-ranked lifetimes or types. It is part of the
/// compiler's representation for things like `for<'a> Fn(&'a isize)`
/// (which would be represented by the type `PolyTraitRef == Binder<I, TraitRef>`).
///
/// See <https://rustc-dev-guide.rust-lang.org/ty_module/instantiating_binders.html>
/// for more details.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner, T)]
#[derive(GenericTypeVisitable, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct Binder<I: Interner, T> {
    value: T,
    bound_vars: I::BoundVarKinds,
}

/// A source contract rebased into the scope of one owning telescope.
///
/// Every member in `clauses` retains its own `Clause` binder. References to
/// the owning telescope therefore occur one de Bruijn level outside that
/// member binder. This is intentionally distinct from
/// [`crate::solve::RequiredContract`]: removing the owning telescope
/// instantiates those outer references.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable, TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct BoundRequiredContractData<I: Interner> {
    pub identity: ty::solve::InstantiatedItemContract<I>,
    pub clauses: I::Clauses,
    /// Index of the trait clause whose evidence owns this atomic bundle.
    pub principal_index: u32,
    /// Opening substitution, if this bound contract has already crossed an
    /// enclosing telescope. Freshly rebased callable contracts use `None`;
    /// removing their owning binder records one shared substitution here.
    pub ordinary_args: Option<I::GenericArgs>,
}

impl<I: Interner> Eq for BoundRequiredContractData<I> {}

impl<I: Interner> BoundRequiredContractData<I> {
    pub fn assert_well_formed(&self) {
        let principal = self
            .clauses
            .get(self.principal_index as usize)
            .and_then(|clause| clause.as_trait_clause())
            .expect("required-contract principal must be a trait clause");
        assert_eq!(
            principal.skip_binder().polarity,
            ty::ClausePolarity::Positive,
            "required-contract principal must be positive"
        );
    }

    fn instantiate(self, cx: I, ordinary_args: I::GenericArgs) -> ty::solve::RequiredContract<I> {
        assert!(self.ordinary_args.is_none(), "contract has already been instantiated");
        ty::solve::RequiredContract::new(
            cx,
            self.identity,
            self.clauses,
            self.principal_index,
            Some(ordinary_args),
        )
    }
}

/// One compiler-internal proof declaration in a dependent binder telescope.
///
/// `clause` is the principal predicate proved by the evidence slot. When the
/// slot originates from a source trait bound, `required_contract` carries the
/// complete bundle belonging to that exact source bound. Keeping the bundle
/// on the declaration, rather than looking it up from the principal trait ref,
/// lets two slots with identical principals retain different associated-item
/// equalities.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable, TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct EvidenceVariable<I: Interner> {
    pub clause: ty::ClauseKind<I>,
    pub required_contract: Option<I::BoundRequiredContract>,
}

impl<I: Interner> Eq for EvidenceVariable<I> {}

impl<I: Interner> EvidenceVariable<I> {
    pub fn principal(clause: ty::ClauseKind<I>) -> Self {
        EvidenceVariable { clause, required_contract: None }
    }
}

/// A clause introduced by one entry of a dependent binder telescope.
///
/// `clause` is instantiated for the current inference context. `identity`
/// keeps the original binder and clause together, so a universally-instantiated
/// clause can retain its stable telescope identity after being appended to a
/// parameter environment.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable, TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct InstantiatedTelescopeClause<I: Interner> {
    pub index: u32,
    pub clause: I::Clause,
    pub identity: I::Clause,
    /// The ordinary lifetime/type/const substitution used to instantiate the
    /// owning binder. Evidence entries form a telescope suffix and therefore
    /// never consume an argument in this list.
    pub instantiation: I::GenericArgs,
    /// Complete source contract instantiated by the same ordinary/evidence
    /// substitution as `clause`. Typed-const declarations and synthetic
    /// evidence declarations which have no source bundle use `None`.
    pub required_contract: Option<ty::solve::RequiredContract<I>>,
}

impl<I: Interner> Eq for InstantiatedTelescopeClause<I> {}

impl<I: Interner, T: Eq> Eq for Binder<I, T> {}

impl<I: Interner, T> Binder<I, T>
where
    T: TypeVisitable<I>,
{
    /// Wraps `value` in a binder, asserting that `value` does not
    /// contain any bound vars that would be bound by the
    /// binder. This is commonly used to 'inject' a value T into a
    /// different binding level.
    #[track_caller]
    pub fn dummy(value: T) -> Binder<I, T> {
        assert!(
            !value.has_escaping_bound_vars(),
            "`{value:?}` has escaping bound vars, so it cannot be wrapped in a dummy binder."
        );
        Binder { value, bound_vars: Default::default() }
    }

    pub fn bind_with_vars(value: T, bound_vars: I::BoundVarKinds) -> Binder<I, T> {
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            validator.validate_telescope();
            let _ = value.visit_with(&mut validator);
        }
        Binder { value, bound_vars }
    }
}

fn evidence_clause_trait_ref<I: Interner>(clause: ty::ClauseKind<I>) -> ty::TraitRef<I> {
    match clause {
        ty::ClauseKind::Trait(ty::TraitClause {
            trait_ref,
            polarity: ty::ClausePolarity::Positive,
        }) => trait_ref,
        _ => panic!("binder evidence entry must be a positive trait clause, found {clause:?}"),
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Binder<I, T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_binder(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_binder(self)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Binder<I, T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_binder(self)
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeSuperFoldable<I> for Binder<I, T> {
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let Binder { value, bound_vars } = self;
        let bound_vars = if bound_vars.iter().any(|entry| {
            matches!(entry, BoundVariableKind::Const(Some(_)) | BoundVariableKind::Evidence(_))
        }) {
            let entries = bound_vars
                .iter()
                .map(|entry| entry.try_fold_with(folder))
                .collect::<Result<Vec<_>, F::Error>>()?;
            I::BoundVarKinds::from_vars(folder.cx(), entries)
        } else {
            bound_vars
        };
        let value = value.try_fold_with(folder)?;
        Ok(Binder::bind_with_vars(value, bound_vars))
    }

    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        let Binder { value, bound_vars } = self;
        let bound_vars = if bound_vars.iter().any(|entry| {
            matches!(entry, BoundVariableKind::Const(Some(_)) | BoundVariableKind::Evidence(_))
        }) {
            I::BoundVarKinds::from_vars(
                folder.cx(),
                bound_vars.iter().map(|entry| entry.fold_with(folder)),
            )
        } else {
            bound_vars
        };
        Binder::bind_with_vars(value.fold_with(folder), bound_vars)
    }
}

impl<I: Interner, T: TypeFoldable<I>> Binder<I, T> {
    /// Instantiates this binder with an explicit substitution for its ordinary
    /// lifetime/type/const variables.
    ///
    /// Binder-owned typed-const and evidence clauses must not be silently
    /// discarded. This operation is only valid when the binder has no telescope
    /// clauses. Use [`Binder::instantiate_with_args_and_telescope_clauses`] for
    /// ordinary dependent declarations, or the evidence-aware counterpart when
    /// a value or declaration refers to a bound evidence entry.
    pub fn instantiate_with_args(self, cx: I, args: I::GenericArgs) -> T {
        assert!(
            !self.has_telescope_clauses(),
            "instantiating a dependent binder without its telescope clauses: {self:?}"
        );
        self.validate_instantiation(cx, args);
        self.skip_binder().fold_with(&mut LateBoundArgFolder::new(cx, args))
    }

    /// Instantiates this binder with separate substitutions for its ordinary
    /// variables and its evidence suffix.
    ///
    /// `evidence_args` is dense and ordered like [`Binder::evidence_bound_vars`].
    /// A [`BoundEvidence`] nevertheless stores its index in the complete
    /// telescope, so lookup subtracts the ordinary prefix length. Every proof
    /// is checked against the fully-instantiated predicate of its declaration
    /// before the bound value is folded.
    ///
    /// Supplying evidence discharges principal trait clauses. Typed const
    /// declarations and complete contracts require the counterpart returning
    /// telescope clauses and are rejected here.
    pub fn instantiate_with_args_and_evidence(
        self,
        cx: I,
        args: I::GenericArgs,
        evidence_args: I::TraitEvidences,
    ) -> T {
        assert!(
            !self.bound_vars.iter().any(|entry| {
                matches!(entry, BoundVariableKind::Const(Some(_)))
                    || matches!(entry, BoundVariableKind::Evidence(evidence)
                        if evidence.required_contract.is_some())
            }),
            "instantiating a telescope without its declaration obligations: {self:?}"
        );
        self.validate_evidence_instantiation(cx, args, evidence_args);

        let ordinary_count = self.ordinary_bound_var_count();
        self.skip_binder().fold_with(&mut LateBoundArgFolder::with_evidence(
            cx,
            args,
            ordinary_count,
            evidence_args,
        ))
    }

    /// Instantiates the value and every binder-owned declaration using one
    /// ordinary substitution and one proof substitution.
    ///
    /// Unlike [`Binder::instantiate_with_args_and_evidence`], this operation
    /// is safe for a telescope containing typed const declarations: those
    /// declarations are returned as [`InstantiatedTelescopeClause`]s instead
    /// of being silently discharged. Evidence declarations are returned too,
    /// preserving the exact binder identity, telescope index, and ordinary
    /// substitution which produced each proof argument.
    pub fn instantiate_with_args_and_evidence_and_telescope_clauses(
        self,
        cx: I,
        args: I::GenericArgs,
        evidence_args: I::TraitEvidences,
    ) -> (T, Vec<InstantiatedTelescopeClause<I>>) {
        self.validate_evidence_instantiation(cx, args, evidence_args);

        let ordinary_count = self.ordinary_bound_var_count();
        let telescope_clauses =
            self.instantiate_telescope_clauses(cx, args, Some((ordinary_count, evidence_args)));
        let value = self.value.fold_with(&mut LateBoundArgFolder::with_evidence(
            cx,
            args,
            ordinary_count,
            evidence_args,
        ));
        (value, telescope_clauses)
    }

    /// Universally instantiates this telescope and represents each evidence
    /// declaration by the exact parameter-environment origin its clause will
    /// receive. Source-contract slots use their instantiated item-contract
    /// identity; synthetic slots retain their binder-owned identity.
    ///
    /// Evidence declarations form a dependent suffix. Recipes are therefore
    /// built in telescope order: the predicate for one slot may use proof
    /// values from earlier slots, but can never observe itself or a later
    /// slot. The returned clauses must still be installed in the universal
    /// parameter environment; carrying the same origin in the selected proof
    /// prevents normalization from drifting to a different assumption.
    pub fn instantiate_with_args_and_binder_assumption_evidence(
        self,
        cx: I,
        args: I::GenericArgs,
    ) -> (T, Vec<InstantiatedTelescopeClause<I>>) {
        self.validate_instantiation(cx, args);

        let ordinary_count = self.ordinary_bound_var_count();
        let mut evidence = Vec::with_capacity(self.evidence_bound_vars().count());

        for (index, entry) in self.bound_vars.iter().enumerate() {
            let BoundVariableKind::Evidence(evidence_variable) = entry else { continue };
            let telescope_index = u32::try_from(index).expect("binder telescope index overflow");
            let clause = evidence_variable.clause;
            let evidence_prefix = cx.mk_trait_evidences(&evidence);
            let mut folder =
                LateBoundArgFolder::with_evidence(cx, args, ordinary_count, evidence_prefix);
            let trait_ref = evidence_clause_trait_ref(clause).fold_with(&mut folder);
            let required_contract = evidence_variable
                .required_contract
                .map(|contract| (*contract).clone().fold_with(&mut folder).instantiate(cx, args));
            let identity = Binder::bind_with_vars(clause, self.bound_vars).upcast(cx);
            let source = ty::solve::CandidateEvidenceSource::ParamEnv {
                source: ty::solve::ParamEnvSource::NonGlobal,
                origin: required_contract.map_or_else(
                    || ty::solve::ParamEnvAssumption::Binder {
                        telescope_index,
                        identity,
                        instantiation: args,
                    },
                    |required_contract| ty::solve::ParamEnvAssumption::ItemContract {
                        contract: required_contract.identity,
                    },
                ),
            };
            evidence.push(cx.mk_trait_evidence(ty::solve::CandidateEvidence::new(
                trait_ref,
                source,
                [],
            )));
        }

        let evidence = cx.mk_trait_evidences(&evidence);
        self.instantiate_with_args_and_evidence_and_telescope_clauses(cx, args, evidence)
    }

    /// Instantiates the trait predicate of the next evidence declaration using
    /// the ordinary substitution and the already-created proof prefix.
    ///
    /// Existential binder consumers use this to allocate one evidence variable
    /// at a time. The telescope index must name the entry immediately following
    /// `evidence_prefix`; this prevents a caller from observing a declaration
    /// before all proofs on which it depends have been created. The prefix is
    /// checked against the declarations before the next predicate is returned.
    pub fn instantiate_evidence_trait_ref_with_prefix(
        &self,
        cx: I,
        args: I::GenericArgs,
        telescope_index: u32,
        evidence_prefix: I::TraitEvidences,
    ) -> ty::TraitRef<I> {
        self.validate_instantiation(cx, args);

        let ordinary_count = self.ordinary_bound_var_count();
        let evidence_entries = self.evidence_bound_vars().collect::<Vec<_>>();
        assert!(
            evidence_prefix.len() < evidence_entries.len(),
            "requested an evidence declaration past the end of the binder telescope"
        );
        let (current_index, current_clause) = evidence_entries[evidence_prefix.len()];
        assert_eq!(
            current_index, telescope_index,
            "evidence declarations must be instantiated in telescope order"
        );
        assert_eq!(
            usize::try_from(telescope_index).expect("binder telescope index overflow"),
            ordinary_count + evidence_prefix.len(),
            "evidence entries must be a contiguous binder suffix"
        );

        let mut folder =
            LateBoundArgFolder::with_evidence(cx, args, ordinary_count, evidence_prefix);
        for ((_, clause), replacement) in
            evidence_entries.iter().take(evidence_prefix.len()).zip(evidence_prefix.iter())
        {
            let expected = evidence_clause_trait_ref(*clause).fold_with(&mut folder);
            replacement.assert_well_formed();
            assert_eq!(
                replacement.trait_ref, expected,
                "evidence prefix proves the wrong instantiated telescope predicate"
            );
        }

        evidence_clause_trait_ref(current_clause).fold_with(&mut folder)
    }

    fn validate_evidence_instantiation(
        &self,
        cx: I,
        args: I::GenericArgs,
        evidence_args: I::TraitEvidences,
    ) {
        self.validate_instantiation(cx, args);

        let ordinary_count = self.ordinary_bound_var_count();
        let evidence_entries = self.evidence_bound_vars().collect::<Vec<_>>();
        assert_eq!(
            evidence_entries.len(),
            evidence_args.len(),
            "wrong number of evidence arguments for binder instantiation: binder={self:?}, evidence_args={evidence_args:?}"
        );

        let mut folder = LateBoundArgFolder::with_evidence(cx, args, ordinary_count, evidence_args);
        for (suffix_index, ((telescope_index, clause), replacement)) in
            evidence_entries.into_iter().zip(evidence_args.iter()).enumerate()
        {
            assert_eq!(
                usize::try_from(telescope_index).expect("binder telescope index overflow"),
                ordinary_count + suffix_index,
                "evidence entries must be a contiguous binder suffix"
            );
            let expected = evidence_clause_trait_ref(clause).fold_with(&mut folder);
            replacement.assert_well_formed();
            assert_eq!(
                replacement.trait_ref, expected,
                "evidence argument proves the wrong instantiated predicate at telescope index {telescope_index}"
            );
        }
    }

    /// Instantiates this binder's value and every binder-owned clause with one
    /// explicit ordinary substitution.
    ///
    /// The returned clauses retain both their stable binder/telescope identity
    /// and the exact substitution used at this instantiation site. This is
    /// required even when an ordinary parameter is erased from the instantiated
    /// trait ref: proof identity must not be reconstructed from the result.
    /// References to bound evidence require
    /// [`Binder::instantiate_with_args_and_evidence_and_telescope_clauses`].
    pub fn instantiate_with_args_and_telescope_clauses(
        self,
        cx: I,
        args: I::GenericArgs,
    ) -> (T, Vec<InstantiatedTelescopeClause<I>>) {
        let telescope_clauses = self.instantiate_telescope_clauses_with_args(cx, args);
        let value = self.value.fold_with(&mut LateBoundArgFolder::new(cx, args));
        (value, telescope_clauses)
    }

    /// Instantiates only the clauses owned by this binder's telescope.
    ///
    /// This lets an inference context inspect each instantiated telescope
    /// predicate before it allocates corresponding evidence variables and
    /// substitutes those proof values into the binder's main value. The method
    /// only supplies ordinary arguments; use the evidence-aware method when a
    /// value or declaration refers to a bound evidence entry. It is deliberately
    /// separate from `skip_binder`: no bound value is exposed or claimed to have
    /// been instantiated by this operation.
    pub fn instantiate_telescope_clauses_with_args(
        &self,
        cx: I,
        args: I::GenericArgs,
    ) -> Vec<InstantiatedTelescopeClause<I>> {
        self.validate_instantiation(cx, args);
        self.instantiate_telescope_clauses(cx, args, None)
    }

    fn instantiate_telescope_clauses(
        &self,
        cx: I,
        args: I::GenericArgs,
        evidence: Option<(usize, I::TraitEvidences)>,
    ) -> Vec<InstantiatedTelescopeClause<I>> {
        let bound_vars = self.bound_vars;
        let telescope_clauses = bound_vars
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let index = u32::try_from(index).expect("binder telescope index overflow");
                let (clause, required_contract) = match entry {
                    BoundVariableKind::Const(Some(expected_ty)) => {
                        let ct = I::Const::new_bound(
                            cx,
                            ty::INNERMOST,
                            ty::BoundConst::new(ty::BoundVar::from_u32(index)),
                        );
                        (ty::ClauseKind::ConstArgHasType(ct, expected_ty), None)
                    }
                    BoundVariableKind::Evidence(evidence) => {
                        (evidence.clause, evidence.required_contract)
                    }
                    BoundVariableKind::Ty(_)
                    | BoundVariableKind::Region(_)
                    | BoundVariableKind::Const(None) => return None,
                };
                let identity = Binder::bind_with_vars(clause, bound_vars).upcast(cx);
                Some((index, clause, identity, required_contract))
            })
            .collect::<Vec<_>>();

        let mut folder = LateBoundArgFolder::new(cx, args);
        folder.evidence =
            evidence.map(|(ordinary_count, args)| LateBoundEvidenceArgs { ordinary_count, args });
        let telescope_clauses = telescope_clauses
            .into_iter()
            .map(|(index, clause, identity, required_contract)| {
                // `identity` names the declaration in its original binder and
                // must never be instantiated. In particular, folding it would
                // enter that binder and make variables from an enclosing binder
                // look like variables owned by the binder being removed here.
                let clause = clause.fold_with(&mut folder);
                // `I::Clause` is itself binder-shaped. If the instantiated
                // clause still references an enclosing binder, moving it under
                // this empty clause binder must shift that reference in once.
                let clause = ty::shift_vars(cx, clause, 1);
                let clause = Binder::bind_with_vars(clause, Default::default()).upcast(cx);
                let required_contract = required_contract.map(|contract| {
                    (*contract).clone().fold_with(&mut folder).instantiate(cx, args)
                });
                InstantiatedTelescopeClause {
                    index,
                    clause,
                    identity,
                    instantiation: args,
                    required_contract,
                }
            })
            .collect();
        telescope_clauses
    }

    fn validate_instantiation(&self, cx: I, args: I::GenericArgs) {
        // Check declarations and uses together before removing their scope.
        let mut validator = ValidateBoundVars::new(self.bound_vars);
        validator.cx = Some(cx);
        validator.validate_telescope();
        let _ = self.value.visit_with(&mut validator);
        assert_eq!(
            self.ordinary_bound_var_count(),
            args.len(),
            "wrong number of arguments for binder instantiation: binder={self:?}, args={args:?}"
        );
        for (index, (bound_var, arg)) in self.ordinary_bound_vars().zip(args.iter()).enumerate() {
            let valid = matches!(
                (bound_var, arg.kind()),
                (BoundVariableKind::Region(_), ty::GenericArgKind::Lifetime(_))
                    | (BoundVariableKind::Ty(_), ty::GenericArgKind::Type(_))
                    | (BoundVariableKind::Const(_), ty::GenericArgKind::Const(_))
            );
            assert!(
                valid,
                "argument kind mismatch at binder index {index}: binder={self:?}, args={args:?}"
            );
        }
    }
}

/// Removes one late binder while substituting its ordinary variables.
///
/// Variables supplied by `args` are shifted into any nested binders traversed
/// while folding. Variables which were bound outside the removed binder are
/// shifted out once. Keeping both adjustments in one folder is what makes an
/// explicit instantiation reusable for a value and all dependent clauses.
struct LateBoundArgFolder<I: Interner> {
    cx: I,
    args: I::GenericArgs,
    current_index: DebruijnIndex,
    evidence: Option<LateBoundEvidenceArgs<I>>,
}

#[derive(Clone, Copy)]
struct LateBoundEvidenceArgs<I: Interner> {
    ordinary_count: usize,
    args: I::TraitEvidences,
}

impl<I: Interner> LateBoundArgFolder<I> {
    fn new(cx: I, args: I::GenericArgs) -> Self {
        LateBoundArgFolder { cx, args, current_index: ty::INNERMOST, evidence: None }
    }

    fn with_evidence(
        cx: I,
        args: I::GenericArgs,
        ordinary_count: usize,
        evidence: I::TraitEvidences,
    ) -> Self {
        LateBoundArgFolder {
            cx,
            args,
            current_index: ty::INNERMOST,
            evidence: Some(LateBoundEvidenceArgs { ordinary_count, args: evidence }),
        }
    }

    fn shifted_arg<T: TypeFoldable<I>>(&self, value: T) -> T {
        ty::shift_vars(self.cx, value, self.current_index.as_u32())
    }
}

impl<I: Interner> TypeFolder<I> for LateBoundArgFolder<I> {
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_binder<U: TypeFoldable<I>>(&mut self, binder: Binder<I, U>) -> Binder<I, U> {
        self.current_index.shift_in(1);
        let binder = binder.super_fold_with(self);
        self.current_index.shift_out(1);
        binder
    }

    fn fold_ty(&mut self, value: I::Ty) -> I::Ty {
        match value.kind() {
            ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_ty)
                if debruijn == self.current_index =>
            {
                let arg = self.args.get(bound_ty.var().as_usize()).unwrap_or_else(|| {
                    panic!("bound type {bound_ty:?} is outside instantiation {:#?}", self.args)
                });
                let ty::GenericArgKind::Type(ty) = arg.kind() else {
                    panic!("expected type argument for {bound_ty:?}, found {arg:?}")
                };
                self.shifted_arg(ty)
            }
            ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_ty)
                if debruijn > self.current_index =>
            {
                I::Ty::new_bound(self.cx, debruijn.shifted_out(1), bound_ty)
            }
            _ if value.has_vars_bound_at_or_above(self.current_index) => {
                value.super_fold_with(self)
            }
            _ => value,
        }
    }

    fn fold_region(&mut self, value: Region<I>) -> Region<I> {
        match value.kind() {
            ty::ReBound(ty::BoundVarIndexKind::Bound(debruijn), bound_region)
                if debruijn == self.current_index =>
            {
                let arg = self.args.get(bound_region.var().as_usize()).unwrap_or_else(|| {
                    panic!(
                        "bound region {bound_region:?} is outside instantiation {:#?}",
                        self.args
                    )
                });
                let ty::GenericArgKind::Lifetime(region) = arg.kind() else {
                    panic!("expected lifetime argument for {bound_region:?}, found {arg:?}")
                };
                ty::shift_region(self.cx, region, self.current_index.as_u32())
            }
            ty::ReBound(ty::BoundVarIndexKind::Bound(debruijn), bound_region)
                if debruijn > self.current_index =>
            {
                Region::new_bound(self.cx, debruijn.shifted_out(1), bound_region)
            }
            _ => value,
        }
    }

    fn fold_const(&mut self, value: I::Const) -> I::Const {
        match value.kind() {
            ty::ConstKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_const)
                if debruijn == self.current_index =>
            {
                let arg = self.args.get(bound_const.var().as_usize()).unwrap_or_else(|| {
                    panic!("bound const {bound_const:?} is outside instantiation {:#?}", self.args)
                });
                let ty::GenericArgKind::Const(ct) = arg.kind() else {
                    panic!("expected const argument for {bound_const:?}, found {arg:?}")
                };
                self.shifted_arg(ct)
            }
            ty::ConstKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_const)
                if debruijn > self.current_index =>
            {
                I::Const::new_bound(self.cx, debruijn.shifted_out(1), bound_const)
            }
            _ => value.super_fold_with(self),
        }
    }

    fn fold_trait_evidence(&mut self, evidence: I::TraitEvidence) -> I::TraitEvidence {
        match &evidence.kind {
            ty::solve::TraitEvidenceKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound)
                if *debruijn == self.current_index =>
            {
                let expected_trait_ref = evidence.trait_ref.fold_with(self);
                let Some(substitution) = self.evidence else {
                    panic!(
                        "instantiating bound evidence without an explicit evidence substitution: {evidence:?}"
                    );
                };
                let telescope_index = bound.var().as_usize();
                let suffix_index =
                    telescope_index.checked_sub(substitution.ordinary_count).unwrap_or_else(|| {
                        panic!(
                            "bound evidence points into the ordinary binder prefix: {evidence:?}"
                        )
                    });
                let replacement = substitution.args.get(suffix_index).unwrap_or_else(|| {
                    panic!(
                        "bound evidence index {telescope_index} is outside the binder evidence suffix"
                    )
                });
                let replacement = ty::shift_vars(self.cx, replacement, self.current_index.as_u32());
                replacement.assert_well_formed();
                assert_eq!(
                    replacement.trait_ref, expected_trait_ref,
                    "bound evidence replacement proves a different instantiated predicate"
                );
                replacement
            }
            ty::solve::TraitEvidenceKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound)
                if *debruijn > self.current_index =>
            {
                let trait_ref = evidence.trait_ref.fold_with(self);
                self.cx.mk_trait_evidence_kind(
                    trait_ref,
                    ty::solve::TraitEvidenceKind::Bound(
                        ty::BoundVarIndexKind::Bound(debruijn.shifted_out(1)),
                        *bound,
                    ),
                )
            }
            _ => self.cx.mk_trait_evidence_data((*evidence).clone().fold_with(self)),
        }
    }

    fn fold_predicate<P: PredicateProxy<I>>(&mut self, value: P) -> P {
        if value.has_vars_bound_at_or_above(self.current_index) {
            value.super_fold_with(self)
        } else {
            value
        }
    }

    fn fold_clauses(&mut self, value: I::Clauses) -> I::Clauses {
        if value.has_vars_bound_at_or_above(self.current_index) {
            value.super_fold_with(self)
        } else {
            value
        }
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeSuperVisitable<I> for Binder<I, T> {
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        for entry in self.bound_vars.iter() {
            try_visit!(entry.visit_with(visitor));
        }
        self.as_ref().skip_binder().visit_with(visitor)
    }
}

impl<I: Interner, T> Binder<I, T> {
    /// Returns the value contained inside of this `for<'a>`. Accessing generic args
    /// in the returned value is generally incorrect.
    ///
    /// Please read <https://rustc-dev-guide.rust-lang.org/ty_module/instantiating_binders.html>
    /// before using this function. It is usually better to discharge the binder using
    /// `no_bound_vars` or `instantiate_bound_regions` or something like that.
    ///
    /// `skip_binder` is only valid when you are either extracting data that does not reference
    /// any generic arguments, e.g. a `DefId`, or when you're making sure you only pass the
    /// value to things which can handle escaping bound vars.
    ///
    /// See existing uses of `.skip_binder()` in `rustc_trait_selection::traits::select`
    /// or `rustc_next_trait_solver` for examples.
    pub fn skip_binder(self) -> T {
        self.value
    }

    pub fn bound_vars(&self) -> I::BoundVarKinds {
        self.bound_vars
    }

    /// Returns the source-level lifetime/type/const entries in this telescope.
    ///
    /// Evidence entries are an internal suffix. Consumers which classify user
    /// generic parameters or allocate new `BoundVar` indices must use this view
    /// instead of treating every telescope entry as an ordinary bound variable.
    pub fn ordinary_bound_vars(&self) -> impl Iterator<Item = BoundVariableKind<I>> + '_ {
        self.bound_vars.iter().take_while(|entry| !matches!(entry, BoundVariableKind::Evidence(_)))
    }

    pub fn ordinary_bound_var_count(&self) -> usize {
        self.ordinary_bound_vars().count()
    }

    pub fn has_ordinary_bound_vars(&self) -> bool {
        self.ordinary_bound_vars().next().is_some()
    }

    /// Returns the compiler-internal proof assumptions owned by this binder,
    /// together with their stable telescope indices.
    ///
    /// Evidence entries are required to form a suffix, so exposing them does
    /// not change the indices of ordinary lifetime/type/const bound variables.
    pub fn evidence_bound_vars(&self) -> impl Iterator<Item = (u32, ty::ClauseKind<I>)> + '_ {
        self.bound_vars.iter().enumerate().filter_map(|(index, entry)| match entry {
            BoundVariableKind::Evidence(evidence) => Some((
                u32::try_from(index).expect("binder telescope index overflow"),
                evidence.clause,
            )),
            BoundVariableKind::Ty(_)
            | BoundVariableKind::Region(_)
            | BoundVariableKind::Const(_) => None,
        })
    }

    pub fn has_evidence_bound_vars(&self) -> bool {
        self.evidence_bound_vars().next().is_some()
    }

    /// Whether instantiating this binder must also instantiate clauses owned
    /// by telescope entries. Typed const declarations contribute a
    /// `ConstArgHasType` clause and evidence entries contribute their stored
    /// predicate.
    pub fn has_telescope_clauses(&self) -> bool {
        self.bound_vars.iter().any(|entry| {
            matches!(entry, BoundVariableKind::Const(Some(_)) | BoundVariableKind::Evidence(_))
        })
    }

    pub fn as_ref(&self) -> Binder<I, &T> {
        Binder { value: &self.value, bound_vars: self.bound_vars }
    }

    pub fn as_deref(&self) -> Binder<I, &T::Target>
    where
        T: Deref,
    {
        Binder { value: &self.value, bound_vars: self.bound_vars }
    }

    pub fn map_bound_ref<F, U: TypeVisitable<I>>(&self, f: F) -> Binder<I, U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U: TypeVisitable<I>>(self, f: F) -> Binder<I, U>
    where
        F: FnOnce(T) -> U,
    {
        let Binder { value, bound_vars } = self;
        let value = f(value);
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            validator.validate_telescope();
            let _ = value.visit_with(&mut validator);
        }
        Binder { value, bound_vars }
    }

    pub fn try_map_bound<F, U: TypeVisitable<I>, E>(self, f: F) -> Result<Binder<I, U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let Binder { value, bound_vars } = self;
        let value = f(value)?;
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            validator.validate_telescope();
            let _ = value.visit_with(&mut validator);
        }
        Ok(Binder { value, bound_vars })
    }

    /// Wraps a `value` in a binder, using the same bound variables as the
    /// current `Binder`. This should not be used if the new value *changes*
    /// the bound variables. Note: the (old or new) value itself does not
    /// necessarily need to *name* all the bound variables.
    ///
    /// This currently doesn't do anything different than `bind`, because we
    /// don't actually track bound vars. However, semantically, it is different
    /// because bound vars aren't allowed to change here, whereas they are
    /// in `bind`. This may be (debug) asserted in the future.
    pub fn rebind<U>(&self, value: U) -> Binder<I, U>
    where
        U: TypeVisitable<I>,
    {
        Binder::bind_with_vars(value, self.bound_vars)
    }

    /// Unwraps and returns the value within, but only if it contains
    /// no bound vars at all. (In other words, if this binder --
    /// and indeed any enclosing binder -- doesn't bind anything at
    /// all.) Dependent declarations must be explicitly instantiated, even if
    /// unused by the value. Otherwise, returns `None`.
    ///
    /// (One could imagine having a method that just unwraps a single
    /// binder, but permits late-bound vars bound by enclosing
    /// binders, but that would require adjusting the debruijn
    /// indices, and given the shallow binding structure we often use,
    /// would not be that useful.)
    pub fn no_bound_vars(self) -> Option<T>
    where
        T: TypeVisitable<I>,
    {
        // Dependent declarations must be explicitly instantiated, even if the
        // value does not refer to them. Unwrapping here would lose their clauses.
        if self.has_telescope_clauses() || self.value.has_escaping_bound_vars() {
            None
        } else {
            Some(self.skip_binder())
        }
    }
}

impl<I: Interner, T> Binder<I, Option<T>> {
    pub fn transpose(self) -> Option<Binder<I, T>> {
        let Binder { value, bound_vars } = self;
        value.map(|value| Binder { value, bound_vars })
    }
}

impl<I: Interner, T: IntoIterator> Binder<I, T> {
    pub fn iter(self) -> impl Iterator<Item = Binder<I, T::Item>> {
        let Binder { value, bound_vars } = self;
        value.into_iter().map(move |value| Binder { value, bound_vars })
    }
}

pub struct ValidateBoundVars<I: Interner> {
    bound_vars: I::BoundVarKinds,
    binder_index: ty::DebruijnIndex,
    // We only cache types because any complex const will have to step through
    // a type at some point anyways. We may encounter the same variable at
    // different levels of binding, so this can't just be `Ty`.
    visited: SsoHashSet<(ty::DebruijnIndex, I::Ty)>,
    /// While validating a telescope entry, references at the current
    /// binder may only target earlier entries.
    entry_limit: Option<usize>,
    cx: Option<I>,
}

impl<I: Interner> ValidateBoundVars<I> {
    pub fn new(bound_vars: I::BoundVarKinds) -> Self {
        ValidateBoundVars {
            bound_vars,
            binder_index: ty::INNERMOST,
            visited: SsoHashSet::default(),
            entry_limit: None,
            cx: None,
        }
    }

    fn validate_telescope(&mut self) {
        let mut saw_evidence = false;
        for (index, entry) in self.bound_vars.iter().enumerate() {
            match entry {
                BoundVariableKind::Evidence(evidence) => {
                    let _ = evidence_clause_trait_ref(evidence.clause);
                    if let Some(contract) = evidence.required_contract {
                        contract.assert_well_formed();
                        if let Some(cx) = self.cx {
                            let principal = contract
                                .clauses
                                .get(contract.principal_index as usize)
                                .expect("required-contract principal is out of bounds");
                            let expected = Binder::bind_with_vars(
                                ty::shift_vars(cx, evidence.clause, 1),
                                Default::default(),
                            );
                            assert_eq!(
                                principal.kind(),
                                expected,
                                "evidence declaration does not match its contract principal"
                            );
                        }
                    }
                    saw_evidence = true;
                }
                _ if saw_evidence => {
                    panic!("ordinary binder variable after evidence entry: {:?}", self.bound_vars)
                }
                _ => {}
            }
            self.entry_limit = Some(index);
            let _ = entry.visit_with(self);
        }
        self.entry_limit = None;
    }

    fn assert_in_entry_scope(&self, index: usize) {
        if let Some(limit) = self.entry_limit
            && index >= limit
        {
            panic!(
                "binder telescope entry references non-previous variable {index} in {:?}",
                self.bound_vars
            );
        }
    }
}

impl<I: Interner> TypeVisitor<I> for ValidateBoundVars<I> {
    type Result = ControlFlow<()>;

    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &Binder<I, T>) -> Self::Result {
        self.binder_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: I::Ty) -> Self::Result {
        if t.outer_exclusive_binder() < self.binder_index
            || !self.visited.insert((self.binder_index, t))
        {
            return ControlFlow::Continue(());
        }
        match t.kind() {
            ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_ty)
                if debruijn == self.binder_index =>
            {
                let idx = bound_ty.var().as_usize();
                if self.bound_vars.len() <= idx {
                    panic!("Not enough bound vars: {:?} not found in {:?}", t, self.bound_vars);
                }
                self.assert_in_entry_scope(idx);
                bound_ty.assert_eq(self.bound_vars.get(idx).unwrap());
            }
            _ => {}
        };

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: I::Const) -> Self::Result {
        if c.outer_exclusive_binder() < self.binder_index {
            return ControlFlow::Continue(());
        }
        match c.kind() {
            ty::ConstKind::Bound(debruijn, bound_const)
                if debruijn == ty::BoundVarIndexKind::Bound(self.binder_index) =>
            {
                let idx = bound_const.var().as_usize();
                if self.bound_vars.len() <= idx {
                    panic!("Not enough bound vars: {:?} not found in {:?}", c, self.bound_vars);
                }
                self.assert_in_entry_scope(idx);
                bound_const.assert_eq(self.bound_vars.get(idx).unwrap());
            }
            _ => {}
        };

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: Region<I>) -> Self::Result {
        match r.kind() {
            ty::ReBound(index, br) if index == ty::BoundVarIndexKind::Bound(self.binder_index) => {
                let idx = br.var().as_usize();
                if self.bound_vars.len() <= idx {
                    panic!("Not enough bound vars: {:?} not found in {:?}", r, self.bound_vars);
                }
                self.assert_in_entry_scope(idx);
                br.assert_eq(self.bound_vars.get(idx).unwrap());
            }

            _ => (),
        };

        ControlFlow::Continue(())
    }

    fn visit_trait_evidence(&mut self, evidence: I::TraitEvidence) -> Self::Result {
        if let ty::solve::TraitEvidenceKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound) =
            &evidence.kind
            && *debruijn == self.binder_index
        {
            let idx = bound.var().as_usize();
            if self.bound_vars.len() <= idx {
                panic!(
                    "Not enough bound vars: evidence {evidence:?} not found in {:?}",
                    self.bound_vars
                );
            }
            self.assert_in_entry_scope(idx);
            let clause = bound.assert_eq(self.bound_vars.get(idx).unwrap());
            let expected_trait_ref = evidence_clause_trait_ref(clause);
            if let Some(cx) = self.cx {
                let expected_trait_ref =
                    ty::shift_vars(cx, expected_trait_ref, self.binder_index.as_u32());
                assert_eq!(
                    evidence.trait_ref, expected_trait_ref,
                    "bound evidence predicate does not match binder telescope entry {idx}"
                );
            } else {
                // Binder construction has no interner with which to shift the declaration.
                // Explicit instantiation also checks the arguments at the current depth.
                assert_eq!(evidence.trait_ref.def_id, expected_trait_ref.def_id);
                if self.binder_index == ty::INNERMOST
                    || !expected_trait_ref.has_escaping_bound_vars()
                {
                    assert_eq!(evidence.trait_ref, expected_trait_ref);
                }
            }
        }

        (*evidence).visit_with(self)
    }
}

/// Similar to [`Binder`] except that it tracks early bound generics, i.e. `struct Foo<T>(T)`
/// needs `T` instantiated immediately. This type primarily exists to avoid forgetting to call
/// `instantiate`.
///
/// See <https://rustc-dev-guide.rust-lang.org/ty_module/early_binder.html> for more details.
#[derive_where(Clone, Copy, PartialOrd, Ord, PartialEq, Hash, Debug; I: Interner, T)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct EarlyBinder<I: Interner, T> {
    value: T,
    #[derive_where(skip(Debug))]
    _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner, T: Eq> Eq for EarlyBinder<I, T> {}

// FIXME(154045): Recommended as per https://github.com/rust-lang/rust/issues/154045, this is so sad :((
#[cfg(feature = "nightly")]
macro_rules! generate { ($( $tt:tt )*) => { $( $tt )* } }

#[cfg(feature = "nightly")]
generate!(
    /// For early binders, you should first call `instantiate` before using any visitors.
    impl<I: Interner, T> !TypeFoldable<I> for ty::EarlyBinder<I, T> {}
    /// For early binders, you should first call `instantiate` before using any visitors.
    impl<I: Interner, T> !TypeVisitable<I> for ty::EarlyBinder<I, T> {}
);

impl<I: Interner, T: TypeFoldable<I>> EarlyBinder<I, T> {
    pub fn bind(cx: I, value: T) -> EarlyBinder<I, T> {
        // Instantiation will require normalization.
        let value = ty::set_aliases_to_non_rigid(cx, value).skip_normalization();
        EarlyBinder { value, _tcx: PhantomData }
    }
}

impl<I: Interner, T: IntoIterator<Item: TypeVisitable<I>> + Clone> EarlyBinder<I, T> {
    pub fn bind_iter(value: T) -> EarlyBinder<I, T> {
        #[cfg(debug_assertions)]
        {
            value.clone().into_iter().for_each(|v| assert!(!v.has_rigid_aliases()));
        }

        EarlyBinder { value, _tcx: PhantomData }
    }
}

impl<I: Interner, T: TypeVisitable<I>> EarlyBinder<I, T> {
    pub fn bind_no_rigid_aliases(value: T) -> EarlyBinder<I, T> {
        debug_assert!(!value.has_rigid_aliases());
        EarlyBinder { value, _tcx: PhantomData }
    }
}

impl<I: Interner, T> EarlyBinder<I, T> {
    /// Use `bind/bind_iter/bind_no_rigid_aliases` instead.
    /// Don't use this unless you know what you're doing.
    pub fn bind_unchecked(value: T) -> EarlyBinder<I, T> {
        EarlyBinder { value, _tcx: PhantomData }
    }
}

impl<I: Interner, T> EarlyBinder<I, T> {
    pub fn as_ref(&self) -> EarlyBinder<I, &T> {
        EarlyBinder { value: &self.value, _tcx: PhantomData }
    }

    pub fn map_bound_ref<F, U>(&self, f: F) -> EarlyBinder<I, U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U>(self, f: F) -> EarlyBinder<I, U>
    where
        F: FnOnce(T) -> U,
    {
        let value = f(self.value);
        EarlyBinder { value, _tcx: PhantomData }
    }

    pub fn try_map_bound<F, U, E>(self, f: F) -> Result<EarlyBinder<I, U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let value = f(self.value)?;
        Ok(EarlyBinder { value, _tcx: PhantomData })
    }

    pub fn rebind<U>(&self, value: U) -> EarlyBinder<I, U> {
        EarlyBinder { value, _tcx: PhantomData }
    }

    /// Skips the binder and returns the "bound" value. Accessing generic args
    /// in the returned value is generally incorrect.
    ///
    /// Please read <https://rustc-dev-guide.rust-lang.org/ty_module/early_binder.html>
    /// before using this function.
    ///
    /// Only use this to extract data that does not depend on generic parameters, e.g.
    /// to get the `DefId` of the inner value or the number of arguments ofan `FnSig`,
    /// or while making sure to only pass the value to functions which are explicitly
    /// set up to handle these uninstantiated generic parameters.
    ///
    /// To skip the binder on `x: &EarlyBinder<I, T>` to obtain `&T`, leverage
    /// [`EarlyBinder::as_ref`](EarlyBinder::as_ref): `x.as_ref().skip_binder()`.
    ///
    /// See also [`Binder::skip_binder`](Binder::skip_binder), which is
    /// the analogous operation on [`Binder`].
    pub fn skip_binder(self) -> T {
        self.value
    }
}

impl<I: Interner> EarlyBinder<I, ty::TraitRef<I>> {
    pub fn def_id(&self) -> I::TraitId {
        self.value.def_id
    }
}

impl<I: Interner, T> EarlyBinder<I, Option<T>> {
    pub fn transpose(self) -> Option<EarlyBinder<I, T>> {
        self.value.map(|value| EarlyBinder { value, _tcx: PhantomData })
    }
}

impl<I: Interner, Iter: IntoIterator> EarlyBinder<I, Iter>
where
    Iter::Item: TypeFoldable<I>,
{
    pub fn iter_instantiated<A>(self, cx: I, args: A) -> IterInstantiated<I, Iter, A>
    where
        A: SliceLike<Item = I::GenericArg>,
    {
        IterInstantiated { it: self.value.into_iter(), cx, args }
    }

    /// Similar to [`instantiate_identity`](EarlyBinder::instantiate_identity),
    /// but on an iterator of `TypeFoldable` values.
    pub fn iter_identity(self) -> impl Iterator<Item = Unnormalized<I, Iter::Item>> {
        self.value.into_iter().map(Unnormalized::new)
    }
}

pub struct IterInstantiated<I: Interner, Iter: IntoIterator, A> {
    it: Iter::IntoIter,
    cx: I,
    args: A,
}

impl<I: Interner, Iter: IntoIterator, A> Iterator for IterInstantiated<I, Iter, A>
where
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
    type Item = Unnormalized<I, Iter::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
            EarlyBinder { value: self.it.next()?, _tcx: PhantomData }
                .instantiate(self.cx, self.args),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I: Interner, Iter: IntoIterator, A> DoubleEndedIterator for IterInstantiated<I, Iter, A>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(
            EarlyBinder { value: self.it.next_back()?, _tcx: PhantomData }
                .instantiate(self.cx, self.args),
        )
    }
}

impl<I: Interner, Iter: IntoIterator, A> ExactSizeIterator for IterInstantiated<I, Iter, A>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
}

impl<'s, I: Interner, Iter: IntoIterator> EarlyBinder<I, Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    pub fn iter_instantiated_copied(
        self,
        cx: I,
        args: &'s [I::GenericArg],
    ) -> IterInstantiatedCopied<'s, I, Iter> {
        IterInstantiatedCopied { it: self.value.into_iter(), cx, args }
    }

    /// Similar to [`instantiate_identity`](EarlyBinder::instantiate_identity),
    /// but on an iterator of values that deref to a `TypeFoldable`.
    pub fn iter_identity_copied(self) -> IterIdentityCopied<I, Iter> {
        IterIdentityCopied { it: self.value.into_iter(), _tcx: PhantomData }
    }
}

pub struct IterInstantiatedCopied<'a, I: Interner, Iter: IntoIterator> {
    it: Iter::IntoIter,
    cx: I,
    args: &'a [I::GenericArg],
}

impl<'a, I: Interner, Iter: IntoIterator<IntoIter: Clone>> Clone
    for IterInstantiatedCopied<'a, I, Iter>
{
    fn clone(&self) -> IterInstantiatedCopied<'a, I, Iter> {
        IterInstantiatedCopied { it: self.it.clone(), cx: self.cx, args: self.args }
    }
}

impl<I: Interner, Iter: IntoIterator> Iterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    type Item = Unnormalized<I, <Iter::Item as Deref>::Target>;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|value| {
            EarlyBinder { value: *value, _tcx: PhantomData }.instantiate(self.cx, self.args)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I: Interner, Iter: IntoIterator> DoubleEndedIterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(|value| {
            EarlyBinder { value: *value, _tcx: PhantomData }.instantiate(self.cx, self.args)
        })
    }
}

impl<I: Interner, Iter: IntoIterator> ExactSizeIterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
}

pub struct IterIdentityCopied<I: Interner, Iter: IntoIterator> {
    it: Iter::IntoIter,
    _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner, Iter: IntoIterator<IntoIter: Clone>> Clone for IterIdentityCopied<I, Iter> {
    fn clone(&self) -> IterIdentityCopied<I, Iter> {
        IterIdentityCopied { it: self.it.clone(), _tcx: self._tcx }
    }
}

impl<I: Interner, Iter: IntoIterator> Iterator for IterIdentityCopied<I, Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
    type Item = Unnormalized<I, <Iter::Item as Deref>::Target>;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|i| Unnormalized::new(*i))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I: Interner, Iter: IntoIterator> DoubleEndedIterator for IterIdentityCopied<I, Iter>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(|i| Unnormalized::new(*i))
    }
}

impl<I: Interner, Iter: IntoIterator> ExactSizeIterator for IterIdentityCopied<I, Iter>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
}
pub struct EarlyBinderIter<I, T> {
    t: T,
    _tcx: PhantomData<I>,
}

impl<I: Interner, T: IntoIterator> EarlyBinder<I, T> {
    pub fn transpose_iter(self) -> EarlyBinderIter<I, T::IntoIter> {
        EarlyBinderIter { t: self.value.into_iter(), _tcx: PhantomData }
    }
}

impl<I: Interner, T: Iterator> Iterator for EarlyBinderIter<I, T> {
    type Item = EarlyBinder<I, T::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.t.next().map(|value| EarlyBinder { value, _tcx: PhantomData })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.t.size_hint()
    }
}

impl<I: Interner, T: TypeFoldable<I>> ty::EarlyBinder<I, T> {
    pub fn instantiate<A>(self, cx: I, args: A) -> Unnormalized<I, T>
    where
        A: SliceLike<Item = I::GenericArg>,
    {
        // Nothing to fold, so let's avoid visiting things and possibly re-hashing/equating
        // them when interning. Perf testing found this to be a modest improvement.
        // See: <https://github.com/rust-lang/rust/pull/142317>
        if args.is_empty() {
            assert!(
                !self.value.has_param(),
                "{:?} has parameters, but no args were provided in instantiate",
                self.value,
            );
            return Unnormalized::new(self.value);
        }
        let mut folder = ArgFolder { cx, args: args.as_slice(), binders_passed: 0 };
        Unnormalized::new(self.value.fold_with(&mut folder))
    }

    /// Makes the identity replacement `T0 => T0, ..., TN => TN`.
    /// Conceptually, this converts universally bound variables into placeholders
    /// when inside of a given item.
    ///
    /// For example, consider `for<T> fn foo<T>(){ .. }`:
    /// - Outside of `foo`, `T` is bound (represented by the presence of `EarlyBinder`).
    /// - Inside of the body of `foo`, we treat `T` as a placeholder by calling
    /// `instantiate_identity` to discharge the `EarlyBinder`.
    pub fn instantiate_identity(self) -> Unnormalized<I, T> {
        // FIXME(#155345): In case the bound value was already normalized, this
        // is unnecessary. We may want to track explicitly whether `EarlyBinder`
        // contains something that has been normalized already.
        // Also do that for other types who have `instantiate_identity` method,
        // e.g., `GenericClauses` and `ConstConditions`.
        //
        // This is annoying, as e.g. `type_of` for opaque types is normalized,
        // while `type_of` for free type aliases is not.
        Unnormalized::new(self.value)
    }

    /// Returns the inner value, but only if it contains no bound vars.
    pub fn no_bound_vars(self) -> Option<T> {
        if !self.value.has_param() { Some(self.value) } else { None }
    }
}

///////////////////////////////////////////////////////////////////////////
// The actual instantiation engine itself is a type folder.

struct ArgFolder<'a, I: Interner> {
    cx: I,
    args: &'a [I::GenericArg],

    /// Number of region binders we have passed through while doing the instantiation
    binders_passed: u32,
}

impl<'a, I: Interner> TypeFolder<I> for ArgFolder<'a, I> {
    #[inline]
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_binder<T: TypeFoldable<I>>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T> {
        self.binders_passed += 1;
        let t = t.super_fold_with(self);
        self.binders_passed -= 1;
        t
    }

    fn fold_region(&mut self, r: Region<I>) -> Region<I> {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region instantiation of the bound
        // regions that appear in a function signature is done using
        // the specialized routine `ty::replace_late_regions()`.
        match r.kind() {
            ty::ReEarlyParam(data) => {
                let rk = self.args.get(data.index() as usize).map(|arg| arg.kind());
                match rk {
                    Some(ty::GenericArgKind::Lifetime(lt)) => self.shift_region_through_binders(lt),
                    Some(other) => self.region_param_expected(data, r, other),
                    None => self.region_param_out_of_range(data, r),
                }
            }
            ty::ReBound(..)
            | ty::ReLateParam(_)
            | ty::ReStatic
            | ty::RePlaceholder(_)
            | ty::ReErased
            | ty::ReError(_) => r,
            ty::ReVar(_) => panic!("unexpected region: {r:?}"),
        }
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        if !t.has_param() {
            return t;
        }

        match t.kind() {
            ty::Param(p) => self.ty_for_param(p, t),
            _ => t.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const {
        if let ty::ConstKind::Param(p) = c.kind() {
            self.const_for_param(p, c)
        } else {
            c.super_fold_with(self)
        }
    }

    fn fold_predicate<P: PredicateProxy<I>>(&mut self, p: P) -> P {
        if p.has_param() { p.super_fold_with(self) } else { p }
    }

    fn fold_clauses(&mut self, c: I::Clauses) -> I::Clauses {
        if c.has_param() { c.super_fold_with(self) } else { c }
    }
}

impl<'a, I: Interner> ArgFolder<'a, I> {
    fn ty_for_param(&self, p: I::ParamTy, source_ty: I::Ty) -> I::Ty {
        // Look up the type in the args. It really should be in there.
        let opt_ty = self.args.get(p.index() as usize).map(|arg| arg.kind());
        let ty = match opt_ty {
            Some(ty::GenericArgKind::Type(ty)) => ty,
            Some(kind) => self.type_param_expected(p, source_ty, kind),
            None => self.type_param_out_of_range(p, source_ty),
        };

        self.shift_vars_through_binders(ty)
    }

    #[cold]
    #[inline(never)]
    fn type_param_expected(&self, p: I::ParamTy, ty: I::Ty, kind: ty::GenericArgKind<I>) -> ! {
        panic!(
            "expected type for `{:?}` ({:?}/{}) but found {:?} when instantiating, args={:?}",
            p,
            ty,
            p.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn type_param_out_of_range(&self, p: I::ParamTy, ty: I::Ty) -> ! {
        panic!(
            "type parameter `{:?}` ({:?}/{}) out of range when instantiating, args={:?}",
            p,
            ty,
            p.index(),
            self.args,
        )
    }

    fn const_for_param(&self, p: I::ParamConst, source_ct: I::Const) -> I::Const {
        // Look up the const in the args. It really should be in there.
        let opt_ct = self.args.get(p.index() as usize).map(|arg| arg.kind());
        let ct = match opt_ct {
            Some(ty::GenericArgKind::Const(ct)) => ct,
            Some(kind) => self.const_param_expected(p, source_ct, kind),
            None => self.const_param_out_of_range(p, source_ct),
        };

        self.shift_vars_through_binders(ct)
    }

    #[cold]
    #[inline(never)]
    fn const_param_expected(
        &self,
        p: I::ParamConst,
        ct: I::Const,
        kind: ty::GenericArgKind<I>,
    ) -> ! {
        panic!(
            "expected const for `{:?}` ({:?}/{}) but found {:?} when instantiating args={:?}",
            p,
            ct,
            p.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn const_param_out_of_range(&self, p: I::ParamConst, ct: I::Const) -> ! {
        panic!(
            "const parameter `{:?}` ({:?}/{}) out of range when instantiating args={:?}",
            p,
            ct,
            p.index(),
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn region_param_expected(
        &self,
        ebr: I::EarlyParamRegion,
        r: Region<I>,
        kind: ty::GenericArgKind<I>,
    ) -> ! {
        panic!(
            "expected region for `{:?}` ({:?}/{}) but found {:?} when instantiating args={:?}",
            ebr,
            r,
            ebr.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn region_param_out_of_range(&self, ebr: I::EarlyParamRegion, r: Region<I>) -> ! {
        panic!(
            "region parameter `{:?}` ({:?}/{}) out of range when instantiating args={:?}",
            ebr,
            r,
            ebr.index(),
            self.args,
        )
    }

    /// It is sometimes necessary to adjust the De Bruijn indices during instantiation. This occurs
    /// when we are instantiating a type with escaping bound vars into a context where we have
    /// passed through binders. That's quite a mouthful. Let's see an example:
    ///
    /// ```
    /// type Func<A> = fn(A);
    /// type MetaFunc = for<'a> fn(Func<&'a i32>);
    /// ```
    ///
    /// The type `MetaFunc`, when fully expanded, will be
    /// ```ignore (illustrative)
    /// for<'a> fn(fn(&'a i32))
    /// //      ^~ ^~ ^~~
    /// //      |  |  |
    /// //      |  |  DebruijnIndex of 2
    /// //      Binders
    /// ```
    /// Here the `'a` lifetime is bound in the outer function, but appears as an argument of the
    /// inner one. Therefore, that appearance will have a DebruijnIndex of 2, because we must skip
    /// over the inner binder (remember that we count De Bruijn indices from 1). However, in the
    /// definition of `MetaFunc`, the binder is not visible, so the type `&'a i32` will have a
    /// De Bruijn index of 1. It's only during the instantiation that we can see we must increase the
    /// depth by 1 to account for the binder that we passed through.
    ///
    /// As a second example, consider this twist:
    ///
    /// ```
    /// type FuncTuple<A> = (A,fn(A));
    /// type MetaFuncTuple = for<'a> fn(FuncTuple<&'a i32>);
    /// ```
    ///
    /// Here the final type will be:
    /// ```ignore (illustrative)
    /// for<'a> fn((&'a i32, fn(&'a i32)))
    /// //          ^~~         ^~~
    /// //          |           |
    /// //   DebruijnIndex of 1 |
    /// //               DebruijnIndex of 2
    /// ```
    /// As indicated in the diagram, here the same type `&'a i32` is instantiated once, but in the
    /// first case we do not increase the De Bruijn index and in the second case we do. The reason
    /// is that only in the second case have we passed through a fn binder.
    #[instrument(level = "trace", skip(self), fields(binders_passed = self.binders_passed), ret)]
    fn shift_vars_through_binders<T: TypeFoldable<I>>(&self, val: T) -> T {
        if self.binders_passed == 0 || !val.has_escaping_bound_vars() {
            val
        } else {
            ty::shift_vars(self.cx, val, self.binders_passed)
        }
    }

    fn shift_region_through_binders(&self, region: Region<I>) -> Region<I> {
        if self.binders_passed == 0 || !region.has_escaping_bound_vars() {
            region
        } else {
            ty::shift_region(self.cx, region, self.binders_passed)
        }
    }
}

/// Okay, we do something fun for `Bound` types/regions/consts:
/// Specifically, we distinguish between *canonically* bound things and
/// `for<>` bound things. And, really, it comes down to caching during
/// canonicalization and instantiation.
///
/// To understand why we do this, imagine we have a type `(T, for<> fn(T))`.
/// If we just tracked canonically bound types with a `DebruijnIndex` (as we
/// used to), then the canonicalized type would be something like
/// `for<0> (^0.0, for<> fn(^1.0))` and so we can't cache `T -> ^0.0`,
/// we have to also factor in binder level. (Of course, we don't cache that
/// exactly, but rather the entire enclosing type, but the point stands.)
///
/// Of course, this is okay because we don't ever nest canonicalization, so
/// `BoundVarIndexKind::Canonical` is unambiguous. We, alternatively, could
/// have some sentinel `DebruijinIndex`, but that just seems too scary.
///
/// This doesn't seem to have a huge perf swing either way, but in the next
/// solver, canonicalization is hot and there are some pathological cases where
/// this is needed (`post-mono-higher-ranked-hang`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(Encodable_NoContext, Decodable_NoContext, StableHash))]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
pub enum BoundVarIndexKind {
    Bound(DebruijnIndex),
    Canonical,
}

/// The "placeholder index" fully defines a placeholder region, type, or const. Placeholders are
/// identified by both a universe, as well as a name residing within that universe. Distinct bound
/// regions/types/consts within the same universe simply have an unknown relationship to one
#[derive_where(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash; I: Interner, T)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, GenericTypeVisitable, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct Placeholder<I: Interner, T> {
    #[lift(identity)]
    pub universe: UniverseIndex,
    pub bound: T,
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner, T: fmt::Debug> fmt::Debug for ty::Placeholder<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.universe == ty::UniverseIndex::ROOT {
            write!(f, "!{:?}", self.bound)
        } else {
            write!(f, "!{}_{:?}", self.universe.index(), self.bound)
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[derive(Lift_Generic, GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub enum BoundRegionKind<I: Interner> {
    /// An anonymous region parameter for a given fn (&T)
    Anon,

    /// An anonymous region parameter with a `Symbol` name.
    ///
    /// Used to give late-bound regions names for things like pretty printing.
    NamedForPrinting(I::Symbol),

    /// Late-bound regions that appear in the AST.
    Named(I::DefId),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    ClosureEnv,
}

impl<I: Interner> fmt::Debug for ty::BoundRegionKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::BoundRegionKind::Anon => write!(f, "BrAnon"),
            ty::BoundRegionKind::NamedForPrinting(name) => {
                write!(f, "BrNamedForPrinting({:?})", name)
            }
            ty::BoundRegionKind::Named(did) => {
                write!(f, "BrNamed({did:?})")
            }
            ty::BoundRegionKind::ClosureEnv => write!(f, "BrEnv"),
        }
    }
}

impl<I: Interner> BoundRegionKind<I> {
    pub fn is_named(&self, tcx: I) -> bool {
        self.get_name(tcx).is_some()
    }

    pub fn get_name(&self, tcx: I) -> Option<I::Symbol> {
        match *self {
            ty::BoundRegionKind::Named(def_id) => {
                let name = tcx.item_name(def_id);
                if name == I::Symbol::KW_UNDERSCORE_LIFETIME { None } else { Some(name) }
            }
            ty::BoundRegionKind::NamedForPrinting(name) => Some(name),
            _ => None,
        }
    }

    pub fn get_id(&self) -> Option<I::DefId> {
        match *self {
            ty::BoundRegionKind::Named(id) => Some(id),
            _ => None,
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Debug, Hash; I: Interner)]
#[derive(Lift_Generic, GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub enum BoundTyKind<I: Interner> {
    Anon,
    Param(I::DefId),
}

#[derive_where(Clone, Copy, PartialEq, Eq, Debug, Hash; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub enum BoundVariableKind<I: Interner> {
    Ty(BoundTyKind<I>),
    Region(BoundRegionKind<I>),
    /// A const binder entry. New generalized binders record the binder-scoped
    /// const type; `None` is retained as a migration representation for legacy
    /// syntax-only binder construction sites.
    Const(Option<I::Ty>),
    /// Compiler-internal proof assumption owned by this binder. Evidence
    /// entries form a dependent suffix and may reference only earlier entries
    /// of the same telescope.
    Evidence(EvidenceVariable<I>),
}

impl<I: Interner> TypeVisitable<I> for BoundVariableKind<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match self {
            BoundVariableKind::Const(Some(ty)) => ty.visit_with(visitor),
            BoundVariableKind::Evidence(evidence) => evidence.visit_with(visitor),
            BoundVariableKind::Ty(_)
            | BoundVariableKind::Region(_)
            | BoundVariableKind::Const(None) => V::Result::output(),
        }
    }
}

impl<I: Interner> TypeFoldable<I> for BoundVariableKind<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(match self {
            BoundVariableKind::Const(Some(ty)) => {
                BoundVariableKind::Const(Some(ty.try_fold_with(folder)?))
            }
            BoundVariableKind::Evidence(evidence) => {
                BoundVariableKind::Evidence(evidence.try_fold_with(folder)?)
            }
            _ => self,
        })
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        match self {
            BoundVariableKind::Const(Some(ty)) => {
                BoundVariableKind::Const(Some(ty.fold_with(folder)))
            }
            BoundVariableKind::Evidence(evidence) => {
                BoundVariableKind::Evidence(evidence.fold_with(folder))
            }
            _ => self,
        }
    }
}

impl<I: Interner> BoundVariableKind<I> {
    pub fn expect_region(self) -> BoundRegionKind<I> {
        match self {
            BoundVariableKind::Region(lt) => lt,
            _ => panic!("expected a region, but found another kind"),
        }
    }

    pub fn expect_ty(self) -> BoundTyKind<I> {
        match self {
            BoundVariableKind::Ty(ty) => ty,
            _ => panic!("expected a type, but found another kind"),
        }
    }

    pub fn expect_const(self) {
        match self {
            BoundVariableKind::Const(_) => (),
            _ => panic!("expected a const, but found another kind"),
        }
    }

    pub fn const_ty(self) -> Option<I::Ty> {
        match self {
            BoundVariableKind::Const(ty) => ty,
            _ => panic!("expected a const, but found another kind"),
        }
    }

    pub fn expect_evidence(self) -> ty::ClauseKind<I> {
        match self {
            BoundVariableKind::Evidence(evidence) => evidence.clause,
            _ => panic!("expected evidence, but found another kind"),
        }
    }

    pub fn expect_evidence_variable(self) -> EvidenceVariable<I> {
        match self {
            BoundVariableKind::Evidence(evidence) => evidence,
            _ => panic!("expected evidence, but found another kind"),
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, StableHash_NoContext, Decodable_NoContext)
)]
pub struct BoundRegion<I: Interner> {
    pub var: ty::BoundVar,
    pub kind: BoundRegionKind<I>,
}

impl<I: Interner> core::fmt::Debug for BoundRegion<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            BoundRegionKind::Anon => write!(f, "{:?}", self.var),
            BoundRegionKind::ClosureEnv => write!(f, "{:?}.Env", self.var),
            BoundRegionKind::Named(def) => {
                write!(f, "{:?}.Named({:?})", self.var, def)
            }
            BoundRegionKind::NamedForPrinting(symbol) => {
                write!(f, "{:?}.NamedAnon({:?})", self.var, symbol)
            }
        }
    }
}

impl<I: Interner> BoundRegion<I> {
    pub fn var(self) -> ty::BoundVar {
        self.var
    }

    pub fn assert_eq(self, var: BoundVariableKind<I>) {
        assert_eq!(self.kind, var.expect_region())
    }
}

pub type PlaceholderRegion<I> = ty::Placeholder<I, BoundRegion<I>>;

impl<I: Interner> PlaceholderRegion<I> {
    pub fn universe(self) -> UniverseIndex {
        self.universe
    }

    pub fn var(self) -> ty::BoundVar {
        self.bound.var()
    }

    pub fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Self { universe: ui, bound: self.bound, _tcx: PhantomData }
    }

    pub fn new(ui: UniverseIndex, bound: BoundRegion<I>) -> Self {
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn new_anon(ui: UniverseIndex, var: ty::BoundVar) -> Self {
        let bound = BoundRegion { var, kind: BoundRegionKind::Anon };
        Self { universe: ui, bound, _tcx: PhantomData }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[derive(GenericTypeVisitable, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct BoundTy<I: Interner> {
    #[lift(identity)]
    pub var: ty::BoundVar,
    pub kind: BoundTyKind<I>,
}

impl<I: Interner> fmt::Debug for ty::BoundTy<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ty::BoundTyKind::Anon => write!(f, "{:?}", self.var),
            ty::BoundTyKind::Param(def_id) => write!(f, "{def_id:?}"),
        }
    }
}

impl<I: Interner> BoundTy<I> {
    pub fn var(self) -> ty::BoundVar {
        self.var
    }

    pub fn assert_eq(self, var: BoundVariableKind<I>) {
        assert_eq!(self.kind, var.expect_ty())
    }
}

pub type PlaceholderType<I> = ty::Placeholder<I, BoundTy<I>>;

impl<I: Interner> PlaceholderType<I> {
    pub fn universe(self) -> UniverseIndex {
        self.universe
    }

    pub fn var(self) -> ty::BoundVar {
        self.bound.var
    }

    pub fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Self { universe: ui, bound: self.bound, _tcx: PhantomData }
    }

    pub fn new(ui: UniverseIndex, bound: BoundTy<I>) -> Self {
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn new_anon(ui: UniverseIndex, var: ty::BoundVar) -> Self {
        let bound = BoundTy { var, kind: BoundTyKind::Anon };
        Self { universe: ui, bound, _tcx: PhantomData }
    }
}

#[derive_where(Clone, Copy, PartialEq, Debug, Eq, Hash; I: Interner)]
#[derive(GenericTypeVisitable, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct BoundConst<I: Interner> {
    #[lift(identity)]
    pub var: ty::BoundVar,
    #[derive_where(skip(Debug))]
    pub _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner> BoundConst<I> {
    pub fn var(self) -> ty::BoundVar {
        self.var
    }

    pub fn assert_eq(self, var: BoundVariableKind<I>) {
        var.expect_const()
    }

    pub fn new(var: ty::BoundVar) -> Self {
        Self { var, _tcx: PhantomData }
    }
}

pub type PlaceholderConst<I> = ty::Placeholder<I, BoundConst<I>>;

impl<I: Interner> PlaceholderConst<I> {
    pub fn universe(self) -> UniverseIndex {
        self.universe
    }

    pub fn var(self) -> ty::BoundVar {
        self.bound.var
    }

    pub fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Self { universe: ui, bound: self.bound, _tcx: PhantomData }
    }

    pub fn new(ui: UniverseIndex, bound: BoundConst<I>) -> Self {
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn new_anon(ui: UniverseIndex, var: ty::BoundVar) -> Self {
        let bound = BoundConst::new(var);
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn find_const_ty_from_env(self, env: I::ParamEnv) -> I::Ty {
        let mut candidates = env.caller_bounds().filter_map(|clause| {
            // `ConstArgHasType` are never desugared to be higher ranked.
            match clause.kind().skip_binder() {
                ty::ClauseKind::ConstArgHasType(placeholder_ct, ty) => {
                    assert!(!(placeholder_ct, ty).has_escaping_bound_vars());

                    match placeholder_ct.kind() {
                        ty::ConstKind::Placeholder(placeholder_ct) if placeholder_ct == self => {
                            Some(ty)
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        });

        // N.B. it may be tempting to fix ICEs by making this function return
        // `Option<Ty<'tcx>>` instead of `Ty<'tcx>`; however, this is generally
        // considered to be a bandaid solution, since it hides more important
        // underlying issues with how we construct generics and predicates of
        // items. It's advised to fix the underlying issue rather than trying
        // to modify this function.
        let ty = candidates.next().unwrap_or_else(|| {
            panic!("cannot find `{self:?}` in param-env: {env:#?}");
        });
        assert!(
            candidates.next().is_none(),
            "did not expect duplicate `ConstParamHasTy` for `{self:?}` in param-env: {env:#?}"
        );
        ty
    }
}

/// The binder-local identity of a trait evidence value.
///
/// Evidence variables share the telescope index space with ordinary bound
/// variables, but occupy a compiler-internal suffix and are not passed through
/// [`GenericArgs`](crate::GenericArgs). Keeping the index in its own type makes
/// it impossible to accidentally treat a proof as a type or const argument.
#[derive_where(Clone, Copy, PartialEq, Debug, Eq, Hash; I: Interner)]
#[derive(GenericTypeVisitable, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct BoundEvidence<I: Interner> {
    #[lift(identity)]
    pub var: ty::BoundVar,
    #[derive_where(skip(Debug))]
    pub _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner> BoundEvidence<I> {
    pub fn var(self) -> ty::BoundVar {
        self.var
    }

    pub fn assert_eq(self, var: BoundVariableKind<I>) -> ty::ClauseKind<I> {
        var.expect_evidence()
    }

    pub fn new(var: ty::BoundVar) -> Self {
        Self { var, _tcx: PhantomData }
    }
}

pub type PlaceholderEvidence<I> = ty::Placeholder<I, BoundEvidence<I>>;

impl<I: Interner> PlaceholderEvidence<I> {
    pub fn universe(self) -> UniverseIndex {
        self.universe
    }

    pub fn var(self) -> ty::BoundVar {
        self.bound.var
    }

    pub fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Self { universe: ui, bound: self.bound, _tcx: PhantomData }
    }

    pub fn new(ui: UniverseIndex, bound: BoundEvidence<I>) -> Self {
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn new_anon(ui: UniverseIndex, var: ty::BoundVar) -> Self {
        Self::new(ui, BoundEvidence::new(var))
    }
}
