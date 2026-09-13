use std::iter;

use derive_where::derive_where;
use rustc_ast_ir::Mutability;
use tracing::{instrument, trace};

use crate::error::{ExpectedFound, TypeError};
use crate::fold::TypeFoldable;
use crate::inherent::*;
use crate::{self as ty, Interner, Region, Upcast};

pub mod combine;
pub mod solver_relating;

pub type RelateResult<I, T> = Result<T, TypeError<I>>;

/// Extra information about why we ended up with a particular variance.
/// This is only used to add more information to error messages, and
/// has no effect on soundness. While choosing the 'wrong' `VarianceDiagInfo`
/// may lead to confusing notes in error messages, it will never cause
/// a miscompilation or unsoundness.
///
/// When in doubt, use `VarianceDiagInfo::default()`
#[derive_where(Clone, Copy, PartialEq, Debug, Default; I: Interner)]
pub enum VarianceDiagInfo<I: Interner> {
    /// No additional information - this is the default.
    /// We will not add any additional information to error messages.
    #[derive_where(default)]
    None,
    /// We switched our variance because a generic argument occurs inside
    /// the invariant generic argument of another type.
    Invariant {
        /// The generic type containing the generic parameter
        /// that changes the variance (e.g. `*mut T`, `MyStruct<T>`)
        ty: I::Ty,
        /// The index of the generic parameter being used
        /// (e.g. `0` for `*mut T`, `1` for `MyStruct<'CovariantParam, 'InvariantParam>`)
        param_index: u32,
    },
}

impl<I: Interner> Eq for VarianceDiagInfo<I> {}

impl<I: Interner> VarianceDiagInfo<I> {
    /// Mirrors `Variance::xform` - used to 'combine' the existing
    /// and new `VarianceDiagInfo`s when our variance changes.
    pub fn xform(self, other: VarianceDiagInfo<I>) -> VarianceDiagInfo<I> {
        // For now, just use the first `VarianceDiagInfo::Invariant` that we see
        match self {
            VarianceDiagInfo::None => other,
            VarianceDiagInfo::Invariant { .. } => self,
        }
    }
}

pub trait TypeRelation<I: Interner>: Sized {
    fn cx(&self) -> I;

    /// Generic relation routine suitable for most anything.
    fn relate<T: Relate<I>>(&mut self, a: T, b: T) -> RelateResult<I, T> {
        Relate::relate(self, a, b)
    }

    fn relate_ty_args(
        &mut self,
        a_ty: I::Ty,
        b_ty: I::Ty,
        ty_def_id: I::DefId,
        a_args: I::GenericArgs,
        b_args: I::GenericArgs,
        mk: impl FnOnce(I::GenericArgs) -> I::Ty,
    ) -> RelateResult<I, I::Ty>;

    /// Switch variance for the purpose of relating `a` and `b`.
    fn relate_with_variance<T: Relate<I>>(
        &mut self,
        variance: ty::Variance,
        info: VarianceDiagInfo<I>,
        a: T,
        b: T,
    ) -> RelateResult<I, T>;

    // Overridable relations. You shouldn't typically call these
    // directly, instead call `relate()`, which in turn calls
    // these. This is both more uniform but also allows us to add
    // additional hooks for other types in the future if needed
    // without making older code, which called `relate`, obsolete.

    fn tys(&mut self, a: I::Ty, b: I::Ty) -> RelateResult<I, I::Ty>;

    fn regions(&mut self, a: Region<I>, b: Region<I>) -> RelateResult<I, Region<I>>;

    fn consts(&mut self, a: I::Const, b: I::Const) -> RelateResult<I, I::Const>;

    /// Relates the proof identities carried by evidence-indexed projections.
    ///
    /// Evidence is intentionally not a `GenericArg`, so it needs an explicit
    /// relation hook just like types and consts. The default is the strict
    /// structural relation for fully-instantiated proofs. Inference relations
    /// override this hook to unify an evidence variable with the checked proof
    /// on the other side; they must still relate the proved trait refs first.
    fn evidences(
        &mut self,
        a: I::TraitEvidence,
        b: I::TraitEvidence,
    ) -> RelateResult<I, I::TraitEvidence> {
        relate_trait_evidence_invariantly(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<I, T>,
        b: ty::Binder<I, T>,
    ) -> RelateResult<I, ty::Binder<I, T>>
    where
        T: Relate<I>;
}

pub trait Relate<I: Interner>: TypeFoldable<I> + PartialEq + Copy {
    fn relate<R: TypeRelation<I>>(relation: &mut R, a: Self, b: Self) -> RelateResult<I, Self>;
}

///////////////////////////////////////////////////////////////////////////
// Relate impls

#[inline]
pub fn relate_args_invariantly<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a_arg: I::GenericArgs,
    b_arg: I::GenericArgs,
) -> RelateResult<I, I::GenericArgs> {
    relation.cx().mk_args_from_iter(iter::zip(a_arg.iter(), b_arg.iter()).map(|(a, b)| {
        relation.relate_with_variance(ty::Invariant, VarianceDiagInfo::default(), a, b)
    }))
}

pub fn relate_args_with_variances<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    variances: I::VariancesOf,
    a_args: I::GenericArgs,
    b_args: I::GenericArgs,
) -> RelateResult<I, I::GenericArgs> {
    let cx = relation.cx();
    let args = iter::zip(a_args.iter(), b_args.iter()).enumerate().map(|(i, (a, b))| {
        let variance = variances.get(i).unwrap();
        relation.relate_with_variance(variance, VarianceDiagInfo::None, a, b)
    });
    // FIXME: We can probably try to reuse `a_args` here if it did not change.
    cx.mk_args_from_iter(args)
}

impl<I: Interner> Relate<I> for ty::FnSig<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::FnSig<I>,
        b: ty::FnSig<I>,
    ) -> RelateResult<I, ty::FnSig<I>> {
        let cx = relation.cx();

        if a.c_variadic() != b.c_variadic() {
            return Err(TypeError::VariadicMismatch(ExpectedFound::new(
                a.c_variadic(),
                b.c_variadic(),
            )));
        }

        if a.safety() != b.safety() {
            return Err(TypeError::SafetyMismatch(ExpectedFound::new(a.safety(), b.safety())));
        }

        if a.abi() != b.abi() {
            return Err(TypeError::AbiMismatch(ExpectedFound::new(a.abi(), b.abi())));
        };

        if a.splatted() != b.splatted() {
            return Err(TypeError::SplatMismatch(ExpectedFound::new(a.splatted(), b.splatted())));
        }

        let a_inputs = a.inputs();
        let b_inputs = b.inputs();
        if a_inputs.len() != b_inputs.len() {
            return Err(TypeError::ArgCount);
        }

        let inputs_and_output = iter::zip(a_inputs.iter(), b_inputs.iter())
            .map(|(a, b)| ((a, b), false))
            .chain(iter::once(((a.output(), b.output()), true)))
            .map(|((a, b), is_output)| {
                if is_output {
                    relation.relate(a, b)
                } else {
                    relation.relate_with_variance(
                        ty::Contravariant,
                        VarianceDiagInfo::default(),
                        a,
                        b,
                    )
                }
            })
            .enumerate()
            .map(|(i, r)| match r {
                Err(TypeError::Sorts(exp_found) | TypeError::ArgumentSorts(exp_found, _)) => {
                    Err(TypeError::ArgumentSorts(exp_found, i))
                }
                Err(TypeError::Mutability | TypeError::ArgumentMutability(_)) => {
                    Err(TypeError::ArgumentMutability(i))
                }
                r => r,
            });
        Ok(ty::FnSig {
            inputs_and_output: cx.mk_type_list_from_iter(inputs_and_output)?,
            fn_sig_kind: a.fn_sig_kind,
        })
    }
}

enum EvidenceRelateError<I: Interner> {
    DifferentProof,
    Type(TypeError<I>),
}

impl<I: Interner> From<TypeError<I>> for EvidenceRelateError<I> {
    fn from(error: TypeError<I>) -> Self {
        EvidenceRelateError::Type(error)
    }
}

fn relate_evidence_source<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a: ty::solve::CandidateEvidenceSource<I>,
    b: ty::solve::CandidateEvidenceSource<I>,
    trait_ref: ty::TraitRef<I>,
) -> Result<ty::solve::CandidateEvidenceSource<I>, EvidenceRelateError<I>> {
    use ty::solve::CandidateEvidenceSource;

    Ok(match (a, b) {
        (CandidateEvidenceSource::Unique(_), CandidateEvidenceSource::Unique(_)) => {
            CandidateEvidenceSource::Unique(ty::solve::CoherenceKey { trait_ref })
        }
        (
            CandidateEvidenceSource::Impl { impl_def_id: a_impl, args: a_args },
            CandidateEvidenceSource::Impl { impl_def_id: b_impl, args: b_args },
        ) if a_impl == b_impl => CandidateEvidenceSource::Impl {
            impl_def_id: a_impl,
            args: relate_args_invariantly(relation, a_args, b_args)?,
        },
        (
            CandidateEvidenceSource::Builtin { source: a_source, evidence: a_evidence },
            CandidateEvidenceSource::Builtin { source: b_source, evidence: b_evidence },
        ) if a_source == b_source => {
            use ty::solve::BuiltinEvidence;

            let evidence = match (a_evidence, b_evidence) {
                (BuiltinEvidence::RuleOnly, BuiltinEvidence::RuleOnly) => BuiltinEvidence::RuleOnly,
                (
                    BuiltinEvidence::Fn { output: a_output, instantiation: a_instantiation },
                    BuiltinEvidence::Fn { output: b_output, instantiation: b_instantiation },
                ) => BuiltinEvidence::Fn {
                    instantiation: relate_args_invariantly(
                        relation,
                        a_instantiation,
                        b_instantiation,
                    )?,
                    output: relation.relate_with_variance(
                        ty::Invariant,
                        VarianceDiagInfo::default(),
                        a_output,
                        b_output,
                    )?,
                },
                (
                    BuiltinEvidence::AsyncFn { output: a_output, instantiation: a_instantiation },
                    BuiltinEvidence::AsyncFn { output: b_output, instantiation: b_instantiation },
                ) => BuiltinEvidence::AsyncFn {
                    instantiation: relate_args_invariantly(
                        relation,
                        a_instantiation,
                        b_instantiation,
                    )?,
                    output: relation.relate_with_variance(
                        ty::Invariant,
                        VarianceDiagInfo::default(),
                        a_output,
                        b_output,
                    )?,
                },
                _ => return Err(EvidenceRelateError::DifferentProof),
            };
            CandidateEvidenceSource::Builtin { source: a_source, evidence }
        }
        (
            CandidateEvidenceSource::Dyn {
                object_bound: a_bound,
                instantiation: a_instantiation,
                vtable_slot: a_slot,
                operation: a_operation,
            },
            CandidateEvidenceSource::Dyn {
                object_bound: b_bound,
                instantiation: b_instantiation,
                vtable_slot: b_slot,
                operation: b_operation,
            },
        ) if a_slot == b_slot => {
            let a_bound = a_bound.as_trait_clause().ok_or(EvidenceRelateError::DifferentProof)?;
            let b_bound = b_bound.as_trait_clause().ok_or(EvidenceRelateError::DifferentProof)?;
            let object_bound = relation
                .relate(a_bound, b_bound)?
                .map_bound(ty::ClauseKind::Trait)
                .upcast(relation.cx());
            let instantiation = match (a_instantiation, b_instantiation) {
                (Some(a), Some(b)) => Some(relate_args_invariantly(relation, a, b)?),
                (None, None) => None,
                // Normalization can erase all uses of an object bound's
                // lifetime parameters. Its universal proof and an exact
                // instantiation then name the same dictionary.
                (None, Some(_)) | (Some(_), None)
                    if a_bound.no_bound_vars().is_some() && b_bound.no_bound_vars().is_some() =>
                {
                    None
                }
                _ => return Err(EvidenceRelateError::DifferentProof),
            };
            let operation = match (a_operation, b_operation) {
                (Some(a), Some(b)) => {
                    let a_bound = a
                        .projection_bound
                        .as_projection_clause()
                        .ok_or(EvidenceRelateError::DifferentProof)?;
                    let b_bound = b
                        .projection_bound
                        .as_projection_clause()
                        .ok_or(EvidenceRelateError::DifferentProof)?;
                    let projection_bound = relation
                        .relate(a_bound, b_bound)?
                        .map_bound(ty::ClauseKind::Projection)
                        .upcast(relation.cx());
                    Some(ty::solve::DynProjectionOperation {
                        projection_bound,
                        ordinary_args: relate_args_invariantly(
                            relation,
                            a.ordinary_args,
                            b.ordinary_args,
                        )?,
                    })
                }
                (None, None) => None,
                _ => return Err(EvidenceRelateError::DifferentProof),
            };
            CandidateEvidenceSource::Dyn {
                object_bound,
                instantiation,
                vtable_slot: a_slot,
                operation,
            }
        }
        (
            CandidateEvidenceSource::ParamEnv { source: a_source, origin: a_origin },
            CandidateEvidenceSource::ParamEnv { source: b_source, origin: b_origin },
        ) if a_source == b_source => {
            let origin = match (a_origin, b_origin) {
                (
                    ty::solve::ParamEnvAssumption::ItemContract { contract: a_contract },
                    ty::solve::ParamEnvAssumption::ItemContract { contract: b_contract },
                ) if a_contract.key == b_contract.key => {
                    ty::solve::ParamEnvAssumption::ItemContract {
                        contract: ty::solve::InstantiatedItemContract {
                            key: a_contract.key,
                            complete_early_args: relate_args_invariantly(
                                relation,
                                a_contract.complete_early_args,
                                b_contract.complete_early_args,
                            )?,
                        },
                    }
                }
                (
                    ty::solve::ParamEnvAssumption::Binder {
                        telescope_index: a_index,
                        identity: a_identity,
                        instantiation: a_instantiation,
                    },
                    ty::solve::ParamEnvAssumption::Binder {
                        telescope_index: b_index,
                        identity: b_identity,
                        instantiation: b_instantiation,
                    },
                ) if a_index == b_index && a_identity == b_identity => {
                    ty::solve::ParamEnvAssumption::Binder {
                        telescope_index: a_index,
                        identity: a_identity,
                        instantiation: relate_args_invariantly(
                            relation,
                            a_instantiation,
                            b_instantiation,
                        )?,
                    }
                }
                (
                    a_origin @ ty::solve::ParamEnvAssumption::ItemClause { .. },
                    ty::solve::ParamEnvAssumption::ItemClause { .. },
                ) => {
                    // ItemClause is used only for a source clause without an
                    // associated-equality contract. Different items may repeat
                    // the same principal bound while referring to the same
                    // coherent impl, so the origin itself is not observable by
                    // an associated projection. ItemContract remains strict.
                    a_origin
                }
                (a_origin, b_origin) if a_origin == b_origin => a_origin,
                _ => return Err(EvidenceRelateError::DifferentProof),
            };
            CandidateEvidenceSource::ParamEnv { source: a_source, origin }
        }
        (CandidateEvidenceSource::AliasBound(a), CandidateEvidenceSource::AliasBound(b))
            if a == b =>
        {
            CandidateEvidenceSource::AliasBound(a)
        }
        (
            CandidateEvidenceSource::Recursive { cycle: a_cycle },
            CandidateEvidenceSource::Recursive { cycle: b_cycle },
        ) if a_cycle == b_cycle => CandidateEvidenceSource::Recursive { cycle: a_cycle },
        (CandidateEvidenceSource::Error, CandidateEvidenceSource::Error) => {
            CandidateEvidenceSource::Error
        }
        (
            CandidateEvidenceSource::CoherenceUnknowable,
            CandidateEvidenceSource::CoherenceUnknowable,
        ) => CandidateEvidenceSource::CoherenceUnknowable,
        _ => {
            trace!(?a, ?b, ?trait_ref, "evidence source mismatch");
            return Err(EvidenceRelateError::DifferentProof);
        }
    })
}

fn relate_trait_evidence<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a: I::TraitEvidence,
    b: I::TraitEvidence,
) -> Result<I::TraitEvidence, EvidenceRelateError<I>> {
    if a == b {
        return Ok(a);
    }

    use ty::solve::TraitEvidenceKind;

    let kind = match (&a.kind, &b.kind) {
        (TraitEvidenceKind::Selected(a), TraitEvidenceKind::Selected(b)) => {
            let evidence = relate_candidate_evidence(relation, a, b)?;
            return Ok(relation.cx().mk_trait_evidence(evidence));
        }
        (TraitEvidenceKind::Infer(a_vid), TraitEvidenceKind::Infer(b_vid)) if a_vid == b_vid => {
            TraitEvidenceKind::Infer(*a_vid)
        }
        (
            TraitEvidenceKind::Bound(a_index, a_bound),
            TraitEvidenceKind::Bound(b_index, b_bound),
        ) if a_index == b_index && a_bound == b_bound => {
            TraitEvidenceKind::Bound(*a_index, *a_bound)
        }
        (
            TraitEvidenceKind::Placeholder(a_placeholder),
            TraitEvidenceKind::Placeholder(b_placeholder),
        ) if a_placeholder == b_placeholder => TraitEvidenceKind::Placeholder(*a_placeholder),
        // Error evidence is only a recovery marker. Two independently
        // constructed error values must not establish semantic equality.
        (TraitEvidenceKind::Error(_), _)
        | (TraitEvidenceKind::Selected(_), _)
        | (TraitEvidenceKind::Infer(_), _)
        | (TraitEvidenceKind::Bound(..), _)
        | (TraitEvidenceKind::Placeholder(_), _) => {
            trace!(?a, ?b, "trait evidence kind mismatch");
            return Err(EvidenceRelateError::DifferentProof);
        }
    };

    let trait_ref = relation.relate(a.trait_ref, b.trait_ref)?;
    Ok(relation.cx().mk_trait_evidence_kind(trait_ref, kind))
}

/// Invariantly relates two fully-instantiated evidence values.
///
/// Evidence is kept outside [`GenericArg`](ty::GenericArgKind), so it cannot
/// use the blanket [`Relate`] entry point. Canonical response replay still
/// needs the same structural relation for selected proof recipes, however.
/// Keep that operation here next to the proof-DAG relation and expose only a
/// normal [`TypeError`] to callers outside this module.
pub fn relate_trait_evidence_invariantly<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a: I::TraitEvidence,
    b: I::TraitEvidence,
) -> RelateResult<I, I::TraitEvidence> {
    match relate_trait_evidence(relation, a, b) {
        Ok(evidence) => Ok(evidence),
        Err(EvidenceRelateError::Type(error)) => Err(error),
        Err(EvidenceRelateError::DifferentProof) => Err(TypeError::Mismatch),
    }
}

fn canonical_evidence_response_is_prefix<I: Interner>(
    short_values: I::GenericArgs,
    short_evidence_values: I::TraitEvidences,
    short: &ty::solve::CanonicalEvidence<I>,
    long_values: I::GenericArgs,
    long_evidence_values: I::TraitEvidences,
    long: &ty::solve::CanonicalEvidence<I>,
) -> bool {
    short.var_kinds.len() == short_values.len()
        && short.evidence_var_kinds.len() == short_evidence_values.len()
        && long.var_kinds.len() == long_values.len()
        && long.evidence_var_kinds.len() == long_evidence_values.len()
        && short.value.var_values.is_identity()
        && short.value.evidence_var_values.is_identity()
        && long.value.var_values.is_identity()
        && long.value.evidence_var_values.is_identity()
        && short.max_universe == long.max_universe
        && short.value.evidence == long.value.evidence
        && short.var_kinds.iter().eq(long.var_kinds.iter().take(short.var_kinds.len()))
        && short
            .evidence_var_kinds
            .iter()
            .eq(long.evidence_var_kinds.iter().take(short.evidence_var_kinds.len()))
}

fn relate_canonical_evidence_prefix<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    short_values: I::GenericArgs,
    short_evidence_values: I::TraitEvidences,
    short: ty::solve::CanonicalEvidence<I>,
    long_values: I::GenericArgs,
    long_evidence_values: I::TraitEvidences,
    long: ty::solve::CanonicalEvidence<I>,
) -> Result<Option<ty::solve::CandidateEvidenceUse<I>>, EvidenceRelateError<I>> {
    if !canonical_evidence_response_is_prefix(
        short_values,
        short_evidence_values,
        &short,
        long_values,
        long_evidence_values,
        &long,
    ) {
        return Ok(None);
    }

    let long_values = relation.cx().mk_args_from_iter(long_values.iter().take(short_values.len()));
    let original_values = relate_args_invariantly(relation, short_values, long_values)?;

    let long_evidence_values = relation.cx().mk_trait_evidences_from_iter(
        long_evidence_values.iter().take(short_evidence_values.len()),
    );
    let original_evidence_values = short_evidence_values
        .iter()
        .zip(long_evidence_values.iter())
        .map(|(short, long)| relation.evidences(short, long))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Some(ty::solve::CandidateEvidenceUse::Canonical {
        original_values,
        original_evidence_values: relation.cx().mk_trait_evidences(&original_evidence_values),
        response: short,
    }))
}

fn relate_candidate_evidence<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a: &ty::solve::CandidateEvidence<I>,
    b: &ty::solve::CandidateEvidence<I>,
) -> Result<ty::solve::CandidateEvidence<I>, EvidenceRelateError<I>> {
    let a_unique = matches!(a.root_node().source, ty::solve::CandidateEvidenceSource::Unique(_));
    let b_unique = matches!(b.root_node().source, ty::solve::CandidateEvidenceSource::Unique(_));

    // `Unique` is a coherence quotient, not an additional proof step. Its
    // selected recipe is retained so normalization can replay one concrete
    // candidate, but the quotient and that exact member denote the same
    // dictionary. This matters when one elaboration path observes duplicate
    // ParamEnv candidates and creates the quotient while another path sees
    // only the stable selected assumption.
    //
    // Keep this deliberately narrower than making `Unique` transparent: a
    // non-quotiented recipe is related to a quotient only when it structurally
    // matches the recipe actually nested by that quotient. Unrelated members
    // of the coherence class still need their own `Unique` witness.
    match (a_unique, b_unique) {
        (true, false) => {
            let selected = candidate_evidence_subgraph(a, a.root_node().nested[0]);
            let related = relate_candidate_evidence(relation, &selected, b)?;
            let key = ty::solve::CoherenceKey { trait_ref: related.root_node().trait_ref };
            return Ok(related.into_unique(key));
        }
        (false, true) => {
            let selected = candidate_evidence_subgraph(b, b.root_node().nested[0]);
            let related = relate_candidate_evidence(relation, a, &selected)?;
            let key = ty::solve::CoherenceKey { trait_ref: related.root_node().trait_ref };
            return Ok(related.into_unique(key));
        }
        (true, true) => {
            // Candidate choice has already been proven unobservable on both
            // sides. Equality is therefore indexed solely by the coherence
            // key; the concrete recipes may legitimately differ. Preserve a
            // replayable recipe from the left and retarget the repeated key
            // occurrences to the related trait ref.
            let trait_ref = relation.relate(a.root_node().trait_ref, b.root_node().trait_ref)?;
            let mut evidence = a.clone();
            retarget_unique_evidence_root(&mut evidence, trait_ref);
            return Ok(evidence);
        }
        (false, false) => {}
    }

    if a.root != b.root || a.nodes.len() != b.nodes.len() {
        trace!(?a, ?b, "evidence graph shape mismatch");
        return Err(EvidenceRelateError::DifferentProof);
    }
    if a.cycle_keys.len() != b.cycle_keys.len()
        || !a.cycle_keys.iter().zip(&b.cycle_keys).all(|(a, b)| a.is_bisimilar_to(b))
    {
        trace!(?a, ?b, "recursive evidence cycle mismatch");
        return Err(EvidenceRelateError::DifferentProof);
    }

    let mut nodes = Vec::with_capacity(a.nodes.len());
    for (a_node, b_node) in a.nodes.iter().zip(&b.nodes) {
        if a_node.nested != b_node.nested
            || a_node.nested_evidence.len() != b_node.nested_evidence.len()
        {
            trace!(?a_node, ?b_node, "evidence node edge mismatch");
            return Err(EvidenceRelateError::DifferentProof);
        }

        let trait_ref = relation.relate(a_node.trait_ref, b_node.trait_ref)?;
        let source = relate_evidence_source(relation, a_node.source, b_node.source, trait_ref)?;
        let mut nested_evidence = Vec::with_capacity(a_node.nested_evidence.len());
        for (&a_nested, &b_nested) in a_node.nested_evidence.iter().zip(&b_node.nested_evidence) {
            use ty::solve::CandidateEvidenceUse;
            nested_evidence.push(match (a_nested, b_nested) {
                (
                    CandidateEvidenceUse::Instantiated(a_evidence),
                    CandidateEvidenceUse::Instantiated(b_evidence),
                ) => {
                    CandidateEvidenceUse::Instantiated(relation.evidences(a_evidence, b_evidence)?)
                }
                (
                    CandidateEvidenceUse::Canonical {
                        original_values: a_values,
                        original_evidence_values: a_evidence_values,
                        response: a_response,
                    },
                    CandidateEvidenceUse::Canonical {
                        original_values: b_values,
                        original_evidence_values: b_evidence_values,
                        response: b_response,
                    },
                ) => {
                    if let Some(evidence) = relate_canonical_evidence_prefix(
                        relation,
                        a_values,
                        a_evidence_values,
                        a_response,
                        b_values,
                        b_evidence_values,
                        b_response,
                    )? {
                        evidence
                    } else if let Some(evidence) = relate_canonical_evidence_prefix(
                        relation,
                        b_values,
                        b_evidence_values,
                        b_response,
                        a_values,
                        a_evidence_values,
                        a_response,
                    )? {
                        evidence
                    } else {
                        return Err(EvidenceRelateError::DifferentProof);
                    }
                }
                _ => {
                    trace!(?a_nested, ?b_nested, "nested evidence mismatch");
                    return Err(EvidenceRelateError::DifferentProof);
                }
            });
        }

        nodes.push(ty::solve::CandidateEvidenceNode {
            trait_ref,
            source,
            nested: a_node.nested.clone(),
            nested_evidence,
        });
    }

    let mut cycle_keys = a.cycle_keys.clone();
    for node in &nodes {
        if let ty::solve::CandidateEvidenceSource::Recursive { cycle } = node.source {
            cycle_keys[usize::try_from(cycle).expect("proof cycle index overflow")]
                .head_trait_ref = node.trait_ref;
        }
    }
    let evidence = ty::solve::CandidateEvidence { root: a.root, nodes, cycle_keys };
    if matches!(evidence.selected_source(), ty::solve::CandidateEvidenceSource::Recursive { .. }) {
        evidence.assert_serialized_well_formed();
    } else {
        evidence.assert_well_formed();
    }
    Ok(evidence)
}

/// Copies the proof sub-DAG reachable from `root` and gives it dense local
/// node indices. A `Unique` wrapper is serialized in the same DAG as its
/// selected recipe, so merely changing `root` would leave an unreachable node
/// and violate the proof boundary invariant.
fn candidate_evidence_subgraph<I: Interner>(
    evidence: &ty::solve::CandidateEvidence<I>,
    root: u32,
) -> ty::solve::CandidateEvidence<I> {
    let root = usize::try_from(root).expect("proof node index overflow");
    let mut reachable = vec![false; evidence.nodes.len()];
    let mut stack = vec![root];
    while let Some(index) = stack.pop() {
        if reachable[index] {
            continue;
        }
        reachable[index] = true;
        stack.extend(
            evidence.nodes[index]
                .nested
                .iter()
                .map(|&nested| usize::try_from(nested).expect("proof node index overflow")),
        );
    }

    let mut remap = vec![None; evidence.nodes.len()];
    let mut nodes = Vec::with_capacity(reachable.iter().filter(|&&reachable| reachable).count());
    for (old_index, node) in evidence.nodes.iter().enumerate() {
        if reachable[old_index] {
            remap[old_index] = Some(u32::try_from(nodes.len()).expect("too many proof nodes"));
            nodes.push(node.clone());
        }
    }
    for node in &mut nodes {
        for nested in &mut node.nested {
            *nested = remap[usize::try_from(*nested).expect("proof node index overflow")]
                .expect("reachable proof node references an unreachable child");
        }
    }

    let evidence = ty::solve::CandidateEvidence {
        root: remap[root].expect("proof subgraph root is unreachable"),
        nodes,
        cycle_keys: evidence.cycle_keys.clone(),
    };
    evidence.assert_well_formed();
    evidence
}

/// A coherence wrapper, its key, and its selected node repeat one semantic
/// trait-ref identity. Keep all repetitions synchronized when relating two
/// quotients whose concrete replay recipes differ.
fn retarget_unique_evidence_root<I: Interner>(
    evidence: &mut ty::solve::CandidateEvidence<I>,
    trait_ref: ty::TraitRef<I>,
) {
    let key = ty::solve::CoherenceKey { trait_ref };
    let mut index = evidence.root;
    loop {
        let node = &mut evidence.nodes[usize::try_from(index).expect("proof node index overflow")];
        node.trait_ref = trait_ref;
        match node.source {
            ty::solve::CandidateEvidenceSource::Unique(_) => {
                node.source = ty::solve::CandidateEvidenceSource::Unique(key);
                index = node.nested[0];
            }
            _ => break,
        }
    }
    evidence.assert_well_formed();
}

fn relate_evidence_projection<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a: I::EvidenceProjection,
    b: I::EvidenceProjection,
) -> Result<I::EvidenceProjection, EvidenceRelateError<I>> {
    if a.item_def_id != b.item_def_id {
        return Err(EvidenceRelateError::DifferentProof);
    }
    let evidence = relation.evidences(a.evidence, b.evidence).map_err(EvidenceRelateError::Type)?;
    Ok(relation.cx().mk_evidence_projection(ty::EvidenceProjectionData {
        item_def_id: a.item_def_id,
        evidence,
    }))
}

/// Upgrades a surface projection to the evidence-indexed form already present
/// on the other side of a relation. This only checks and reuses carried
/// evidence; it never selects a trait candidate.
fn relate_evidence_projection_to_surface<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    projection: I::EvidenceProjection,
    evidence_own_args: I::GenericArgs,
    surface_item_def_id: I::TraitAssocTermId,
    surface_trait_ref: ty::TraitRef<I>,
    surface_own_args: I::GenericArgs,
    evidence_on_left: bool,
) -> Result<(I::EvidenceProjection, I::GenericArgs), EvidenceRelateError<I>> {
    if projection.item_def_id != surface_item_def_id {
        return Err(EvidenceRelateError::DifferentProof);
    }

    let own_args = if evidence_on_left {
        relation.relate(projection.trait_ref(), surface_trait_ref)?;
        relate_args_invariantly(relation, evidence_own_args, surface_own_args)?
    } else {
        relation.relate(surface_trait_ref, projection.trait_ref())?;
        relate_args_invariantly(relation, surface_own_args, evidence_own_args)?
    };
    Ok((projection, own_args))
}

impl<I: Interner> Relate<I> for ty::AliasTy<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::AliasTy<I>,
        b: ty::AliasTy<I>,
    ) -> RelateResult<I, ty::AliasTy<I>> {
        let kind = if a.kind == b.kind {
            a.kind
        } else if let (
            ty::AliasTyKind::EvidenceProjection { projection: a_projection },
            ty::AliasTyKind::EvidenceProjection { projection: b_projection },
        ) = (a.kind, b.kind)
        {
            match relate_evidence_projection(relation, a_projection, b_projection) {
                Ok(projection) => ty::AliasTyKind::EvidenceProjection { projection },
                Err(EvidenceRelateError::Type(error)) => return Err(error),
                Err(EvidenceRelateError::DifferentProof) => {
                    return Err(TypeError::ProjectionMismatched(ExpectedFound::new(
                        a.kind.into(),
                        b.kind.into(),
                    )));
                }
            }
        } else {
            let cx = relation.cx();
            let surface_and_evidence = match (a.kind, b.kind) {
                (
                    ty::AliasTyKind::EvidenceProjection { projection },
                    ty::AliasTyKind::Projection { def_id: _ },
                ) => Some((projection, a.args, b, true)),
                (
                    ty::AliasTyKind::Projection { def_id: _ },
                    ty::AliasTyKind::EvidenceProjection { projection },
                ) => Some((projection, b.args, a, false)),
                _ => None,
            };
            let Some((projection, evidence_own_args, surface, evidence_on_left)) =
                surface_and_evidence
            else {
                return Err(TypeError::ProjectionMismatched(ExpectedFound::new(
                    a.kind.into(),
                    b.kind.into(),
                )));
            };
            let ty::AliasTyKind::Projection { def_id } = surface.kind else { unreachable!() };
            let (surface_trait_ref, surface_own_args) = surface.trait_ref_and_own_args(cx);
            let surface_own_args = cx.mk_args_from_iter(surface_own_args.iter());
            return match relate_evidence_projection_to_surface(
                relation,
                projection,
                evidence_own_args,
                def_id.into(),
                surface_trait_ref,
                surface_own_args,
                evidence_on_left,
            ) {
                Ok((projection, args)) => Ok(ty::AliasTy::new_from_args(
                    cx,
                    ty::AliasTyKind::EvidenceProjection { projection },
                    args,
                )),
                Err(EvidenceRelateError::Type(error)) => Err(error),
                Err(EvidenceRelateError::DifferentProof) => Err(TypeError::ProjectionMismatched(
                    ExpectedFound::new(a.kind.into(), b.kind.into()),
                )),
            };
        };

        let cx = relation.cx();
        let args = if let Some(variances) = cx.opt_alias_variances(kind) {
            relate_args_with_variances(relation, variances, a.args, b.args)?
        } else {
            relate_args_invariantly(relation, a.args, b.args)?
        };
        Ok(ty::AliasTy::new_from_args(relation.cx(), kind, args))
    }
}

impl<I: Interner> Relate<I> for ty::AliasConst<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::AliasConst<I>,
        b: ty::AliasConst<I>,
    ) -> RelateResult<I, ty::AliasConst<I>> {
        let cx = relation.cx();
        if a.kind != b.kind {
            Err(TypeError::ConstMismatch(ExpectedFound::new(
                Const::new_alias(cx, ty::IsRigid::yes_if_next_solver(cx), a),
                Const::new_alias(cx, ty::IsRigid::yes_if_next_solver(cx), b),
            )))
        } else {
            // FIXME(mgca): remove this
            debug_assert_eq!(a.type_of(cx).skip_norm_wip(), b.type_of(cx).skip_norm_wip());

            let args = relate_args_invariantly(relation, a.args, b.args)?;

            Ok(ty::AliasConst::new(cx, a.kind, args))
        }
    }
}

impl<I: Interner> Relate<I> for ty::AliasTerm<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::AliasTerm<I>,
        b: ty::AliasTerm<I>,
    ) -> RelateResult<I, ty::AliasTerm<I>> {
        let kind = if a.kind == b.kind {
            a.kind
        } else {
            let projections = match (a.kind, b.kind) {
                (
                    ty::AliasTermKind::EvidenceProjectionTy { projection: a_projection },
                    ty::AliasTermKind::EvidenceProjectionTy { projection: b_projection },
                ) => Some((a_projection, b_projection, true)),
                (
                    ty::AliasTermKind::EvidenceProjectionConst { projection: a_projection },
                    ty::AliasTermKind::EvidenceProjectionConst { projection: b_projection },
                ) => Some((a_projection, b_projection, false)),
                _ => None,
            };
            if let Some((a_projection, b_projection, is_ty)) = projections {
                match relate_evidence_projection(relation, a_projection, b_projection) {
                    Ok(projection) if is_ty => {
                        ty::AliasTermKind::EvidenceProjectionTy { projection }
                    }
                    Ok(projection) => ty::AliasTermKind::EvidenceProjectionConst { projection },
                    Err(EvidenceRelateError::Type(error)) => return Err(error),
                    Err(EvidenceRelateError::DifferentProof) => {
                        return Err(TypeError::ProjectionMismatched(ExpectedFound::new(
                            a.kind, b.kind,
                        )));
                    }
                }
            } else {
                let cx = relation.cx();
                let surface_and_evidence = match (a.kind, b.kind) {
                    (
                        ty::AliasTermKind::EvidenceProjectionTy { projection },
                        ty::AliasTermKind::ProjectionTy { .. },
                    )
                    | (
                        ty::AliasTermKind::EvidenceProjectionConst { projection },
                        ty::AliasTermKind::ProjectionConst { .. },
                    ) => Some((projection, a.args, b, true)),
                    (
                        ty::AliasTermKind::ProjectionTy { .. },
                        ty::AliasTermKind::EvidenceProjectionTy { projection },
                    )
                    | (
                        ty::AliasTermKind::ProjectionConst { .. },
                        ty::AliasTermKind::EvidenceProjectionConst { projection },
                    ) => Some((projection, b.args, a, false)),
                    _ => None,
                };
                let Some((projection, evidence_own_args, surface, evidence_on_left)) =
                    surface_and_evidence
                else {
                    return Err(TypeError::ProjectionMismatched(ExpectedFound::new(
                        a.kind, b.kind,
                    )));
                };
                let (surface_trait_ref, surface_own_args) = surface.trait_ref_and_own_args(cx);
                let surface_own_args = cx.mk_args_from_iter(surface_own_args.iter());
                return match relate_evidence_projection_to_surface(
                    relation,
                    projection,
                    evidence_own_args,
                    surface.expect_projection_def_id(),
                    surface_trait_ref,
                    surface_own_args,
                    evidence_on_left,
                ) {
                    Ok((projection, args)) => {
                        let kind = match surface.kind {
                            ty::AliasTermKind::ProjectionTy { .. } => {
                                ty::AliasTermKind::EvidenceProjectionTy { projection }
                            }
                            ty::AliasTermKind::ProjectionConst { .. } => {
                                ty::AliasTermKind::EvidenceProjectionConst { projection }
                            }
                            _ => unreachable!(),
                        };
                        Ok(ty::AliasTerm::new_from_args(cx, kind, args))
                    }
                    Err(EvidenceRelateError::Type(error)) => Err(error),
                    Err(EvidenceRelateError::DifferentProof) => {
                        Err(TypeError::ProjectionMismatched(ExpectedFound::new(a.kind, b.kind)))
                    }
                };
            }
        };

        let args = match kind {
            ty::AliasTermKind::OpaqueTy { def_id } => relate_args_with_variances(
                relation,
                relation.cx().variances_of(def_id.into()),
                a.args,
                b.args,
            )?,
            ty::AliasTermKind::ProjectionTy { .. }
            | ty::AliasTermKind::EvidenceProjectionTy { .. }
            | ty::AliasTermKind::EvidenceProjectionConst { .. }
            | ty::AliasTermKind::FreeConst { .. }
            | ty::AliasTermKind::FreeTy { .. }
            | ty::AliasTermKind::InherentTy { .. }
            | ty::AliasTermKind::InherentConstSelf { .. }
            | ty::AliasTermKind::InherentConstImpl { .. }
            | ty::AliasTermKind::AnonConst { .. }
            | ty::AliasTermKind::ProjectionConst { .. } => {
                relate_args_invariantly(relation, a.args, b.args)?
            }
        };
        Ok(ty::AliasTerm::new_from_args(relation.cx(), kind, args))
    }
}

impl<I: Interner> Relate<I> for ty::ExistentialProjection<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::ExistentialProjection<I>,
        b: ty::ExistentialProjection<I>,
    ) -> RelateResult<I, ty::ExistentialProjection<I>> {
        if a.def_id != b.def_id {
            Err(TypeError::ProjectionMismatched(ExpectedFound::new(
                relation.cx().alias_term_kind_from_def_id(
                    a.def_id.into(),
                    ty::AliasConstInherentArgsKind::WithSelf,
                ),
                relation.cx().alias_term_kind_from_def_id(
                    b.def_id.into(),
                    ty::AliasConstInherentArgsKind::WithSelf,
                ),
            )))
        } else {
            let term = relation.relate_with_variance(
                ty::Invariant,
                VarianceDiagInfo::default(),
                a.term,
                b.term,
            )?;
            let args = relation.relate_with_variance(
                ty::Invariant,
                VarianceDiagInfo::default(),
                a.args,
                b.args,
            )?;
            Ok(ty::ExistentialProjection::new_from_args(relation.cx(), a.def_id, args, term))
        }
    }
}

impl<I: Interner> Relate<I> for ty::TraitRef<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::TraitRef<I>,
        b: ty::TraitRef<I>,
    ) -> RelateResult<I, ty::TraitRef<I>> {
        // Different traits cannot be related.
        if a.def_id != b.def_id {
            Err(TypeError::Traits({
                let a = a.def_id;
                let b = b.def_id;
                ExpectedFound::new(a, b)
            }))
        } else {
            let args = relate_args_invariantly(relation, a.args, b.args)?;
            Ok(ty::TraitRef::new_from_args(relation.cx(), a.def_id, args))
        }
    }
}

impl<I: Interner> Relate<I> for ty::ExistentialTraitRef<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::ExistentialTraitRef<I>,
        b: ty::ExistentialTraitRef<I>,
    ) -> RelateResult<I, ty::ExistentialTraitRef<I>> {
        // Different traits cannot be related.
        if a.def_id != b.def_id {
            Err(TypeError::Traits({
                let a = a.def_id;
                let b = b.def_id;
                ExpectedFound::new(a, b)
            }))
        } else {
            let args = relate_args_invariantly(relation, a.args, b.args)?;
            Ok(ty::ExistentialTraitRef::new_from_args(relation.cx(), a.def_id, args))
        }
    }
}

/// Relates `a` and `b` structurally, calling the relation for all nested values.
/// Any semantic equality, e.g. of projections, and inference variables have to be
/// handled by the caller.
#[instrument(level = "trace", skip(relation), ret)]
pub fn structurally_relate_tys<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a: I::Ty,
    b: I::Ty,
) -> RelateResult<I, I::Ty> {
    let cx = relation.cx();
    match (a.kind(), b.kind()) {
        (ty::Infer(_), _) | (_, ty::Infer(_)) => {
            // The caller should handle these cases!
            panic!("var types encountered in structurally_relate_tys")
        }

        (ty::Bound(..), _) | (_, ty::Bound(..)) => {
            panic!("bound types encountered in structurally_relate_tys")
        }

        (ty::Error(guar), _) | (_, ty::Error(guar)) => Ok(Ty::new_error(cx, guar)),

        (ty::Never, _)
        | (ty::Char, _)
        | (ty::Bool, _)
        | (ty::Int(_), _)
        | (ty::Uint(_), _)
        | (ty::Float(_), _)
        | (ty::Str, _)
            if a == b =>
        {
            Ok(a)
        }

        (ty::Param(a_p), ty::Param(b_p)) if a_p.index() == b_p.index() => {
            // FIXME: Put this back
            //debug_assert_eq!(a_p.name(), b_p.name(), "param types with same index differ in name");
            Ok(a)
        }

        (ty::Placeholder(p1), ty::Placeholder(p2)) if p1 == p2 => Ok(a),

        (ty::Adt(a_def, a_args), ty::Adt(b_def, b_args)) if a_def == b_def => {
            if a_args.is_empty() {
                Ok(a)
            } else {
                relation.relate_ty_args(a, b, a_def.def_id().into(), a_args, b_args, |args| {
                    Ty::new_adt(cx, a_def, args)
                })
            }
        }

        (ty::Foreign(a_id), ty::Foreign(b_id)) if a_id == b_id => Ok(Ty::new_foreign(cx, a_id)),

        (ty::Dynamic(a_obj, a_region), ty::Dynamic(b_obj, b_region)) => Ok(Ty::new_dynamic(
            cx,
            relation.relate(a_obj, b_obj)?,
            relation.relate(a_region, b_region)?,
        )),

        (ty::Coroutine(a_id, a_args), ty::Coroutine(b_id, b_args)) if a_id == b_id => {
            // All Coroutine types with the same id represent
            // the (anonymous) type of the same coroutine expression. So
            // all of their regions should be equated.
            let args = relate_args_invariantly(relation, a_args, b_args)?;
            Ok(Ty::new_coroutine(cx, a_id, args))
        }

        (ty::CoroutineWitness(a_id, a_args), ty::CoroutineWitness(b_id, b_args))
            if a_id == b_id =>
        {
            // All CoroutineWitness types with the same id represent
            // the (anonymous) type of the same coroutine expression. So
            // all of their regions should be equated.
            let args = relate_args_invariantly(relation, a_args, b_args)?;
            Ok(Ty::new_coroutine_witness(cx, a_id, args))
        }

        (ty::Closure(a_id, a_args), ty::Closure(b_id, b_args)) if a_id == b_id => {
            // All Closure types with the same id represent
            // the (anonymous) type of the same closure expression. So
            // all of their regions should be equated.
            let args = relate_args_invariantly(relation, a_args, b_args)?;
            Ok(Ty::new_closure(cx, a_id, args))
        }

        (ty::CoroutineClosure(a_id, a_args), ty::CoroutineClosure(b_id, b_args))
            if a_id == b_id =>
        {
            let args = relate_args_invariantly(relation, a_args, b_args)?;
            Ok(Ty::new_coroutine_closure(cx, a_id, args))
        }

        (ty::RawPtr(a_ty, a_mutbl), ty::RawPtr(b_ty, b_mutbl)) => {
            if a_mutbl != b_mutbl {
                return Err(TypeError::Mutability);
            }

            let (variance, info) = match a_mutbl {
                Mutability::Not => (ty::Covariant, VarianceDiagInfo::None),
                Mutability::Mut => {
                    (ty::Invariant, VarianceDiagInfo::Invariant { ty: a, param_index: 0 })
                }
            };

            let ty = relation.relate_with_variance(variance, info, a_ty, b_ty)?;

            Ok(Ty::new_ptr(cx, ty, a_mutbl))
        }

        (ty::Ref(a_r, a_ty, a_mutbl), ty::Ref(b_r, b_ty, b_mutbl)) => {
            if a_mutbl != b_mutbl {
                return Err(TypeError::Mutability);
            }

            let (variance, info) = match a_mutbl {
                Mutability::Not => (ty::Covariant, VarianceDiagInfo::None),
                Mutability::Mut => {
                    (ty::Invariant, VarianceDiagInfo::Invariant { ty: a, param_index: 0 })
                }
            };

            let r = relation.relate(a_r, b_r)?;
            let ty = relation.relate_with_variance(variance, info, a_ty, b_ty)?;

            Ok(Ty::new_ref(cx, r, ty, a_mutbl))
        }

        (ty::Array(a_t, sz_a), ty::Array(b_t, sz_b)) => {
            let t = relation.relate(a_t, b_t)?;
            match relation.relate(sz_a, sz_b) {
                Ok(sz) => Ok(Ty::new_array_with_const_len(cx, t, sz)),
                Err(TypeError::ConstMismatch(_)) => {
                    Err(TypeError::ArraySize(ExpectedFound::new(sz_a, sz_b)))
                }
                Err(e) => Err(e),
            }
        }

        (ty::Slice(a_t), ty::Slice(b_t)) => {
            let t = relation.relate(a_t, b_t)?;
            Ok(Ty::new_slice(cx, t))
        }

        (ty::Tuple(as_), ty::Tuple(bs)) => {
            if as_.len() == bs.len() {
                Ok(Ty::new_tup_from_iter(
                    cx,
                    iter::zip(as_.iter(), bs.iter()).map(|(a, b)| relation.relate(a, b)),
                )?)
            } else if !(as_.is_empty() || bs.is_empty()) {
                Err(TypeError::TupleSize(ExpectedFound::new(as_.len(), bs.len())))
            } else {
                Err(TypeError::Sorts(ExpectedFound::new(a, b)))
            }
        }

        (ty::FnDef(a_def_id, a_args), ty::FnDef(b_def_id, b_args)) if a_def_id == b_def_id => {
            // Function-item arguments are ordinary early-bound arguments. The
            // binder carried by `FnDef` may additionally contain compiler
            // evidence declarations, but those declarations do not form part
            // of the callable type's relation. Relate the ordinary payload
            // directly while preserving the left-hand telescope; trying to
            // instantiate the synthetic evidence binder here introduces fresh
            // proof variables and makes otherwise identical function items
            // fail equality when a contract is replayed.
            let args = relation.relate_ty_args(
                a,
                b,
                a_def_id.into(),
                a_args.fn_def_args(),
                b_args.fn_def_args(),
                |args| Ty::new_fn_def(cx, a_def_id, a_args.rebind(args)),
            )?;
            Ok(args)
        }

        (ty::FnPtr(a_sig_tys, a_hdr), ty::FnPtr(b_sig_tys, b_hdr)) => {
            let fty = relation.relate(a_sig_tys.with(a_hdr), b_sig_tys.with(b_hdr))?;
            Ok(Ty::new_fn_ptr(cx, fty))
        }

        // Alias tend to mostly already be handled downstream due to normalization.
        (ty::Alias(is_rigid_a, alias_a), ty::Alias(is_rigid_b, alias_b)) => {
            // Users shouldn't know about this so the mismatch should be caught
            // during development rather than presented as type error.
            debug_assert_eq!(is_rigid_a, is_rigid_b, "{a:?} != {b:?}");
            let alias_ty = relation.relate(alias_a, alias_b)?;
            Ok(Ty::new_alias(cx, is_rigid_a, alias_ty))
        }

        (ty::Pat(a_ty, a_pat), ty::Pat(b_ty, b_pat)) => {
            let ty = relation.relate(a_ty, b_ty)?;
            let pat = relation.relate(a_pat, b_pat)?;
            Ok(Ty::new_pat(cx, ty, pat))
        }

        (ty::UnsafeBinder(a_binder), ty::UnsafeBinder(b_binder)) => {
            Ok(Ty::new_unsafe_binder(cx, relation.binders(*a_binder, *b_binder)?))
        }

        _ => Err(TypeError::Sorts(ExpectedFound::new(a, b))),
    }
}

/// Relates `a` and `b` structurally, calling the relation for all nested values.
/// Any semantic equality, e.g. of alias consts, and inference variables have
/// to be handled by the caller.
///
/// FIXME: This is not totally structural, which probably should be fixed.
/// See the HACKs below.
pub fn structurally_relate_consts<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    mut a: I::Const,
    mut b: I::Const,
) -> RelateResult<I, I::Const> {
    trace!(
        "structurally_relate_consts::<{}>(a = {:?}, b = {:?})",
        std::any::type_name::<R>(),
        a,
        b
    );
    let cx = relation.cx();

    if cx.features().generic_const_exprs() {
        a = cx.expand_abstract_consts(a);
        b = cx.expand_abstract_consts(b);
    }

    trace!(
        "structurally_relate_consts::<{}>(normed_a = {:?}, normed_b = {:?})",
        std::any::type_name::<R>(),
        a,
        b
    );

    // Currently, the values that can be unified are primitive types,
    // and those that derive both `PartialEq` and `Eq`, corresponding
    // to structural-match types.
    let is_match = match (a.kind(), b.kind()) {
        (ty::ConstKind::Infer(_), _) | (_, ty::ConstKind::Infer(_)) => {
            // The caller should handle these cases!
            panic!("var types encountered in structurally_relate_consts: {:?} {:?}", a, b)
        }

        (ty::ConstKind::Error(_), _) => return Ok(a),
        (_, ty::ConstKind::Error(_)) => return Ok(b),

        (ty::ConstKind::Param(a_p), ty::ConstKind::Param(b_p)) if a_p.index() == b_p.index() => {
            // FIXME: Put this back
            // debug_assert_eq!(a_p.name, b_p.name, "param types with same index differ in name");
            true
        }
        (ty::ConstKind::Placeholder(p1), ty::ConstKind::Placeholder(p2)) => p1 == p2,
        (ty::ConstKind::Value(a_val), ty::ConstKind::Value(b_val)) => {
            match (a_val.valtree().kind(), b_val.valtree().kind()) {
                (ty::ValTreeKind::Leaf(scalar_a), ty::ValTreeKind::Leaf(scalar_b)) => {
                    scalar_a == scalar_b
                }
                (ty::ValTreeKind::Branch(branches_a), ty::ValTreeKind::Branch(branches_b))
                    if branches_a.len() == branches_b.len() =>
                {
                    branches_a
                        .iter()
                        .zip(branches_b.iter())
                        .all(|(a, b)| relation.relate(a, b).is_ok())
                }
                _ => false,
            }
        }

        // While this is slightly incorrect, it shouldn't matter for `min_const_generics`
        // and is the better alternative to waiting until `generic_const_exprs` can
        // be stabilized.
        (ty::ConstKind::Alias(is_rigid_a, au), ty::ConstKind::Alias(is_rigid_b, bu)) => {
            // Users shouldn't know about this so the mismatch should be caught
            // during development rather than presented as type error.
            debug_assert_eq!(is_rigid_a, is_rigid_b, "{a:?} != {b:?}");
            return Ok(Const::new_alias(cx, is_rigid_a, relation.relate(au, bu)?));
        }
        (ty::ConstKind::Expr(ae), ty::ConstKind::Expr(be)) => {
            let expr = relation.relate(ae, be)?;
            return Ok(Const::new_expr(cx, expr));
        }
        _ => false,
    };
    if is_match { Ok(a) } else { Err(TypeError::ConstMismatch(ExpectedFound::new(a, b))) }
}

impl<I: Interner, T: Relate<I>> Relate<I> for ty::Binder<I, T> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::Binder<I, T>,
        b: ty::Binder<I, T>,
    ) -> RelateResult<I, ty::Binder<I, T>> {
        relation.binders(a, b)
    }
}

impl<I: Interner> Relate<I> for ty::TraitClause<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::TraitClause<I>,
        b: ty::TraitClause<I>,
    ) -> RelateResult<I, ty::TraitClause<I>> {
        let trait_ref = relation.relate(a.trait_ref, b.trait_ref)?;
        if a.polarity != b.polarity {
            return Err(TypeError::PolarityMismatch(ExpectedFound::new(a.polarity, b.polarity)));
        }
        Ok(ty::TraitClause { trait_ref, polarity: a.polarity })
    }
}

impl<I: Interner> Relate<I> for ty::ProjectionClause<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::ProjectionClause<I>,
        b: ty::ProjectionClause<I>,
    ) -> RelateResult<I, ty::ProjectionClause<I>> {
        let projection_term = relation.relate(a.projection_term, b.projection_term)?;
        let term = relation.relate_with_variance(
            ty::Invariant,
            VarianceDiagInfo::default(),
            a.term,
            b.term,
        )?;
        Ok(ty::ProjectionClause { projection_term, term })
    }
}
