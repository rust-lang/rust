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
    /// structural relation for fully-instantiated proofs.
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

fn relate_evidence_args<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a: I::GenericArgs,
    b: I::GenericArgs,
) -> Result<I::GenericArgs, EvidenceRelateError<I>> {
    if a.len() != b.len() {
        return Err(EvidenceRelateError::DifferentProof);
    }
    Ok(relate_args_invariantly(relation, a, b)?)
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
            args: relate_evidence_args(relation, a_args, b_args)?,
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
                    instantiation: relate_evidence_args(
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
                    instantiation: relate_evidence_args(
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
                (Some(a), Some(b)) => Some(relate_evidence_args(relation, a, b)?),
                (None, None) => None,
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
                        ordinary_args: relate_evidence_args(
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
                            complete_early_args: relate_evidence_args(
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
                        instantiation: relate_evidence_args(
                            relation,
                            a_instantiation,
                            b_instantiation,
                        )?,
                    }
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
    if a.trait_ref.args.len() != b.trait_ref.args.len() {
        return Err(EvidenceRelateError::DifferentProof);
    }

    use ty::solve::TraitEvidenceKind;

    let kind = match (&a.kind, &b.kind) {
        (TraitEvidenceKind::Selected(a), TraitEvidenceKind::Selected(b)) => {
            let evidence = relate_candidate_evidence(relation, a, b)?;
            return Ok(relation.cx().mk_trait_evidence(evidence));
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
/// use the blanket [`Relate`] entry point. Relate selected proof recipes
/// structurally and expose a normal [`TypeError`] to callers.
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

fn relate_candidate_evidence<I: Interner, R: TypeRelation<I>>(
    relation: &mut R,
    a: &ty::solve::CandidateEvidence<I>,
    b: &ty::solve::CandidateEvidence<I>,
) -> Result<ty::solve::CandidateEvidence<I>, EvidenceRelateError<I>> {
    if a.root != b.root || a.nodes.len() != b.nodes.len() {
        trace!(?a, ?b, "evidence graph shape mismatch");
        return Err(EvidenceRelateError::DifferentProof);
    }
    let mut nodes = Vec::with_capacity(a.nodes.len());
    for (a_node, b_node) in a.nodes.iter().zip(&b.nodes) {
        if a_node.trait_ref.args.len() != b_node.trait_ref.args.len()
            || a_node.nested != b_node.nested
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
            let (
                CandidateEvidenceUse::Instantiated(a_evidence),
                CandidateEvidenceUse::Instantiated(b_evidence),
            ) = (a_nested, b_nested);
            nested_evidence.push(CandidateEvidenceUse::Instantiated(
                relation.evidences(a_evidence, b_evidence)?,
            ));
        }

        nodes.push(ty::solve::CandidateEvidenceNode {
            trait_ref,
            source,
            nested: a_node.nested.clone(),
            nested_evidence,
        });
    }

    let evidence = ty::solve::CandidateEvidence { root: a.root, nodes };
    evidence.assert_well_formed();
    Ok(evidence)
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
            return Err(TypeError::ProjectionMismatched(ExpectedFound::new(
                a.kind.into(),
                b.kind.into(),
            )));
        };

        let cx = relation.cx();
        let args = if let Some(variances) = cx.opt_alias_variances(kind) {
            if let ty::AliasTyKind::EvidenceProjection { projection } = kind {
                let parent_count = projection.trait_ref().args.len();
                cx.mk_args_from_iter(iter::zip(a.args.iter(), b.args.iter()).enumerate().map(
                    |(index, (a, b))| {
                        relation.relate_with_variance(
                            variances.get(parent_count + index).unwrap(),
                            VarianceDiagInfo::None,
                            a,
                            b,
                        )
                    },
                ))?
            } else {
                relate_args_with_variances(relation, variances, a.args, b.args)?
            }
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
        let mismatch = || {
            TypeError::ConstMismatch(ExpectedFound::new(
                Const::new_alias(cx, ty::IsRigid::yes_if_next_solver(cx), a),
                Const::new_alias(cx, ty::IsRigid::yes_if_next_solver(cx), b),
            ))
        };
        let kind = if a.kind == b.kind {
            a.kind
        } else if let (
            ty::AliasConstKind::EvidenceProjection { projection: a_projection },
            ty::AliasConstKind::EvidenceProjection { projection: b_projection },
        ) = (a.kind, b.kind)
        {
            match relate_evidence_projection(relation, a_projection, b_projection) {
                Ok(projection) => ty::AliasConstKind::EvidenceProjection { projection },
                Err(EvidenceRelateError::Type(error)) => return Err(error),
                Err(EvidenceRelateError::DifferentProof) => return Err(mismatch()),
            }
        } else {
            return Err(mismatch());
        };
        // FIXME(mgca): remove this
        debug_assert_eq!(a.type_of(cx).skip_norm_wip(), b.type_of(cx).skip_norm_wip());

        let args = relate_args_invariantly(relation, a.args, b.args)?;
        Ok(ty::AliasConst::new(cx, kind, args))
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
                return Err(TypeError::ProjectionMismatched(ExpectedFound::new(a.kind, b.kind)));
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
            if a_args.skip_binder().is_empty() {
                Ok(a)
            } else {
                // FIXME: this behavior is wrong; relations with binders needs fixing.
                //        need to relate the bound vars first.
                let x = relation.relate_ty_args(
                    a,
                    b,
                    a_def_id.into(),
                    a_args.skip_binder(),
                    b_args.skip_binder(),
                    |args| Ty::new_fn_def(cx, a_def_id, a_args.rebind(args)),
                );
                x
            }
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
