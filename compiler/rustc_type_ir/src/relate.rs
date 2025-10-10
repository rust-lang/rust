use std::iter;

use derive_where::derive_where;
use rustc_ast_ir::Mutability;
use tracing::{instrument, trace};

use crate::error::{ExpectedFound, TypeError};
use crate::fold::TypeFoldable;
use crate::inherent::*;
use crate::{self as ty, Interner};

pub mod combine;
pub mod solver_relating;

pub type RelateResult<I, T> = Result<T, TypeError<I>>;

/// Whether aliases should be related structurally or not. Used
/// to adjust the behavior of generalization and combine.
///
/// This should always be `No` unless in a few special-cases when
/// instantiating canonical responses and in the new solver. Each
/// such case should have a comment explaining why it is used.
#[derive(Debug, Copy, Clone)]
pub enum StructurallyRelateAliases {
    Yes,
    No,
}

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

    /// Relate the two args for the given item. The default
    /// is to look up the variance for the item and proceed
    /// accordingly.
    #[instrument(skip(self), level = "trace")]
    fn relate_item_args(
        &mut self,
        item_def_id: I::DefId,
        a_arg: I::GenericArgs,
        b_arg: I::GenericArgs,
    ) -> RelateResult<I, I::GenericArgs> {
        let cx = self.cx();
        let opt_variances = cx.variances_of(item_def_id);
        relate_args_with_variances(self, item_def_id, opt_variances, a_arg, b_arg, true)
    }

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

    fn regions(&mut self, a: I::Region, b: I::Region) -> RelateResult<I, I::Region>;

    fn consts(&mut self, a: I::Const, b: I::Const) -> RelateResult<I, I::Const>;

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
    ty_def_id: I::DefId,
    variances: I::VariancesOf,
    a_arg: I::GenericArgs,
    b_arg: I::GenericArgs,
    fetch_ty_for_diag: bool,
) -> RelateResult<I, I::GenericArgs> {
    let cx = relation.cx();

    let mut cached_ty = None;
    let params = iter::zip(a_arg.iter(), b_arg.iter()).enumerate().map(|(i, (a, b))| {
        let variance = variances.get(i).unwrap();
        let variance_info = if variance == ty::Invariant && fetch_ty_for_diag {
            let ty = *cached_ty.get_or_insert_with(|| cx.type_of(ty_def_id).instantiate(cx, a_arg));
            VarianceDiagInfo::Invariant { ty, param_index: i.try_into().unwrap() }
        } else {
            VarianceDiagInfo::default()
        };
        relation.relate_with_variance(variance, variance_info, a, b)
    });

    cx.mk_args_from_iter(params)
}

impl<I: Interner> Relate<I> for ty::FnSig<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::FnSig<I>,
        b: ty::FnSig<I>,
    ) -> RelateResult<I, ty::FnSig<I>> {
        let cx = relation.cx();

        if a.c_variadic != b.c_variadic {
            return Err(TypeError::VariadicMismatch({
                let a = a.c_variadic;
                let b = b.c_variadic;
                ExpectedFound::new(a, b)
            }));
        }

        if a.safety != b.safety {
            return Err(TypeError::SafetyMismatch(ExpectedFound::new(a.safety, b.safety)));
        }

        if a.abi != b.abi {
            return Err(TypeError::AbiMismatch(ExpectedFound::new(a.abi, b.abi)));
        };

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
            c_variadic: a.c_variadic,
            safety: a.safety,
            abi: a.abi,
        })
    }
}

impl<I: Interner> Relate<I> for ty::AliasTy<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::AliasTy<I>,
        b: ty::AliasTy<I>,
    ) -> RelateResult<I, ty::AliasTy<I>> {
        if a.def_id != b.def_id {
            Err(TypeError::ProjectionMismatched({
                let a = a.def_id;
                let b = b.def_id;
                ExpectedFound::new(a, b)
            }))
        } else {
            let cx = relation.cx();
            let args = if let Some(variances) = cx.opt_alias_variances(a.kind(cx), a.def_id) {
                relate_args_with_variances(
                    relation, a.def_id, variances, a.args, b.args,
                    false, // do not fetch `type_of(a_def_id)`, as it will cause a cycle
                )?
            } else {
                relate_args_invariantly(relation, a.args, b.args)?
            };
            Ok(ty::AliasTy::new_from_args(relation.cx(), a.def_id, args))
        }
    }
}

impl<I: Interner> Relate<I> for ty::AliasTerm<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::AliasTerm<I>,
        b: ty::AliasTerm<I>,
    ) -> RelateResult<I, ty::AliasTerm<I>> {
        if a.def_id != b.def_id {
            Err(TypeError::ProjectionMismatched({
                let a = a.def_id;
                let b = b.def_id;
                ExpectedFound::new(a, b)
            }))
        } else {
            let args = match a.kind(relation.cx()) {
                ty::AliasTermKind::OpaqueTy => relate_args_with_variances(
                    relation,
                    a.def_id,
                    relation.cx().variances_of(a.def_id),
                    a.args,
                    b.args,
                    false, // do not fetch `type_of(a_def_id)`, as it will cause a cycle
                )?,
                ty::AliasTermKind::ProjectionTy
                | ty::AliasTermKind::FreeConst
                | ty::AliasTermKind::FreeTy
                | ty::AliasTermKind::InherentTy
                | ty::AliasTermKind::InherentConst
                | ty::AliasTermKind::UnevaluatedConst
                | ty::AliasTermKind::ProjectionConst => {
                    relate_args_invariantly(relation, a.args, b.args)?
                }
            };
            Ok(ty::AliasTerm::new_from_args(relation.cx(), a.def_id, args))
        }
    }
}

impl<I: Interner> Relate<I> for ty::ExistentialProjection<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::ExistentialProjection<I>,
        b: ty::ExistentialProjection<I>,
    ) -> RelateResult<I, ty::ExistentialProjection<I>> {
        if a.def_id != b.def_id {
            Err(TypeError::ProjectionMismatched({
                let a = a.def_id;
                let b = b.def_id;
                ExpectedFound::new(a, b)
            }))
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
            Ok(if a_args.is_empty() {
                a
            } else {
                let args = relation.relate_item_args(a_def.def_id().into(), a_args, b_args)?;
                if args == a_args { a } else { Ty::new_adt(cx, a_def, args) }
            })
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
            Ok(if a_args.is_empty() {
                a
            } else {
                let args = relation.relate_item_args(a_def_id.into(), a_args, b_args)?;
                if args == a_args { a } else { Ty::new_fn_def(cx, a_def_id, args) }
            })
        }

        (ty::FnPtr(a_sig_tys, a_hdr), ty::FnPtr(b_sig_tys, b_hdr)) => {
            let fty = relation.relate(a_sig_tys.with(a_hdr), b_sig_tys.with(b_hdr))?;
            Ok(Ty::new_fn_ptr(cx, fty))
        }

        // Alias tend to mostly already be handled downstream due to normalization.
        (ty::Alias(a_kind, a_data), ty::Alias(b_kind, b_data)) => {
            let alias_ty = relation.relate(a_data, b_data)?;
            assert_eq!(a_kind, b_kind);
            Ok(Ty::new_alias(cx, a_kind, alias_ty))
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
/// Any semantic equality, e.g. of unevaluated consts, and inference variables have
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
            a_val.valtree() == b_val.valtree()
        }

        // While this is slightly incorrect, it shouldn't matter for `min_const_generics`
        // and is the better alternative to waiting until `generic_const_exprs` can
        // be stabilized.
        (ty::ConstKind::Unevaluated(au), ty::ConstKind::Unevaluated(bu)) if au.def == bu.def => {
            if cfg!(debug_assertions) {
                let a_ty = cx.type_of(au.def).instantiate(cx, au.args);
                let b_ty = cx.type_of(bu.def).instantiate(cx, bu.args);
                assert_eq!(a_ty, b_ty);
            }

            let args = relation.relate_with_variance(
                ty::Invariant,
                VarianceDiagInfo::default(),
                au.args,
                bu.args,
            )?;
            return Ok(Const::new_unevaluated(cx, ty::UnevaluatedConst { def: au.def, args }));
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

impl<I: Interner> Relate<I> for ty::TraitPredicate<I> {
    fn relate<R: TypeRelation<I>>(
        relation: &mut R,
        a: ty::TraitPredicate<I>,
        b: ty::TraitPredicate<I>,
    ) -> RelateResult<I, ty::TraitPredicate<I>> {
        let trait_ref = relation.relate(a.trait_ref, b.trait_ref)?;
        if a.polarity != b.polarity {
            return Err(TypeError::PolarityMismatch(ExpectedFound::new(a.polarity, b.polarity)));
        }
        Ok(ty::TraitPredicate { trait_ref, polarity: a.polarity })
    }
}
