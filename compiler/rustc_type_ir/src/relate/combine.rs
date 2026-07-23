use std::iter;

use tracing::debug;

use super::{
    ExpectedFound, RelateResult, TypeRelation, structurally_relate_consts, structurally_relate_tys,
};
use crate::error::TypeError;
use crate::inherent::*;
use crate::relate::VarianceDiagInfo;
use crate::solve::Goal;
use crate::visit::TypeVisitableExt as _;
use crate::{self as ty, InferCtxtLike, Interner, TypingMode, Upcast};

pub trait PredicateEmittingRelation<Infcx, I = <Infcx as InferCtxtLike>::Interner>:
    TypeRelation<I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn span(&self) -> I::Span;

    fn param_env(&self) -> I::ParamEnv;

    /// Register obligations that must hold in order for this relation to hold
    fn register_goals(&mut self, obligations: impl IntoIterator<Item = Goal<I, I::Predicate>>);

    /// Register predicates that must hold in order for this relation to hold.
    /// This uses the default `param_env` of the obligation.
    fn register_predicates(
        &mut self,
        obligations: impl IntoIterator<Item: Upcast<I, I::Predicate>>,
    );

    fn ambient_variance(&self) -> ty::Variance;
}

pub fn super_combine_tys<Infcx, I, R>(
    infcx: &Infcx,
    relation: &mut R,
    a: I::Ty,
    b: I::Ty,
) -> RelateResult<I, I::Ty>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    R: PredicateEmittingRelation<Infcx>,
{
    debug!("super_combine_tys::<{}>({:?}, {:?})", std::any::type_name::<R>(), a, b);
    debug_assert!(!a.has_escaping_bound_vars());
    debug_assert!(!b.has_escaping_bound_vars());

    match (a.kind(), b.kind()) {
        (ty::Error(e), _) | (_, ty::Error(e)) => {
            infcx.set_tainted_by_errors(e);
            return Ok(Ty::new_error(infcx.cx(), e));
        }

        // Relate integral variables to other types
        (ty::Infer(ty::IntVar(a_id)), ty::Infer(ty::IntVar(b_id))) => {
            infcx.equate_int_vids_raw(a_id, b_id);
            Ok(a)
        }
        (ty::Infer(ty::IntVar(v_id)), ty::Int(v)) => {
            infcx.instantiate_int_var_raw(v_id, ty::IntVarValue::IntType(v));
            Ok(b)
        }
        (ty::Int(v), ty::Infer(ty::IntVar(v_id))) => {
            infcx.instantiate_int_var_raw(v_id, ty::IntVarValue::IntType(v));
            Ok(a)
        }
        (ty::Infer(ty::IntVar(v_id)), ty::Uint(v)) => {
            infcx.instantiate_int_var_raw(v_id, ty::IntVarValue::UintType(v));
            Ok(b)
        }
        (ty::Uint(v), ty::Infer(ty::IntVar(v_id))) => {
            infcx.instantiate_int_var_raw(v_id, ty::IntVarValue::UintType(v));
            Ok(a)
        }

        // Relate floating-point variables to other types
        (ty::Infer(ty::FloatVar(a_id)), ty::Infer(ty::FloatVar(b_id))) => {
            infcx.equate_float_vids_raw(a_id, b_id);
            Ok(a)
        }
        (ty::Infer(ty::FloatVar(v_id)), ty::Float(v)) => {
            infcx.instantiate_float_var_raw(v_id, ty::FloatVarValue::Known(v));
            Ok(b)
        }
        (ty::Float(v), ty::Infer(ty::FloatVar(v_id))) => {
            infcx.instantiate_float_var_raw(v_id, ty::FloatVarValue::Known(v));
            Ok(a)
        }

        // We don't expect `TyVar` or `Fresh*` vars at this point with lazy norm.
        (ty::Alias(..), ty::Infer(ty::TyVar(_))) | (ty::Infer(ty::TyVar(_)), ty::Alias(..))
            if infcx.next_trait_solver() =>
        {
            panic!(
                "We do not expect to encounter `TyVar` this late in combine \
                    -- they should have been handled earlier"
            )
        }
        (_, ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)))
        | (ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)), _)
            if infcx.next_trait_solver() =>
        {
            panic!("We do not expect to encounter `Fresh` variables in the new solver")
        }

        (ty::Alias(ty::IsRigid::No, alias), _) | (_, ty::Alias(ty::IsRigid::No, alias))
            if infcx.next_trait_solver() =>
        {
            // If both sides are aliases, arbitrarily do the LHS first
            let terms_are_inverted = !matches!(a.kind(), ty::Alias(ty::IsRigid::No, _));
            let other = if terms_are_inverted { a } else { b };
            match (relation.ambient_variance(), terms_are_inverted) {
                (ty::Invariant, _) => relation.register_predicates([ty::ProjectionPredicate {
                    projection_term: alias.into(),
                    term: other.into(),
                }]),
                (ty::Covariant, false) | (ty::Contravariant, true) => {
                    // Generate a new var to represent `alias <: other`
                    // with `alias == ?A && ?A <: other`
                    let new_var = infcx.next_ty_infer();
                    relation.register_predicates([
                        ty::PredicateKind::Clause(ty::ClauseKind::Projection(
                            ty::ProjectionPredicate {
                                projection_term: alias.into(),
                                term: new_var.into(),
                            },
                        )),
                        ty::PredicateKind::Subtype(ty::SubtypePredicate {
                            a_is_expected: !terms_are_inverted,
                            a: new_var,
                            b: other,
                        }),
                    ]);
                }
                (ty::Contravariant, false) | (ty::Covariant, true) => {
                    // a :> b is b <: a
                    let new_var = infcx.next_ty_infer();
                    relation.register_predicates([
                        ty::PredicateKind::Clause(ty::ClauseKind::Projection(
                            ty::ProjectionPredicate {
                                projection_term: alias.into(),
                                term: new_var.into(),
                            },
                        )),
                        ty::PredicateKind::Subtype(ty::SubtypePredicate {
                            a_is_expected: terms_are_inverted,
                            a: other,
                            b: new_var,
                        }),
                    ]);
                }
                (ty::Bivariant, _) => {
                    unreachable!(
                        "cannot handle bivariant aliases in register_projection_with_variance"
                    )
                }
            }
            Ok(a)
        }

        // All other cases of inference are errors
        (ty::Infer(_), _) | (_, ty::Infer(_)) => Err(TypeError::Sorts(ExpectedFound::new(a, b))),

        (ty::Alias(_, ty::AliasTy { kind: ty::Opaque { .. }, .. }), _)
        | (_, ty::Alias(_, ty::AliasTy { kind: ty::Opaque { .. }, .. }))
            if !infcx.next_trait_solver() =>
        {
            match infcx.typing_mode_raw().assert_not_erased() {
                // During coherence, opaque types should be treated as *possibly*
                // equal to any other type. This is an
                // extremely heavy hammer, but can be relaxed in a forwards-compatible
                // way later.
                TypingMode::Coherence => {
                    relation.register_predicates([ty::Binder::dummy(ty::PredicateKind::Ambiguous)]);
                    Ok(a)
                }
                TypingMode::Typeck { .. }
                | TypingMode::Reflection
                | TypingMode::PostTypeckUntilBorrowck { .. }
                | TypingMode::PostBorrowck { .. }
                | TypingMode::PostAnalysis
                | TypingMode::Codegen => structurally_relate_tys(relation, a, b),
            }
        }

        _ => structurally_relate_tys(relation, a, b),
    }
}

pub fn super_combine_consts<Infcx, I, R>(
    infcx: &Infcx,
    relation: &mut R,
    a: I::Const,
    b: I::Const,
) -> RelateResult<I, I::Const>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    R: PredicateEmittingRelation<Infcx>,
{
    debug!("super_combine_consts::<{}>({:?}, {:?})", std::any::type_name::<R>(), a, b);
    debug_assert!(!a.has_escaping_bound_vars());
    debug_assert!(!b.has_escaping_bound_vars());

    if a == b {
        return Ok(a);
    }

    let a = infcx.shallow_resolve_const(a);
    let b = infcx.shallow_resolve_const(b);

    match (a.kind(), b.kind()) {
        (
            ty::ConstKind::Infer(ty::InferConst::Var(a_vid)),
            ty::ConstKind::Infer(ty::InferConst::Var(b_vid)),
        ) => {
            infcx.equate_const_vids_raw(a_vid, b_vid);
            Ok(a)
        }

        // All other cases of inference with other variables are errors.
        (ty::ConstKind::Infer(ty::InferConst::Var(_)), ty::ConstKind::Infer(_))
        | (ty::ConstKind::Infer(_), ty::ConstKind::Infer(ty::InferConst::Var(_))) => {
            panic!(
                "tried to combine ConstKind::Infer/ConstKind::Infer(InferConst::Var): {a:?} and {b:?}"
            )
        }

        (ty::ConstKind::Infer(ty::InferConst::Var(vid)), _) => {
            infcx.instantiate_const_var(relation, true, vid, b)?;
            Ok(b)
        }

        (_, ty::ConstKind::Infer(ty::InferConst::Var(vid))) => {
            infcx.instantiate_const_var(relation, false, vid, a)?;
            Ok(a)
        }

        (ty::ConstKind::Alias(ty::IsRigid::No, alias), _)
        | (_, ty::ConstKind::Alias(ty::IsRigid::No, alias))
            if (infcx.cx().features().generic_const_exprs() || infcx.next_trait_solver()) =>
        {
            if infcx.next_trait_solver() {
                let other = if matches!(a.kind(), ty::ConstKind::Alias(..)) { b } else { a };
                relation.register_predicates([ty::ProjectionPredicate {
                    projection_term: alias.into(),
                    term: other.into(),
                }])
            } else {
                relation.register_predicates([ty::PredicateKind::ConstEquate(a, b)]);
            }

            Ok(b)
        }

        _ => structurally_relate_consts(relation, a, b),
    }
}

pub fn combine_ty_args<Infcx, I, R>(
    infcx: &Infcx,
    relation: &mut R,
    a_ty: I::Ty,
    b_ty: I::Ty,
    variances: I::VariancesOf,
    a_args: I::GenericArgs,
    b_args: I::GenericArgs,
    mk: impl FnOnce(I::GenericArgs) -> I::Ty,
) -> RelateResult<I, I::Ty>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
    R: PredicateEmittingRelation<Infcx>,
{
    let cx = infcx.cx();
    let mut has_unconstrained_bivariant_arg = false;
    let args = iter::zip(a_args.iter(), b_args.iter()).enumerate().map(|(i, (a, b))| {
        let variance = variances.get(i).unwrap();
        let variance_info = match variance {
            ty::Invariant => {
                VarianceDiagInfo::Invariant { ty: a_ty, param_index: i.try_into().unwrap() }
            }
            ty::Covariant | ty::Contravariant => VarianceDiagInfo::default(),
            ty::Bivariant => {
                let has_non_region_infer = |arg: I::GenericArg| {
                    arg.has_non_region_infer()
                        && infcx.resolve_vars_if_possible(arg).has_non_region_infer()
                };
                if has_non_region_infer(a) || has_non_region_infer(b) {
                    has_unconstrained_bivariant_arg = true;
                }
                VarianceDiagInfo::default()
            }
        };
        relation.relate_with_variance(variance, variance_info, a, b)
    });
    let args = cx.mk_args_from_iter(args)?;

    // In general, we do not check whether all types which occur during
    // type checking are well-formed. We only check wf of user-provided types
    // and when actually using a type, e.g. for method calls.
    //
    // This means that when subtyping, we may end up with unconstrained
    // inference variables if a generalized type has bivariant parameters.
    // A parameter may only be bivariant if it is constrained by a projection
    // bound in a where-clause. As an example, imagine a type:
    //
    //     struct Foo<A, B> where A: Iterator<Item = B> {
    //         data: A
    //     }
    //
    // here, `A` will be covariant, but `B` is unconstrained. However, whatever it is,
    // for `Foo` to be WF, it must be equal to `A::Item`.
    //
    // If we have an input `Foo<?A, ?B>`, then after generalization we will wind
    // up with a type like `Foo<?C, ?D>`. When we enforce `Foo<?A, ?B> <: Foo<?C, ?D>`,
    // we will wind up with the requirement that `?A <: ?C`, but no particular
    // relationship between `?B` and `?D` (after all, these types may be completely
    // different). If we do nothing else, this may mean that `?D` goes unconstrained
    // (as in #41677). To avoid this we emit a `WellFormed` when relating types with
    // bivariant arguments.
    if has_unconstrained_bivariant_arg {
        relation.register_predicates([
            ty::ClauseKind::WellFormed(a_ty.into()),
            ty::ClauseKind::WellFormed(b_ty.into()),
        ]);
    }

    if a_args == args { Ok(a_ty) } else { Ok(mk(args)) }
}
