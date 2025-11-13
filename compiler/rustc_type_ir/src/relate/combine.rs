use std::iter;

use tracing::debug;

use super::{
    ExpectedFound, RelateResult, StructurallyRelateAliases, TypeRelation,
    structurally_relate_consts, structurally_relate_tys,
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

    /// Whether aliases should be related structurally. This is pretty much
    /// always `No` unless you're equating in some specific locations of the
    /// new solver. See the comments in these use-cases for more details.
    fn structurally_relate_aliases(&self) -> StructurallyRelateAliases;

    /// Register obligations that must hold in order for this relation to hold
    fn register_goals(&mut self, obligations: impl IntoIterator<Item = Goal<I, I::Predicate>>);

    /// Register predicates that must hold in order for this relation to hold.
    /// This uses the default `param_env` of the obligation.
    fn register_predicates(
        &mut self,
        obligations: impl IntoIterator<Item: Upcast<I, I::Predicate>>,
    );

    /// Register `AliasRelate` obligation(s) that both types must be related to each other.
    fn register_alias_relate_predicate(&mut self, a: I::Ty, b: I::Ty);
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

        (_, ty::Alias(..)) | (ty::Alias(..), _) if infcx.next_trait_solver() => {
            match relation.structurally_relate_aliases() {
                StructurallyRelateAliases::Yes => structurally_relate_tys(relation, a, b),
                StructurallyRelateAliases::No => {
                    relation.register_alias_relate_predicate(a, b);
                    Ok(a)
                }
            }
        }

        // All other cases of inference are errors
        (ty::Infer(_), _) | (_, ty::Infer(_)) => Err(TypeError::Sorts(ExpectedFound::new(a, b))),

        (ty::Alias(ty::Opaque, _), _) | (_, ty::Alias(ty::Opaque, _)) => {
            assert!(!infcx.next_trait_solver());
            match infcx.typing_mode() {
                // During coherence, opaque types should be treated as *possibly*
                // equal to any other type. This is an
                // extremely heavy hammer, but can be relaxed in a forwards-compatible
                // way later.
                TypingMode::Coherence => {
                    relation.register_predicates([ty::Binder::dummy(ty::PredicateKind::Ambiguous)]);
                    Ok(a)
                }
                TypingMode::Analysis { .. }
                | TypingMode::Borrowck { .. }
                | TypingMode::PostBorrowckAnalysis { .. }
                | TypingMode::PostAnalysis => structurally_relate_tys(relation, a, b),
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
            infcx.instantiate_const_var_raw(relation, true, vid, b)?;
            Ok(b)
        }

        (_, ty::ConstKind::Infer(ty::InferConst::Var(vid))) => {
            infcx.instantiate_const_var_raw(relation, false, vid, a)?;
            Ok(a)
        }

        (ty::ConstKind::Unevaluated(..), _) | (_, ty::ConstKind::Unevaluated(..))
            if infcx.cx().features().generic_const_exprs() || infcx.next_trait_solver() =>
        {
            match relation.structurally_relate_aliases() {
                StructurallyRelateAliases::No => {
                    relation.register_predicates([if infcx.next_trait_solver() {
                        ty::PredicateKind::AliasRelate(
                            a.into(),
                            b.into(),
                            ty::AliasRelationDirection::Equate,
                        )
                    } else {
                        ty::PredicateKind::ConstEquate(a, b)
                    }]);

                    Ok(b)
                }
                StructurallyRelateAliases::Yes => structurally_relate_consts(relation, a, b),
            }
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
