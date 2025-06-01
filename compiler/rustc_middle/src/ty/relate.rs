use std::iter;

pub use rustc_type_ir::relate::*;

use crate::ty::error::{ExpectedFound, TypeError};
use crate::ty::{self as ty, Ty, TyCtxt};

pub type RelateResult<'tcx, T> = rustc_type_ir::relate::RelateResult<TyCtxt<'tcx>, T>;

impl<'tcx> Relate<TyCtxt<'tcx>> for ty::ImplSubject<'tcx> {
    #[inline]
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: ty::ImplSubject<'tcx>,
        b: ty::ImplSubject<'tcx>,
    ) -> RelateResult<'tcx, ty::ImplSubject<'tcx>> {
        match (a, b) {
            (ty::ImplSubject::Trait(trait_ref_a), ty::ImplSubject::Trait(trait_ref_b)) => {
                let trait_ref = ty::TraitRef::relate(relation, trait_ref_a, trait_ref_b)?;
                Ok(ty::ImplSubject::Trait(trait_ref))
            }
            (ty::ImplSubject::Inherent(ty_a), ty::ImplSubject::Inherent(ty_b)) => {
                let ty = Ty::relate(relation, ty_a, ty_b)?;
                Ok(ty::ImplSubject::Inherent(ty))
            }
            (ty::ImplSubject::Trait(_), ty::ImplSubject::Inherent(_))
            | (ty::ImplSubject::Inherent(_), ty::ImplSubject::Trait(_)) => {
                bug!("can not relate TraitRef and Ty");
            }
        }
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for Ty<'tcx> {
    #[inline]
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>> {
        relation.tys(a, b)
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for ty::Pattern<'tcx> {
    #[inline]
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> RelateResult<'tcx, Self> {
        let tcx = relation.cx();
        match (&*a, &*b) {
            (
                &ty::PatternKind::Range { start: start_a, end: end_a },
                &ty::PatternKind::Range { start: start_b, end: end_b },
            ) => {
                let start = relation.relate(start_a, start_b)?;
                let end = relation.relate(end_a, end_b)?;
                Ok(tcx.mk_pat(ty::PatternKind::Range { start, end }))
            }
            (&ty::PatternKind::Or(a), &ty::PatternKind::Or(b)) => {
                if a.len() != b.len() {
                    return Err(TypeError::Mismatch);
                }
                let v = iter::zip(a, b).map(|(a, b)| relation.relate(a, b));
                let patterns = tcx.mk_patterns_from_iter(v)?;
                Ok(tcx.mk_pat(ty::PatternKind::Or(patterns)))
            }
            (ty::PatternKind::Range { .. } | ty::PatternKind::Or(_), _) => Err(TypeError::Mismatch),
        }
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>> {
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> RelateResult<'tcx, Self> {
        let tcx = relation.cx();
        // Fast path for when the auto traits do not match, or if the principals
        // are from different traits and therefore the projections definitely don't
        // match up.
        if a.len() != b.len() {
            return Err(TypeError::ExistentialMismatch(ExpectedFound::new(a, b)));
        }
        let v =
            iter::zip(a, b).map(|(ep_a, ep_b)| match (ep_a.skip_binder(), ep_b.skip_binder()) {
                (ty::ExistentialPredicate::Trait(a), ty::ExistentialPredicate::Trait(b)) => {
                    Ok(ep_a.rebind(ty::ExistentialPredicate::Trait(
                        relation.relate(ep_a.rebind(a), ep_b.rebind(b))?.skip_binder(),
                    )))
                }
                (
                    ty::ExistentialPredicate::Projection(a),
                    ty::ExistentialPredicate::Projection(b),
                ) => Ok(ep_a.rebind(ty::ExistentialPredicate::Projection(
                    relation.relate(ep_a.rebind(a), ep_b.rebind(b))?.skip_binder(),
                ))),
                (
                    ty::ExistentialPredicate::AutoTrait(a),
                    ty::ExistentialPredicate::AutoTrait(b),
                ) if a == b => Ok(ep_a.rebind(ty::ExistentialPredicate::AutoTrait(a))),
                _ => Err(TypeError::ExistentialMismatch(ExpectedFound::new(a, b))),
            });
        tcx.mk_poly_existential_predicates_from_iter(v)
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for ty::GenericArgsRef<'tcx> {
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: ty::GenericArgsRef<'tcx>,
        b: ty::GenericArgsRef<'tcx>,
    ) -> RelateResult<'tcx, ty::GenericArgsRef<'tcx>> {
        relate_args_invariantly(relation, a, b)
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for ty::Region<'tcx> {
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        relation.regions(a, b)
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for ty::Const<'tcx> {
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        relation.consts(a, b)
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for ty::Expr<'tcx> {
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        ae: ty::Expr<'tcx>,
        be: ty::Expr<'tcx>,
    ) -> RelateResult<'tcx, ty::Expr<'tcx>> {
        // FIXME(generic_const_exprs): is it possible to relate two consts which are not identical
        // exprs? Should we care about that?
        // FIXME(generic_const_exprs): relating the `ty()`s is a little weird since it is supposed to
        // ICE If they mismatch. Unfortunately `ConstKind::Expr` is a little special and can be thought
        // of as being generic over the argument types, however this is implicit so these types don't get
        // related when we relate the args of the item this const arg is for.
        match (ae.kind, be.kind) {
            (ty::ExprKind::Binop(a_binop), ty::ExprKind::Binop(b_binop)) if a_binop == b_binop => {}
            (ty::ExprKind::UnOp(a_unop), ty::ExprKind::UnOp(b_unop)) if a_unop == b_unop => {}
            (ty::ExprKind::FunctionCall, ty::ExprKind::FunctionCall) => {}
            (ty::ExprKind::Cast(a_kind), ty::ExprKind::Cast(b_kind)) if a_kind == b_kind => {}
            _ => return Err(TypeError::Mismatch),
        }

        let args = relation.relate(ae.args(), be.args())?;
        Ok(ty::Expr::new(ae.kind, args))
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for ty::GenericArg<'tcx> {
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: ty::GenericArg<'tcx>,
        b: ty::GenericArg<'tcx>,
    ) -> RelateResult<'tcx, ty::GenericArg<'tcx>> {
        match (a.kind(), b.kind()) {
            (ty::GenericArgKind::Lifetime(a_lt), ty::GenericArgKind::Lifetime(b_lt)) => {
                Ok(relation.relate(a_lt, b_lt)?.into())
            }
            (ty::GenericArgKind::Type(a_ty), ty::GenericArgKind::Type(b_ty)) => {
                Ok(relation.relate(a_ty, b_ty)?.into())
            }
            (ty::GenericArgKind::Const(a_ct), ty::GenericArgKind::Const(b_ct)) => {
                Ok(relation.relate(a_ct, b_ct)?.into())
            }
            _ => bug!("impossible case reached: can't relate: {a:?} with {b:?}"),
        }
    }
}

impl<'tcx> Relate<TyCtxt<'tcx>> for ty::Term<'tcx> {
    fn relate<R: TypeRelation<TyCtxt<'tcx>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> RelateResult<'tcx, Self> {
        Ok(match (a.kind(), b.kind()) {
            (ty::TermKind::Ty(a), ty::TermKind::Ty(b)) => relation.relate(a, b)?.into(),
            (ty::TermKind::Const(a), ty::TermKind::Const(b)) => relation.relate(a, b)?.into(),
            _ => return Err(TypeError::Mismatch),
        })
    }
}
