//! A subset of a mir body used for const evaluatability checking.
use crate::ty::{self, Const, EarlyBinder, FallibleTypeFolder, GenericArg, TyCtxt, TypeFoldable};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::DefId;

#[derive(Hash, Debug, Clone, Copy, Ord, PartialOrd, PartialEq, Eq)]
#[derive(TyDecodable, TyEncodable, HashStable, TypeVisitable, TypeFoldable)]
pub enum CastKind {
    /// thir::ExprKind::As
    As,
    /// thir::ExprKind::Use
    Use,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum NotConstEvaluatable {
    Error(ErrorGuaranteed),
    MentionsInfer,
    MentionsParam,
}

impl From<ErrorGuaranteed> for NotConstEvaluatable {
    fn from(e: ErrorGuaranteed) -> NotConstEvaluatable {
        NotConstEvaluatable::Error(e)
    }
}

TrivialTypeTraversalAndLiftImpls! {
    NotConstEvaluatable,
}

pub type BoundAbstractConst<'tcx> = Result<Option<EarlyBinder<ty::Const<'tcx>>>, ErrorGuaranteed>;

impl<'tcx> TyCtxt<'tcx> {
    /// Returns a const with substs applied by
    pub fn bound_abstract_const(
        self,
        uv: ty::WithOptConstParam<DefId>,
    ) -> BoundAbstractConst<'tcx> {
        self.thir_abstract_const_opt_const_arg(uv).map(|ac| ac.map(|ac| EarlyBinder(ac)))
    }
    #[inline]
    pub fn thir_abstract_const_opt_const_arg(
        self,
        def: ty::WithOptConstParam<DefId>,
    ) -> Result<Option<ty::Const<'tcx>>, ErrorGuaranteed> {
        if let Some((did, param_did)) = def.as_const_arg() {
            self.thir_abstract_const_of_const_arg((did, param_did))
        } else {
            self.thir_abstract_const(def.did)
        }
    }

    pub fn expand_bound_abstract_const(
        self,
        ct: BoundAbstractConst<'tcx>,
        substs: &[GenericArg<'tcx>],
    ) -> Result<Option<Const<'tcx>>, ErrorGuaranteed> {
        struct Expander<'tcx> {
            tcx: TyCtxt<'tcx>,
        }
        impl<'tcx> FallibleTypeFolder<'tcx> for Expander<'tcx> {
            type Error = ErrorGuaranteed;
            fn tcx(&self) -> TyCtxt<'tcx> {
                self.tcx
            }
            fn try_fold_const(&mut self, c: Const<'tcx>) -> Result<Const<'tcx>, ErrorGuaranteed> {
                use ty::ConstKind::*;
                let uv = match c.kind() {
                    Unevaluated(uv) => uv,
                    Param(..) | Infer(..) | Bound(..) | Placeholder(..) | Value(..) | Error(..) => {
                        return Ok(c);
                    }
                    Expr(e) => {
                        let new_expr = match e {
                            ty::Expr::Binop(op, l, r) => {
                                ty::Expr::Binop(op, l.try_fold_with(self)?, r.try_fold_with(self)?)
                            }
                            ty::Expr::UnOp(op, v) => ty::Expr::UnOp(op, v.try_fold_with(self)?),
                            ty::Expr::Cast(k, c, t) => {
                                ty::Expr::Cast(k, c.try_fold_with(self)?, t.try_fold_with(self)?)
                            }
                            ty::Expr::FunctionCall(func, args) => ty::Expr::FunctionCall(
                                func.try_fold_with(self)?,
                                args.try_fold_with(self)?,
                            ),
                        };
                        return Ok(self.tcx().mk_const(ty::ConstKind::Expr(new_expr), c.ty()));
                    }
                };
                let bac = self.tcx.bound_abstract_const(uv.def);
                let ac = self.tcx.expand_bound_abstract_const(bac, uv.substs);
                if let Ok(Some(ac)) = ac { ac.try_fold_with(self) } else { Ok(c) }
            }
        }

        let Some(ac) = ct? else {
            return Ok(None);
        };
        let ac = ac.subst(self, substs);
        Ok(Some(ac.try_fold_with(&mut Expander { tcx: self })?))
    }
}
