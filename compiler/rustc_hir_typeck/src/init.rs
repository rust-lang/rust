#![allow(unused)]
use std::cell::RefCell;

use hir::{Expr, InitKind};
use rustc_abi::ExternAbi;
use rustc_hir::def::DefKind;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{self as hir, ExprKind};
use rustc_hir_analysis::check::check_function_signature;
use rustc_infer::infer::RegionVariableOrigin;
use rustc_infer::traits::{Obligation, WellFormedLoc};
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Binder, Ty, TyCtxt};
use rustc_span::def_id::LocalDefId;
use rustc_span::sym;
use rustc_trait_selection::traits::{ObligationCause, ObligationCauseCode};
use tracing::{debug, instrument};

use crate::coercion::CoerceMany;
use crate::expectation::Expectation;
use crate::gather_locals::GatherLocalsVisitor;
use crate::{CoroutineTypes, Diverges, FnCtxt};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    #[instrument(level = "debug", skip(self, kind))]
    pub(crate) fn check_init_tail(
        &self,
        kind: &InitKind<'tcx>,
        expected: Expectation<'tcx>,
        expr: &'tcx Expr<'tcx>,
    ) -> Ty<'tcx> {
        // If there is expectation, which should be very probable,
        // it should have been a tuple of a target type and a error type.
        let (ret_ty, err_ty) = if let Some(expected) = expected.to_option(self)
            && let &ty::Tuple(tys) = self.try_structurally_resolve_type(expr.span, expected).kind()
            && let [ret_ty, err_ty] = tys[..]
        {
            (ret_ty, err_ty)
        } else {
            (self.next_ty_var(expr.span), self.next_ty_var(expr.span))
        };
        let (ret_ty, err_ty) = self.check_init_tail_inner(kind, ret_ty, err_ty, expr);
        Ty::new_tup(self.tcx, &[ret_ty, err_ty])
    }

    #[instrument(level = "debug", skip(self, kind))]
    fn check_init_tail_inner(
        &self,
        kind: &InitKind<'tcx>,
        ret_ty: Ty<'tcx>,
        err_ty: Ty<'tcx>,
        expr: &'tcx Expr<'tcx>,
    ) -> (Ty<'tcx>, Ty<'tcx>) {
        match kind {
            &InitKind::Free(expr) => {
                let free_ty = self.try_structurally_resolve_type(
                    expr.span,
                    self.check_expr_with_hint(expr, ret_ty),
                );
                let nested_err_ty =
                    self.register_init_tail_obligation_and_project_err(free_ty, ret_ty, expr);
                let mut coerce =
                    CoerceMany::with_coercion_sites(err_ty, std::slice::from_ref(expr));
                coerce.coerce(self, &self.misc(expr.span), expr, nested_err_ty);
                let err_ty = coerce.complete(self);
                (ret_ty, err_ty)
            }
            &InitKind::Block(block, label) => {
                self.check_expr_block(
                    block,
                    Expectation::ExpectHasType(Ty::new_tup(self.tcx, &[ret_ty, err_ty])),
                );
                (ret_ty, err_ty)
            }
            &InitKind::Array(exprs) => {
                let mut coerce_ret =
                    CoerceMany::with_coercion_sites(ret_ty, std::slice::from_ref(expr));
                let mut coerce_err = CoerceMany::with_coercion_sites(err_ty, exprs);
                let elem_ty = self.next_ty_var(expr.span);
                let mut coerce_elem = CoerceMany::with_coercion_sites(elem_ty, exprs);
                for expr in exprs {
                    let ExprKind::InitTail(inner_kind) = expr.kind else {
                        span_bug!(
                            expr.span,
                            "expecting InitTail in HIR array expression, got {:?}",
                            expr.kind
                        )
                    };
                    let (inner_elem_ty, inner_err_ty) =
                        self.check_init_tail_inner(inner_kind, elem_ty, err_ty, expr);
                    coerce_err.coerce(
                        self,
                        &self.misc(expr.span),
                        expr,
                        self.resolve_vars_if_possible(inner_err_ty),
                    );
                    let inner_elem_ty = self.resolve_vars_if_possible(inner_elem_ty);
                    coerce_elem.coerce(self, &self.misc(expr.span), expr, inner_elem_ty);
                    self.write_ty(expr.hir_id, inner_elem_ty);
                }
                let elem_ty = self.resolve_vars_if_possible(coerce_elem.complete(self));
                coerce_ret.coerce(
                    self,
                    &self.misc(expr.span),
                    expr,
                    Ty::new_array(self.tcx, elem_ty, exprs.len() as u64),
                );
                let ret_ty = coerce_ret.complete(self);
                let err_ty = coerce_err.complete(self);
                (ret_ty, err_ty)
            }
            &InitKind::Repeat(_expr, _const_arg) => todo!("dxf: sorry"),
            &InitKind::Tuple(_exprs) => todo!("dxf: sorry"),
            &InitKind::Struct(_qpath, _expr_fields, _struct_tail_expr) => {
                todo!("dxf: sorry but this have to deviate from a typical HIR struct expr")
            }
        }
    }

    fn register_init_tail_obligation_and_project_err(
        &self,
        free_ty: Ty<'tcx>,
        ret_ty: Ty<'tcx>,
        expr: &'tcx Expr<'tcx>,
    ) -> Ty<'tcx> {
        // Now demand the `free_ty` to satisfy `$free_ty: Init<$ret_ty, Error = $err_ty>`
        let cause = self.misc(expr.span);
        // NOTE(@dingxiangfei2009): potential problem, what happens if the error type can be ty::Never?
        let predicate = ty::TraitPredicate {
            trait_ref: ty::TraitRef::new(
                self.tcx,
                self.tcx.require_lang_item(LangItem::Init, expr.span),
                [free_ty, ret_ty],
            ),
            polarity: ty::PredicatePolarity::Positive,
        };
        let obligation = Obligation::new(self.tcx, cause, self.param_env, predicate);
        self.register_predicate(obligation);
        Ty::new_projection(
            self.tcx,
            self.tcx.require_lang_item(LangItem::InitError, expr.span),
            [free_ty, ret_ty],
        )
    }
}
