use rustc_hir::{self as hir, HirId};
use rustc_infer::traits;
use rustc_middle::ty::{Ty, TypeVisitableExt};
use rustc_span::{ErrorGuaranteed, Span};
use smallvec::SmallVec;

use crate::coercion::{CoerceMany, DynamicCoerceMany};
use crate::{Diverges, Expectation, FnCtxt, bug};

#[derive(Clone, Copy, Debug)]
pub(crate) struct IfExprParts<'tcx> {
    pub cond: &'tcx hir::Expr<'tcx>,
    pub then: &'tcx hir::Expr<'tcx>,
    pub else_branch: Option<&'tcx hir::Expr<'tcx>>,
}

#[derive(Clone, Debug)]
struct BranchBody<'tcx> {
    expr: &'tcx hir::Expr<'tcx>,
    ty: Ty<'tcx>,
    diverges: Diverges,
    span: Span,
}

#[derive(Clone, Debug)]
struct IfGuardedBranch<'tcx> {
    if_expr: &'tcx hir::Expr<'tcx>,
    cond_diverges: Diverges,
    body: BranchBody<'tcx>,
}

#[derive(Default, Debug)]
struct IfGuardedBranches<'tcx> {
    branches: SmallVec<[IfGuardedBranch<'tcx>; 4]>,
    cond_error: Option<ErrorGuaranteed>,
}

#[derive(Clone, Debug)]
enum IfChainTail<'tcx> {
    FinalElse(BranchBody<'tcx>),
    Missing(&'tcx hir::Expr<'tcx>),
}

impl<'tcx> IfChainTail<'tcx> {
    fn expr(&self) -> &'tcx hir::Expr<'tcx> {
        match &self {
            IfChainTail::FinalElse(else_branch) => else_branch.expr,
            IfChainTail::Missing(last_if_expr) => *last_if_expr,
        }
    }

    fn diverges(&self) -> Diverges {
        match &self {
            IfChainTail::FinalElse(else_branch) => else_branch.diverges,
            IfChainTail::Missing(_) => Diverges::Maybe,
        }
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn check_expr_if(
        &self,
        expr_id: HirId,
        sp: Span,
        parts: &IfExprParts<'tcx>,
        orig_expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let root_if_expr = self.tcx.hir_expect_expr(expr_id);
        let expected = orig_expected.try_structurally_resolve_and_adjust_for_branches(self, sp);

        let initial_diverges = self.diverges.get();

        let (guarded, tail) = self.collect_if_chain(root_if_expr, parts, expected);

        let coerce_to_ty = expected.coercion_target_type(self, sp);
        let mut coerce: DynamicCoerceMany<'_> = CoerceMany::new(coerce_to_ty);

        let tail_defines_return_position_impl_trait =
            self.return_position_impl_trait_from_match_expectation(orig_expected);

        for (idx, branch) in guarded.branches.iter().enumerate() {
            if idx > 0 {
                let merged_ty = coerce.merged_ty();
                self.ensure_if_branch_type(branch.if_expr.hir_id, merged_ty);
            }

            let branch_body = &branch.body;
            let next_else_expr =
                guarded.branches.get(idx + 1).map(|next| next.if_expr).or(tail.expr().into());
            let opt_prev_branch = if idx > 0 { guarded.branches.get(idx - 1) } else { None };
            let mut branch_cause = if let Some(next_else_expr) = next_else_expr {
                self.if_cause(
                    opt_prev_branch.unwrap_or(branch).if_expr.hir_id,
                    next_else_expr,
                    tail_defines_return_position_impl_trait,
                )
            } else {
                self.misc(branch_body.span)
            };
            let cause_span =
                if idx == 0 { Some(root_if_expr.span) } else { Some(branch_body.span) };

            self.coerce_if_arm(
                &mut coerce,
                &mut branch_cause,
                branch_body.expr,
                branch_body.ty,
                cause_span,
                opt_prev_branch.and_then(|b| b.body.span.into()),
            );
        }

        match &tail {
            IfChainTail::FinalElse(else_branch) => {
                let mut else_cause = self.if_cause(
                    expr_id,
                    else_branch.expr,
                    tail_defines_return_position_impl_trait,
                );
                self.coerce_if_arm(
                    &mut coerce,
                    &mut else_cause,
                    else_branch.expr,
                    else_branch.ty,
                    None,
                    guarded.branches.last().and_then(|b| b.body.span.into()),
                );
            }
            IfChainTail::Missing(last_if_expr) => {
                let hir::ExprKind::If(tail_cond, tail_then, _) = last_if_expr.kind else {
                    bug!("expected `if` expression, found {:#?}", last_if_expr);
                };
                self.if_fallback_coercion(last_if_expr.span, tail_cond, tail_then, &mut coerce);
            }
        }

        let mut tail_diverges = tail.diverges();
        for branch in guarded.branches.iter().rev() {
            tail_diverges = branch.cond_diverges | (branch.body.diverges & tail_diverges);
        }
        self.diverges.set(initial_diverges | tail_diverges);

        let result_ty = coerce.complete(self);

        let final_ty = if let Some(guar) = guarded.cond_error {
            Ty::new_error(self.tcx, guar)
        } else {
            result_ty
        };

        for branch in guarded.branches.iter().skip(1) {
            self.overwrite_if_branch_type(branch.if_expr.hir_id, final_ty);
        }
        if let Err(guar) = final_ty.error_reported() {
            self.set_tainted_by_errors(guar);
        }

        final_ty
    }

    fn coerce_if_arm(
        &self,
        coerce: &mut DynamicCoerceMany<'tcx>,
        cause: &mut traits::ObligationCause<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        ty: Ty<'tcx>,
        cause_span: Option<Span>,
        prev_branch_span: Option<Span>,
    ) {
        if let Some(span) = cause_span {
            cause.span = span;
        }
        coerce.coerce_inner(
            self,
            cause,
            Some(expr),
            ty,
            move |err| {
                if let Some(prev_branch_span) = prev_branch_span {
                    err.span_label(prev_branch_span, "expected because of this");
                }
            },
            false,
        );
    }

    fn check_if_condition(
        &self,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_span: Span,
    ) -> (Ty<'tcx>, Diverges) {
        let cond_ty = self.check_expr_has_type_or_error(cond_expr, self.tcx.types.bool, |_| {});
        self.warn_if_unreachable(
            cond_expr.hir_id,
            then_span,
            "block in `if` or `while` expression",
        );
        let cond_diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);
        (cond_ty, cond_diverges)
    }

    fn collect_if_chain(
        &self,
        mut current_if: &'tcx hir::Expr<'tcx>,
        parts: &IfExprParts<'tcx>,
        expected: Expectation<'tcx>,
    ) -> (IfGuardedBranches<'tcx>, IfChainTail<'tcx>) {
        let mut chain: IfGuardedBranches<'tcx> = IfGuardedBranches::default();
        let mut current_parts = *parts;

        loop {
            let Some(else_expr) =
                self.collect_if_branch(current_if, &current_parts, expected, &mut chain)
            else {
                return (chain, IfChainTail::Missing(current_if));
            };

            if let hir::ExprKind::If(cond, then, else_branch) = else_expr.kind {
                current_if = else_expr;
                current_parts = IfExprParts { cond, then, else_branch };
                continue;
            }

            return (chain, IfChainTail::FinalElse(self.check_branch_body(else_expr, expected)));
        }
    }

    fn collect_if_branch(
        &self,
        if_expr: &'tcx hir::Expr<'tcx>,
        parts: &IfExprParts<'tcx>,
        expected: Expectation<'tcx>,
        chain: &mut IfGuardedBranches<'tcx>,
    ) -> Option<&'tcx hir::Expr<'tcx>> {
        let (cond_ty, cond_diverges) = self.check_if_condition(parts.cond, parts.then.span);
        if let Err(guar) = cond_ty.error_reported() {
            chain.cond_error.get_or_insert(guar);
        }
        let branch_body = self.check_branch_body(parts.then, expected);

        chain.branches.push(IfGuardedBranch { if_expr, cond_diverges, body: branch_body });

        parts.else_branch
    }

    fn ensure_if_branch_type(&self, hir_id: HirId, ty: Ty<'tcx>) {
        let mut typeck = self.typeck_results.borrow_mut();
        let mut node_ty = typeck.node_types_mut();
        node_ty.entry(hir_id).or_insert(ty);
    }

    fn overwrite_if_branch_type(&self, hir_id: HirId, ty: Ty<'tcx>) {
        let mut typeck = self.typeck_results.borrow_mut();
        let mut node_ty = typeck.node_types_mut();
        node_ty.insert(hir_id, ty);
    }

    fn check_branch_body(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> BranchBody<'tcx> {
        self.diverges.set(Diverges::Maybe);
        let ty = self.check_expr_with_expectation(expr, expected);
        let diverges = self.diverges.get();
        self.diverges.set(Diverges::Maybe);
        let span = self.find_block_span_from_hir_id(expr.hir_id);
        BranchBody { expr, ty, diverges, span }
    }
}
