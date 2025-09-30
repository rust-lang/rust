use rustc_errors::Diag;
use rustc_hir::{self as hir, HirId};
use rustc_infer::traits;
use rustc_middle::ty::{Ty, TypeVisitableExt};
use rustc_span::{ErrorGuaranteed, Span};
use smallvec::SmallVec;

use crate::coercion::{CoerceMany, DynamicCoerceMany};
use crate::{Diverges, Expectation, FnCtxt, bug};

#[derive(Clone, Debug)]
struct BranchBody<'tcx> {
    expr: &'tcx hir::Expr<'tcx>,
    ty: Ty<'tcx>,
    diverges: Diverges,
    span: Span,
}

#[derive(Clone, Debug)]
struct IfBranch<'tcx> {
    if_expr: &'tcx hir::Expr<'tcx>,
    cond_diverges: Diverges,
    body: BranchBody<'tcx>,
}

#[derive(Default, Debug)]
struct IfChain<'tcx> {
    branches: SmallVec<[IfBranch<'tcx>; 4]>,
    cond_error: Option<ErrorGuaranteed>,
}

const RECENT_BRANCH_HISTORY_LIMIT: usize = 5;

#[derive(Default)]
struct RecentBranchTypeHistory<'tcx> {
    entries: SmallVec<[(Ty<'tcx>, Span); 4]>,
}

impl<'tcx> RecentBranchTypeHistory<'tcx> {
    fn diagnostic_snapshot(&self) -> SmallVec<[(Ty<'tcx>, Span); 4]> {
        self.entries.clone()
    }

    fn record(&mut self, ty: Ty<'tcx>, span: Span) {
        if ty.is_never() {
            return;
        }

        self.entries.push((ty, span));
        if self.entries.len() > RECENT_BRANCH_HISTORY_LIMIT {
            self.entries.remove(0);
        }
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn check_expr_if(
        &self,
        expr_id: HirId,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_expr: &'tcx hir::Expr<'tcx>,
        opt_else_expr: Option<&'tcx hir::Expr<'tcx>>,
        sp: Span,
        orig_expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let root_if_expr = self.tcx.hir_expect_expr(expr_id);
        if !self.if_chain_has_final_else(root_if_expr) {
            let expected = orig_expected.try_structurally_resolve_and_adjust_for_branches(self, sp);
            return self.evaluate_if_without_final_else(
                expr_id,
                cond_expr,
                then_expr,
                opt_else_expr,
                sp,
                orig_expected,
                expected,
            );
        }

        let expected = orig_expected.try_structurally_resolve_and_adjust_for_branches(self, sp);
        self.evaluate_if_chain_with_final_else(expr_id, root_if_expr, sp, orig_expected, expected)
    }

    fn evaluate_if_without_final_else(
        &self,
        expr_id: HirId,
        cond_expr: &'tcx hir::Expr<'tcx>,
        then_expr: &'tcx hir::Expr<'tcx>,
        opt_else_expr: Option<&'tcx hir::Expr<'tcx>>,
        sp: Span,
        orig_expected: Expectation<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let (cond_ty, cond_diverges) = self.check_if_condition(cond_expr, then_expr.span);

        let BranchBody { ty: then_ty, diverges: then_diverges, .. } =
            self.check_branch_body(then_expr, expected);

        // We've already taken the expected type's preferences
        // into account when typing the `then` branch. To figure
        // out the initial shot at a LUB, we thus only consider
        // `expected` if it represents a *hard* constraint
        // (`only_has_type`); otherwise, we just go with a
        // fresh type variable.
        let coerce_to_ty = expected.coercion_target_type(self, sp);
        let mut coerce: DynamicCoerceMany<'_> = CoerceMany::new(coerce_to_ty);

        coerce.coerce(self, &self.misc(sp), then_expr, then_ty);

        if let Some(else_expr) = opt_else_expr {
            let BranchBody { ty: else_ty, diverges: else_diverges, .. } =
                self.check_branch_body(else_expr, expected);

            let tail_defines_return_position_impl_trait =
                self.return_position_impl_trait_from_match_expectation(orig_expected);
            let if_cause =
                self.if_cause(expr_id, else_expr, tail_defines_return_position_impl_trait);

            coerce.coerce(self, &if_cause, else_expr, else_ty);

            // We won't diverge unless both branches do (or the condition does).
            self.diverges.set(cond_diverges | then_diverges & else_diverges);
        } else {
            self.if_fallback_coercion(sp, cond_expr, then_expr, &mut coerce);

            // If the condition is false we can't diverge.
            self.diverges.set(cond_diverges);
        }

        let result_ty = coerce.complete(self);
        if let Err(guar) = cond_ty.error_reported() {
            Ty::new_error(self.tcx, guar)
        } else {
            result_ty
        }
    }

    fn evaluate_if_chain_with_final_else(
        &self,
        expr_id: HirId,
        root_if_expr: &'tcx hir::Expr<'tcx>,
        sp: Span,
        orig_expected: Expectation<'tcx>,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        let mut chain = IfChain::default();

        let initial_diverges = self.diverges.get();
        let terminal_else = self.collect_if_chain(root_if_expr, expected, &mut chain);

        let Some(else_branch) = terminal_else else {
            bug!("sequential `if` chain expected a final `else` arm");
        };

        let coerce_to_ty = expected.coercion_target_type(self, sp);
        let mut coerce: DynamicCoerceMany<'_> = CoerceMany::new(coerce_to_ty);

        let tail_defines_return_position_impl_trait =
            self.return_position_impl_trait_from_match_expectation(orig_expected);
        let mut recent_branch_types = RecentBranchTypeHistory::default();

        for (idx, branch) in chain.branches.iter().enumerate() {
            if idx > 0 {
                let merged_ty = coerce.merged_ty();
                self.ensure_if_branch_type(branch.if_expr.hir_id, merged_ty);
            }

            let branch_body = &branch.body;
            let next_else_expr =
                chain.branches.get(idx + 1).map(|next| next.if_expr).unwrap_or(else_branch.expr);
            let mut branch_cause = self.if_cause(
                branch.if_expr.hir_id,
                next_else_expr,
                tail_defines_return_position_impl_trait,
            );
            let diag_info = recent_branch_types.diagnostic_snapshot();
            self.coerce_if_arm(
                &mut coerce,
                &mut branch_cause,
                branch_body.expr,
                branch_body.ty,
                branch_body.span,
                diag_info,
            );

            recent_branch_types.record(branch_body.ty, branch_body.span);
        }

        let mut else_cause =
            self.if_cause(expr_id, else_branch.expr, tail_defines_return_position_impl_trait);
        let diag_info = recent_branch_types.diagnostic_snapshot();
        self.coerce_if_arm(
            &mut coerce,
            &mut else_cause,
            else_branch.expr,
            else_branch.ty,
            else_branch.span,
            diag_info,
        );
        recent_branch_types.record(else_branch.ty, else_branch.span);

        let mut tail_diverges = else_branch.diverges;
        for branch in chain.branches.iter().rev() {
            tail_diverges = branch.cond_diverges | (branch.body.diverges & tail_diverges);
        }
        self.diverges.set(initial_diverges | tail_diverges);

        let result_ty = coerce.complete(self);

        let final_ty = if let Some(guar) = chain.cond_error {
            Ty::new_error(self.tcx, guar)
        } else {
            result_ty
        };

        for branch in chain.branches.iter().skip(1) {
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
        span: Span,
        prior_branches: SmallVec<[(Ty<'tcx>, Span); 4]>,
    ) {
        cause.span = span;
        coerce.coerce_inner(
            self,
            cause,
            Some(expr),
            ty,
            move |err| self.explain_if_branch_mismatch(err, span, &prior_branches),
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
        let cond_diverges = self.take_diverges();
        (cond_ty, cond_diverges)
    }

    fn if_chain_has_final_else(&self, mut current: &'tcx hir::Expr<'tcx>) -> bool {
        loop {
            match current.kind {
                hir::ExprKind::If(_, _, Some(else_expr)) => match else_expr.kind {
                    hir::ExprKind::If(..) => current = else_expr,
                    _ => return true,
                },
                _ => return false,
            }
        }
    }

    fn collect_if_chain(
        &self,
        mut current_if: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        chain: &mut IfChain<'tcx>,
    ) -> Option<BranchBody<'tcx>> {
        loop {
            let Some(else_expr) = self.collect_if_branch(current_if, expected, chain) else {
                return None;
            };

            if let hir::ExprKind::If(..) = else_expr.kind {
                current_if = else_expr;
                continue;
            }

            return Some(self.collect_final_else(else_expr, expected));
        }
    }

    fn collect_if_branch(
        &self,
        if_expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
        chain: &mut IfChain<'tcx>,
    ) -> Option<&'tcx hir::Expr<'tcx>> {
        let hir::ExprKind::If(cond_expr, then_expr, opt_else_expr) = if_expr.kind else {
            bug!("expected `if` expression, found {:#?}", if_expr);
        };

        let (cond_ty, cond_diverges) = self.check_if_condition(cond_expr, then_expr.span);
        if let Err(guar) = cond_ty.error_reported() {
            chain.cond_error.get_or_insert(guar);
        }
        let branch_body = self.check_branch_body(then_expr, expected);

        chain.branches.push(IfBranch { if_expr, cond_diverges, body: branch_body });

        opt_else_expr
    }

    fn collect_final_else(
        &self,
        else_expr: &'tcx hir::Expr<'tcx>,
        expected: Expectation<'tcx>,
    ) -> BranchBody<'tcx> {
        self.check_branch_body(else_expr, expected)
    }

    fn reset_diverges_to_maybe(&self) {
        self.diverges.set(Diverges::Maybe);
    }

    fn take_diverges(&self) -> Diverges {
        let diverges = self.diverges.get();
        self.reset_diverges_to_maybe();
        diverges
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
        self.reset_diverges_to_maybe();
        let ty = self.check_expr_with_expectation(expr, expected);
        let diverges = self.take_diverges();
        let span = self.find_block_span_from_hir_id(expr.hir_id);
        BranchBody { expr, ty, diverges, span }
    }

    fn explain_if_branch_mismatch(
        &self,
        err: &mut Diag<'_>,
        branch_span: Span,
        prior_branches: &[(Ty<'tcx>, Span)],
    ) {
        let Some(&(prior_ty, prior_span)) =
            prior_branches.iter().rev().find(|&&(_, span)| span != branch_span)
        else {
            return;
        };

        let expected_ty = self.resolve_vars_if_possible(prior_ty);
        err.span_label(
            prior_span,
            format!("expected `{}` because of this", self.ty_to_string(expected_ty)),
        );
    }
}
