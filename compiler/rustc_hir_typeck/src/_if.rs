use rustc_hir::{self as hir, HirId};
use rustc_infer::traits;
use rustc_middle::ty::{Ty, TypeVisitableExt};
use rustc_span::{ErrorGuaranteed, Span};
use smallvec::SmallVec;

use crate::coercion::{CoerceMany, DynamicCoerceMany};
use crate::{Diverges, Expectation, FnCtxt};

#[derive(Clone, Copy, Debug)]
pub(crate) struct IfExprParts<'tcx> {
    pub cond: &'tcx hir::Expr<'tcx>,
    pub then: &'tcx hir::Expr<'tcx>,
    pub else_branch: Option<&'tcx hir::Expr<'tcx>>,
}

#[derive(Clone, Copy, Debug)]
struct IfExprWithParts<'tcx> {
    expr: &'tcx hir::Expr<'tcx>,
    parts: IfExprParts<'tcx>,
}

#[derive(Clone, Debug, Default)]
struct IfChain<'tcx> {
    guarded_branches: SmallVec<[IfGuardedBranch<'tcx>; 4]>,
    tail: IfChainTail<'tcx>,
    error: Option<ErrorGuaranteed>,
}

impl<'tcx> IfChain<'tcx> {
    fn last_expr(&self) -> Option<&'tcx hir::Expr<'tcx>> {
        if let IfChainTail::FinalElse(final_else) = &self.tail {
            final_else.expr.into()
        } else {
            self.guarded_branches.last().map(|l| l.expr_with_parts.expr)
        }
    }
}

#[derive(Clone, Debug)]
struct IfGuardedBranch<'tcx> {
    expr_with_parts: IfExprWithParts<'tcx>,
    cond_diverges: Diverges,
    body: BranchBody<'tcx>,
}

#[derive(Clone, Debug)]
enum IfChainTail<'tcx> {
    FinalElse(BranchBody<'tcx>),
    Missing,
}

#[derive(Clone, Debug)]
struct BranchBody<'tcx> {
    expr: &'tcx hir::Expr<'tcx>,
    ty: Ty<'tcx>,
    diverges: Diverges,
    span: Span,
}

impl<'tcx> Default for IfChainTail<'tcx> {
    fn default() -> Self {
        IfChainTail::Missing
    }
}

impl<'tcx> IfChainTail<'tcx> {
    fn diverges(&self) -> Diverges {
        match &self {
            IfChainTail::FinalElse(else_branch) => else_branch.diverges,
            IfChainTail::Missing => Diverges::Maybe,
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

        let chain =
            self.collect_if_chain(&IfExprWithParts { expr: root_if_expr, parts: *parts }, expected);

        let coerce_to_ty = expected.coercion_target_type(self, sp);
        let mut coerce: DynamicCoerceMany<'_> = CoerceMany::new(coerce_to_ty);

        let tail_defines_return_position_impl_trait =
            self.return_position_impl_trait_from_match_expectation(orig_expected);

        for (idx, branch) in chain.guarded_branches.iter().enumerate() {
            if idx > 0 {
                let merged_ty = coerce.merged_ty();
                self.ensure_if_branch_type(branch.expr_with_parts.expr.hir_id, merged_ty);
            }

            let branch_body = &branch.body;
            let next_else_expr = chain
                .guarded_branches
                .get(idx + 1)
                .map(|next| next.expr_with_parts.expr)
                .or(chain.last_expr().into());
            let opt_prev_branch = if idx > 0 { chain.guarded_branches.get(idx - 1) } else { None };
            let mut branch_cause = if let Some(next_else_expr) = next_else_expr {
                self.if_cause(
                    opt_prev_branch.unwrap_or(branch).expr_with_parts.expr.hir_id,
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

        match &chain.tail {
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
                    chain.guarded_branches.last().and_then(|b| b.body.span.into()),
                );
            }
            IfChainTail::Missing => {
                let last_if = chain.guarded_branches.last().map(|l| l.expr_with_parts).unwrap();
                self.if_fallback_coercion(
                    last_if.expr.span,
                    last_if.parts.cond,
                    last_if.parts.then,
                    &mut coerce,
                );
            }
        }

        let mut tail_diverges = chain.tail.diverges();
        for branch in chain.guarded_branches.iter().rev() {
            tail_diverges = branch.cond_diverges | (branch.body.diverges & tail_diverges);
        }
        self.diverges.set(initial_diverges | tail_diverges);

        let result_ty = coerce.complete(self);

        let final_ty =
            if let Some(guar) = chain.error { Ty::new_error(self.tcx, guar) } else { result_ty };

        for branch in chain.guarded_branches.iter().skip(1) {
            self.overwrite_if_branch_type(branch.expr_with_parts.expr.hir_id, final_ty);
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
        expr_with_parts: &IfExprWithParts<'tcx>,
        expected: Expectation<'tcx>,
    ) -> IfChain<'tcx> {
        let mut chain: IfChain<'tcx> = IfChain::default();
        let mut current = *expr_with_parts;

        loop {
            self.collect_if_branch(&current, expected, &mut chain);

            match current.parts.else_branch {
                Some(current_else_branch) => {
                    if let hir::ExprKind::If(cond, then, else_branch) = current_else_branch.kind {
                        current = IfExprWithParts {
                            expr: current_else_branch,
                            parts: IfExprParts { cond, then, else_branch },
                        };
                    } else {
                        chain.tail = IfChainTail::FinalElse(
                            self.check_branch_body(current_else_branch, expected),
                        );
                        return chain;
                    }
                }
                None => return chain,
            }
        }
    }

    fn collect_if_branch(
        &self,
        expr_with_parts: &IfExprWithParts<'tcx>,
        expected: Expectation<'tcx>,
        chain: &mut IfChain<'tcx>,
    ) {
        let (cond_ty, cond_diverges) =
            self.check_if_condition(expr_with_parts.parts.cond, expr_with_parts.parts.then.span);
        if let Err(guar) = cond_ty.error_reported() {
            chain.error.get_or_insert(guar);
        }
        let branch_body = self.check_branch_body(expr_with_parts.parts.then, expected);

        chain.guarded_branches.push(IfGuardedBranch {
            expr_with_parts: *expr_with_parts,
            cond_diverges,
            body: branch_body,
        });
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
