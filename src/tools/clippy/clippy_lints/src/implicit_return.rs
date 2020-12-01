use crate::utils::{fn_has_unsatisfiable_preds, match_panic_def_id, snippet_opt, span_lint_and_then};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, Expr, ExprKind, FnDecl, HirId, MatchSource, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for missing return statements at the end of a block.
    ///
    /// **Why is this bad?** Actually omitting the return keyword is idiomatic Rust code. Programmers
    /// coming from other languages might prefer the expressiveness of `return`. It's possible to miss
    /// the last returning statement because the only difference is a missing `;`. Especially in bigger
    /// code with multiple return paths having a `return` keyword makes it easier to find the
    /// corresponding statements.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo(x: usize) -> usize {
    ///     x
    /// }
    /// ```
    /// add return
    /// ```rust
    /// fn foo(x: usize) -> usize {
    ///     return x;
    /// }
    /// ```
    pub IMPLICIT_RETURN,
    restriction,
    "use a return statement like `return expr` instead of an expression"
}

declare_lint_pass!(ImplicitReturn => [IMPLICIT_RETURN]);

static LINT_BREAK: &str = "change `break` to `return` as shown";
static LINT_RETURN: &str = "add `return` as shown";

fn lint(cx: &LateContext<'_>, outer_span: Span, inner_span: Span, msg: &str) {
    let outer_span = outer_span.source_callsite();
    let inner_span = inner_span.source_callsite();

    span_lint_and_then(cx, IMPLICIT_RETURN, outer_span, "missing `return` statement", |diag| {
        if let Some(snippet) = snippet_opt(cx, inner_span) {
            diag.span_suggestion(
                outer_span,
                msg,
                format!("return {}", snippet),
                Applicability::MachineApplicable,
            );
        }
    });
}

fn expr_match(cx: &LateContext<'_>, expr: &Expr<'_>) {
    match expr.kind {
        // loops could be using `break` instead of `return`
        ExprKind::Block(block, ..) | ExprKind::Loop(block, ..) => {
            if let Some(expr) = &block.expr {
                expr_match(cx, expr);
            }
            // only needed in the case of `break` with `;` at the end
            else if let Some(stmt) = block.stmts.last() {
                if_chain! {
                    if let StmtKind::Semi(expr, ..) = &stmt.kind;
                    // make sure it's a break, otherwise we want to skip
                    if let ExprKind::Break(.., break_expr) = &expr.kind;
                    if let Some(break_expr) = break_expr;
                    then {
                            lint(cx, expr.span, break_expr.span, LINT_BREAK);
                    }
                }
            }
        },
        // use `return` instead of `break`
        ExprKind::Break(.., break_expr) => {
            if let Some(break_expr) = break_expr {
                lint(cx, expr.span, break_expr.span, LINT_BREAK);
            }
        },
        ExprKind::Match(.., arms, source) => {
            let check_all_arms = match source {
                MatchSource::IfLetDesugar {
                    contains_else_clause: has_else,
                } => has_else,
                _ => true,
            };

            if check_all_arms {
                for arm in arms {
                    expr_match(cx, &arm.body);
                }
            } else {
                expr_match(cx, &arms.first().expect("`if let` doesn't have a single arm").body);
            }
        },
        // skip if it already has a return statement
        ExprKind::Ret(..) => (),
        // make sure it's not a call that panics
        ExprKind::Call(expr, ..) => {
            if_chain! {
                if let ExprKind::Path(qpath) = &expr.kind;
                if let Some(path_def_id) = cx.qpath_res(qpath, expr.hir_id).opt_def_id();
                if match_panic_def_id(cx, path_def_id);
                then { }
                else {
                    lint(cx, expr.span, expr.span, LINT_RETURN)
                }
            }
        },
        // everything else is missing `return`
        _ => lint(cx, expr.span, expr.span, LINT_RETURN),
    }
}

impl<'tcx> LateLintPass<'tcx> for ImplicitReturn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        _: HirId,
    ) {
        let def_id = cx.tcx.hir().body_owner_def_id(body.id());

        // Building MIR for `fn`s with unsatisfiable preds results in ICE.
        if fn_has_unsatisfiable_preds(cx, def_id.to_def_id()) {
            return;
        }

        let mir = cx.tcx.optimized_mir(def_id.to_def_id());

        // checking return type through MIR, HIR is not able to determine inferred closure return types
        // make sure it's not a macro
        if !mir.return_ty().is_unit() && !span.from_expansion() {
            expr_match(cx, &body.value);
        }
    }
}
