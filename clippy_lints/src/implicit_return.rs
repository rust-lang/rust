use crate::utils::{
    in_macro_or_desugar, match_def_path,
    paths::{BEGIN_PANIC, BEGIN_PANIC_FMT},
    resolve_node, snippet_opt, span_lint_and_then,
};
use if_chain::if_chain;
use rustc::{
    declare_lint_pass, declare_tool_lint,
    hir::{intravisit::FnKind, Body, Expr, ExprKind, FnDecl, HirId, MatchSource, StmtKind},
    lint::{LateContext, LateLintPass, LintArray, LintPass},
};
use rustc_errors::Applicability;
use syntax_pos::Span;

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

fn lint(cx: &LateContext<'_, '_>, outer_span: Span, inner_span: Span, msg: &str) {
    let outer_span = span_to_outer_expn(outer_span);
    let inner_span = span_to_outer_expn(inner_span);

    span_lint_and_then(cx, IMPLICIT_RETURN, outer_span, "missing return statement", |db| {
        if let Some(snippet) = snippet_opt(cx, inner_span) {
            db.span_suggestion(
                outer_span,
                msg,
                format!("return {}", snippet),
                Applicability::MachineApplicable,
            );
        }
    });
}

fn span_to_outer_expn(span: Span) -> Span {
    if let Some(expr) = span.ctxt().outer_expn_info() {
        span_to_outer_expn(expr.call_site)
    } else {
        span
    }
}

fn expr_match(cx: &LateContext<'_, '_>, expr: &Expr, ret_ty_is_never: bool) {
    match &expr.node {
        // loops could be using `break` instead of `return`
        ExprKind::Block(block, ..) | ExprKind::Loop(block, ..) => {
            if let Some(expr) = &block.expr {
                expr_match(cx, expr, ret_ty_is_never);
            }
            // only needed in the case of `break` with `;` at the end
            else if let Some(stmt) = block.stmts.last() {
                if_chain! {
                    if let StmtKind::Semi(expr, ..) = &stmt.node;
                    // make sure it's a break, otherwise we want to skip
                    if let ExprKind::Break(.., break_expr) = &expr.node;
                    if let Some(break_expr) = break_expr;
                    then {
                            lint(cx, expr.span, break_expr.span, "change `break` to `return` as shown");
                    }
                }
            }
        },
        // use `return` instead of `break`
        ExprKind::Break(.., break_expr) => {
            if let Some(break_expr) = break_expr {
                lint(cx, expr.span, break_expr.span, "change `break` to `return` as shown");
            }
        },
        ExprKind::Match(.., arms, source) => {
            let check_all_arms = match source {
                MatchSource::IfLetDesugar {
                    contains_else_clause: has_else,
                } => *has_else,
                _ => true,
            };

            if check_all_arms {
                for arm in arms {
                    expr_match(cx, &arm.body, ret_ty_is_never);
                }
            } else {
                expr_match(
                    cx,
                    &arms.first().expect("if let doesn't have a single arm").body,
                    ret_ty_is_never,
                );
            }
        },
        // skip if it already has a return statement
        ExprKind::Ret(..) => (),
        // make sure it's not a call that panics unless we intend to return a panic
        ExprKind::Call(expr, ..) => {
            if_chain! {
                if let ExprKind::Path(qpath) = &expr.node;
                if let Some(path_def_id) = resolve_node(cx, qpath, expr.hir_id).opt_def_id();
                if match_def_path(cx, path_def_id, &BEGIN_PANIC) ||
                    match_def_path(cx, path_def_id, &BEGIN_PANIC_FMT);
                then {
                    // only put `return` on panics if the return type of the function/closure is a panic
                    if ret_ty_is_never {
                        lint(cx, expr.span, expr.span, "add `return` as shown")
                    }
                }
                else {
                    lint(cx, expr.span, expr.span, "add `return` as shown")
                }
            }
        },
        // everything else is missing `return`
        _ => lint(cx, expr.span, expr.span, "add `return` as shown"),
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ImplicitReturn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl,
        body: &'tcx Body,
        span: syntax::source_map::Span,
        _: HirId,
    ) {
        let def_id = cx.tcx.hir().body_owner_def_id(body.id());
        let mir = cx.tcx.optimized_mir(def_id);
        let ret_ty = mir.return_ty();

        // checking return type through MIR, HIR is not able to determine inferred closure return types
        // make sure it's not a macro
        if !ret_ty.is_unit() && !in_macro_or_desugar(span) {
            expr_match(cx, &body.value, ret_ty.is_never());
        }
    }
}
