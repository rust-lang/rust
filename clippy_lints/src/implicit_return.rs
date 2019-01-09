use crate::utils::{in_macro, snippet_opt, span_lint_and_then};
use rustc::hir::{intravisit::FnKind, Body, ExprKind, FnDecl};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_errors::Applicability;
use syntax::{ast::NodeId, source_map::Span};

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
/// fn foo(x: usize) {
///     x
/// }
/// ```
/// add return
/// ```rust
/// fn foo(x: usize) {
///     return x;
/// }
/// ```
declare_clippy_lint! {
    pub IMPLICIT_RETURN,
    restriction,
    "use a return statement like `return expr` instead of an expression"
}

pub struct Pass;

impl Pass {
    fn lint(cx: &LateContext<'_, '_>, outer_span: syntax_pos::Span, inner_span: syntax_pos::Span, msg: &str) {
        span_lint_and_then(cx, IMPLICIT_RETURN, outer_span, "missing return statement", |db| {
            if let Some(snippet) = snippet_opt(cx, inner_span) {
                db.span_suggestion_with_applicability(
                    outer_span,
                    msg,
                    format!("return {}", snippet),
                    Applicability::MachineApplicable,
                );
            }
        });
    }

    fn expr_match(cx: &LateContext<'_, '_>, expr: &rustc::hir::Expr) {
        match &expr.node {
            // loops could be using `break` instead of `return`
            ExprKind::Block(block, ..) | ExprKind::Loop(block, ..) => {
                if let Some(expr) = &block.expr {
                    Self::expr_match(cx, expr);
                }
                // only needed in the case of `break` with `;` at the end
                else if let Some(stmt) = block.stmts.last() {
                    if let rustc::hir::StmtKind::Semi(expr, ..) = &stmt.node {
                        // make sure it's a break, otherwise we want to skip
                        if let ExprKind::Break(.., break_expr) = &expr.node {
                            if let Some(break_expr) = break_expr {
                                Self::lint(cx, expr.span, break_expr.span, "change `break` to `return` as shown");
                            }
                        }
                    }
                }
            },
            // use `return` instead of `break`
            ExprKind::Break(.., break_expr) => {
                if let Some(break_expr) = break_expr {
                    Self::lint(cx, expr.span, break_expr.span, "change `break` to `return` as shown");
                }
            },
            ExprKind::If(.., if_expr, else_expr) => {
                Self::expr_match(cx, if_expr);

                if let Some(else_expr) = else_expr {
                    Self::expr_match(cx, else_expr);
                }
            },
            ExprKind::Match(_, arms, ..) => {
                for arm in arms {
                    Self::expr_match(cx, &arm.body);
                }
            },
            // skip if it already has a return statement
            ExprKind::Ret(..) => (),
            // everything else is missing `return`
            _ => Self::lint(cx, expr.span, expr.span, "add `return` as shown"),
        }
    }
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(IMPLICIT_RETURN)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        _: NodeId,
    ) {
        let def_id = cx.tcx.hir().body_owner_def_id(body.id());
        let mir = cx.tcx.optimized_mir(def_id);

        // checking return type through MIR, HIR is not able to determine inferred closure return types
        // make sure it's not a macro
        if !mir.return_ty().is_unit() && !in_macro(span) {
            Self::expr_match(cx, &body.value);
        }
    }
}
