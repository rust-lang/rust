use crate::utils::{in_macro_or_desugar, is_expn_of, snippet_opt, span_lint_and_then};
use rustc::hir::{intravisit::FnKind, Body, ExprKind, FnDecl, HirId, MatchSource};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::source_map::Span;

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

impl ImplicitReturn {
    fn lint(cx: &LateContext<'_, '_>, outer_span: syntax_pos::Span, inner_span: syntax_pos::Span, msg: &str) {
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
            ExprKind::Match(.., arms, source) => {
                let check_all_arms = match source {
                    MatchSource::IfLetDesugar {
                        contains_else_clause: has_else,
                    } => *has_else,
                    _ => true,
                };

                if check_all_arms {
                    for arm in arms {
                        Self::expr_match(cx, &arm.body);
                    }
                } else {
                    Self::expr_match(cx, &arms.first().expect("if let doesn't have a single arm").body);
                }
            },
            // skip if it already has a return statement
            ExprKind::Ret(..) => (),
            // everything else is missing `return`
            _ => {
                // make sure it's not just an unreachable expression
                if is_expn_of(expr.span, "unreachable").is_none() {
                    Self::lint(cx, expr.span, expr.span, "add `return` as shown")
                }
            },
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ImplicitReturn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        _: HirId,
    ) {
        let def_id = cx.tcx.hir().body_owner_def_id(body.id());
        let mir = cx.tcx.optimized_mir(def_id);

        // checking return type through MIR, HIR is not able to determine inferred closure return types
        // make sure it's not a macro
        if !mir.return_ty().is_unit() && !in_macro_or_desugar(span) {
            Self::expr_match(cx, &body.value);
        }
    }
}
