use if_chain::if_chain;
use rustc_ast::ast::Attribute;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::{Block, Body, Expr, ExprKind, FnDecl, HirId, MatchSource, PatKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::sym;

use crate::utils::{fn_def_id, in_macro, match_qpath, snippet_opt, span_lint_and_sugg, span_lint_and_then};

declare_clippy_lint! {
    /// **What it does:** Checks for `let`-bindings, which are subsequently
    /// returned.
    ///
    /// **Why is this bad?** It is just extraneous code. Remove it to make your code
    /// more rusty.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo() -> String {
    ///     let x = String::new();
    ///     x
    /// }
    /// ```
    /// instead, use
    /// ```
    /// fn foo() -> String {
    ///     String::new()
    /// }
    /// ```
    pub LET_AND_RETURN,
    style,
    "creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a block"
}

declare_clippy_lint! {
    /// **What it does:** Checks for return statements at the end of a block.
    ///
    /// **Why is this bad?** Removing the `return` and semicolon will make the code
    /// more rusty.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo(x: usize) -> usize {
    ///     return x;
    /// }
    /// ```
    /// simplify to
    /// ```rust
    /// fn foo(x: usize) -> usize {
    ///     x
    /// }
    /// ```
    pub NEEDLESS_RETURN,
    style,
    "using a return statement like `return expr;` where an expression would suffice"
}

#[derive(PartialEq, Eq, Copy, Clone)]
enum RetReplacement {
    Empty,
    Block,
}

declare_lint_pass!(Return => [LET_AND_RETURN, NEEDLESS_RETURN]);

impl<'tcx> LateLintPass<'tcx> for Return {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        // we need both a let-binding stmt and an expr
        if_chain! {
            if let Some(retexpr) = block.expr;
            if let Some(stmt) = block.stmts.iter().last();
            if let StmtKind::Local(local) = &stmt.kind;
            if local.ty.is_none();
            if local.attrs.is_empty();
            if let Some(initexpr) = &local.init;
            if let PatKind::Binding(.., ident, _) = local.pat.kind;
            if let ExprKind::Path(qpath) = &retexpr.kind;
            if match_qpath(qpath, &[&*ident.name.as_str()]);
            if !last_statement_borrows(cx, initexpr);
            if !in_external_macro(cx.sess(), initexpr.span);
            if !in_external_macro(cx.sess(), retexpr.span);
            if !in_external_macro(cx.sess(), local.span);
            if !in_macro(local.span);
            then {
                span_lint_and_then(
                    cx,
                    LET_AND_RETURN,
                    retexpr.span,
                    "returning the result of a `let` binding from a block",
                    |err| {
                        err.span_label(local.span, "unnecessary `let` binding");

                        if let Some(mut snippet) = snippet_opt(cx, initexpr.span) {
                            if !cx.typeck_results().expr_adjustments(&retexpr).is_empty() {
                                snippet.push_str(" as _");
                            }
                            err.multipart_suggestion(
                                "return the expression directly",
                                vec![
                                    (local.span, String::new()),
                                    (retexpr.span, snippet),
                                ],
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_help(initexpr.span, "this expression can be directly returned");
                        }
                    },
                );
            }
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        _: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        _: Span,
        _: HirId,
    ) {
        match kind {
            FnKind::Closure(_) => check_final_expr(cx, &body.value, Some(body.value.span), RetReplacement::Empty),
            FnKind::ItemFn(..) | FnKind::Method(..) => {
                if let ExprKind::Block(ref block, _) = body.value.kind {
                    check_block_return(cx, block);
                }
            },
        }
    }
}

fn attr_is_cfg(attr: &Attribute) -> bool {
    attr.meta_item_list().is_some() && attr.has_name(sym::cfg)
}

fn check_block_return<'tcx>(cx: &LateContext<'tcx>, block: &Block<'tcx>) {
    if let Some(expr) = block.expr {
        check_final_expr(cx, expr, Some(expr.span), RetReplacement::Empty);
    } else if let Some(stmt) = block.stmts.iter().last() {
        match stmt.kind {
            StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => {
                check_final_expr(cx, expr, Some(stmt.span), RetReplacement::Empty);
            },
            _ => (),
        }
    }
}

fn check_final_expr<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    span: Option<Span>,
    replacement: RetReplacement,
) {
    match expr.kind {
        // simple return is always "bad"
        ExprKind::Ret(ref inner) => {
            // allow `#[cfg(a)] return a; #[cfg(b)] return b;`
            if !expr.attrs.iter().any(attr_is_cfg) {
                let borrows = inner.map_or(false, |inner| last_statement_borrows(cx, inner));
                if !borrows {
                    emit_return_lint(
                        cx,
                        span.expect("`else return` is not possible"),
                        inner.as_ref().map(|i| i.span),
                        replacement,
                    );
                }
            }
        },
        // a whole block? check it!
        ExprKind::Block(ref block, _) => {
            check_block_return(cx, block);
        },
        // a match expr, check all arms
        // an if/if let expr, check both exprs
        // note, if without else is going to be a type checking error anyways
        // (except for unit type functions) so we don't match it
        ExprKind::Match(_, ref arms, source) => match source {
            MatchSource::Normal => {
                for arm in arms.iter() {
                    check_final_expr(cx, &arm.body, Some(arm.body.span), RetReplacement::Block);
                }
            },
            MatchSource::IfDesugar {
                contains_else_clause: true,
            }
            | MatchSource::IfLetDesugar {
                contains_else_clause: true,
            } => {
                if let ExprKind::Block(ref ifblock, _) = arms[0].body.kind {
                    check_block_return(cx, ifblock);
                }
                check_final_expr(cx, arms[1].body, None, RetReplacement::Empty);
            },
            _ => (),
        },
        _ => (),
    }
}

fn emit_return_lint(cx: &LateContext<'_>, ret_span: Span, inner_span: Option<Span>, replacement: RetReplacement) {
    match inner_span {
        Some(inner_span) => {
            if in_external_macro(cx.tcx.sess, inner_span) || inner_span.from_expansion() {
                return;
            }

            span_lint_and_then(cx, NEEDLESS_RETURN, ret_span, "unneeded `return` statement", |diag| {
                if let Some(snippet) = snippet_opt(cx, inner_span) {
                    diag.span_suggestion(ret_span, "remove `return`", snippet, Applicability::MachineApplicable);
                }
            })
        },
        None => match replacement {
            RetReplacement::Empty => {
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_RETURN,
                    ret_span,
                    "unneeded `return` statement",
                    "remove `return`",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            },
            RetReplacement::Block => {
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_RETURN,
                    ret_span,
                    "unneeded `return` statement",
                    "replace `return` with an empty block",
                    "{}".to_string(),
                    Applicability::MachineApplicable,
                );
            },
        },
    }
}

fn last_statement_borrows<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    let mut visitor = BorrowVisitor { cx, borrows: false };
    walk_expr(&mut visitor, expr);
    visitor.borrows
}

struct BorrowVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    borrows: bool,
}

impl<'tcx> Visitor<'tcx> for BorrowVisitor<'_, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.borrows {
            return;
        }

        if let Some(def_id) = fn_def_id(self.cx, expr) {
            self.borrows = self
                .cx
                .tcx
                .fn_sig(def_id)
                .output()
                .skip_binder()
                .walk()
                .any(|arg| matches!(arg.unpack(), GenericArgKind::Lifetime(_)));
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
