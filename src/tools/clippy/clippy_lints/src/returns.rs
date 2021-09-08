use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet_opt;
use clippy_utils::{fn_def_id, in_macro, path_to_local_id};
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

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `let`-bindings, which are subsequently
    /// returned.
    ///
    /// ### Why is this bad?
    /// It is just extraneous code. Remove it to make your code
    /// more rusty.
    ///
    /// ### Example
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
    /// ### What it does
    /// Checks for return statements at the end of a block.
    ///
    /// ### Why is this bad?
    /// Removing the `return` and semicolon will make the code
    /// more rusty.
    ///
    /// ### Example
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
            if cx.tcx.hir().attrs(local.hir_id).is_empty();
            if let Some(initexpr) = &local.init;
            if let PatKind::Binding(_, local_id, _, _) = local.pat.kind;
            if path_to_local_id(retexpr, local_id);
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
                            if !cx.typeck_results().expr_adjustments(retexpr).is_empty() {
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
            FnKind::Closure => {
                // when returning without value in closure, replace this `return`
                // with an empty block to prevent invalid suggestion (see #6501)
                let replacement = if let ExprKind::Ret(None) = &body.value.kind {
                    RetReplacement::Block
                } else {
                    RetReplacement::Empty
                };
                check_final_expr(cx, &body.value, Some(body.value.span), replacement);
            },
            FnKind::ItemFn(..) | FnKind::Method(..) => {
                if let ExprKind::Block(block, _) = body.value.kind {
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
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => {
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
            let attrs = cx.tcx.hir().attrs(expr.hir_id);
            if !attrs.iter().any(attr_is_cfg) {
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
        ExprKind::Block(block, _) => {
            check_block_return(cx, block);
        },
        ExprKind::If(_, then, else_clause_opt) => {
            if let ExprKind::Block(ifblock, _) = then.kind {
                check_block_return(cx, ifblock);
            }
            if let Some(else_clause) = else_clause_opt {
                check_final_expr(cx, else_clause, None, RetReplacement::Empty);
            }
        },
        // a match expr, check all arms
        // an if/if let expr, check both exprs
        // note, if without else is going to be a type checking error anyways
        // (except for unit type functions) so we don't match it
        ExprKind::Match(_, arms, MatchSource::Normal) => {
            for arm in arms.iter() {
                check_final_expr(cx, arm.body, Some(arm.body.span), RetReplacement::Block);
            }
        },
        ExprKind::DropTemps(expr) => check_final_expr(cx, expr, None, RetReplacement::Empty),
        _ => (),
    }
}

fn emit_return_lint(cx: &LateContext<'_>, ret_span: Span, inner_span: Option<Span>, replacement: RetReplacement) {
    if ret_span.from_expansion() {
        return;
    }
    match inner_span {
        Some(inner_span) => {
            if in_external_macro(cx.tcx.sess, inner_span) || inner_span.from_expansion() {
                return;
            }

            span_lint_and_then(cx, NEEDLESS_RETURN, ret_span, "unneeded `return` statement", |diag| {
                if let Some(snippet) = snippet_opt(cx, inner_span) {
                    diag.span_suggestion(ret_span, "remove `return`", snippet, Applicability::MachineApplicable);
                }
            });
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
                .walk(self.cx.tcx)
                .any(|arg| matches!(arg.unpack(), GenericArgKind::Lifetime(_)));
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
