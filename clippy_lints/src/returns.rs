use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::source::{snippet_opt, snippet_with_context};
use clippy_utils::visitors::{for_each_expr, Descend};
use clippy_utils::{fn_def_id, path_to_local_id};
use core::ops::ControlFlow;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Block, Body, Expr, ExprKind, FnDecl, HirId, MatchSource, PatKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::BytePos;

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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_RETURN,
    style,
    "using a return statement like `return expr;` where an expression would suffice"
}

#[derive(PartialEq, Eq, Copy, Clone)]
enum RetReplacement {
    Empty,
    Block,
    Unit,
}

impl RetReplacement {
    fn sugg_help(self) -> &'static str {
        match self {
            Self::Empty => "remove `return`",
            Self::Block => "replace `return` with an empty block",
            Self::Unit => "replace `return` with a unit value",
        }
    }
}

impl ToString for RetReplacement {
    fn to_string(&self) -> String {
        match *self {
            Self::Empty => "",
            Self::Block => "{}",
            Self::Unit => "()",
        }
        .to_string()
    }
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
            if !local.span.from_expansion();
            then {
                span_lint_hir_and_then(
                    cx,
                    LET_AND_RETURN,
                    retexpr.hir_id,
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
                check_final_expr(cx, body.value, vec![], replacement);
            },
            FnKind::ItemFn(..) | FnKind::Method(..) => {
                check_block_return(cx, &body.value.kind, vec![]);
            },
        }
    }
}

// if `expr` is a block, check if there are needless returns in it
fn check_block_return<'tcx>(cx: &LateContext<'tcx>, expr_kind: &ExprKind<'tcx>, semi_spans: Vec<Span>) {
    if let ExprKind::Block(block, _) = expr_kind {
        if let Some(block_expr) = block.expr {
            check_final_expr(cx, block_expr, semi_spans, RetReplacement::Empty);
        } else if let Some(stmt) = block.stmts.iter().last() {
            match stmt.kind {
                StmtKind::Expr(expr) => {
                    check_final_expr(cx, expr, semi_spans, RetReplacement::Empty);
                },
                StmtKind::Semi(semi_expr) => {
                    let mut semi_spans_and_this_one = semi_spans;
                    // we only want the span containing the semicolon so we can remove it later. From `entry.rs:382`
                    if let Some(semicolon_span) = stmt.span.trim_start(semi_expr.span) {
                        semi_spans_and_this_one.push(semicolon_span);
                        check_final_expr(cx, semi_expr, semi_spans_and_this_one, RetReplacement::Empty);
                    }
                },
                _ => (),
            }
        }
    }
}

fn check_final_expr<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    semi_spans: Vec<Span>, /* containing all the places where we would need to remove semicolons if finding an
                            * needless return */
    replacement: RetReplacement,
) {
    let peeled_drop_expr = expr.peel_drop_temps();
    match &peeled_drop_expr.kind {
        // simple return is always "bad"
        ExprKind::Ret(ref inner) => {
            if cx.tcx.hir().attrs(expr.hir_id).is_empty() {
                let borrows = inner.map_or(false, |inner| last_statement_borrows(cx, inner));
                if !borrows {
                    // check if expr return nothing
                    let ret_span = if inner.is_none() && replacement == RetReplacement::Empty {
                        extend_span_to_previous_non_ws(cx, peeled_drop_expr.span)
                    } else {
                        peeled_drop_expr.span
                    };

                    emit_return_lint(cx, ret_span, semi_spans, inner.as_ref().map(|i| i.span), replacement);
                }
            }
        },
        ExprKind::If(_, then, else_clause_opt) => {
            check_block_return(cx, &then.kind, semi_spans.clone());
            if let Some(else_clause) = else_clause_opt {
                check_block_return(cx, &else_clause.kind, semi_spans);
            }
        },
        // a match expr, check all arms
        // an if/if let expr, check both exprs
        // note, if without else is going to be a type checking error anyways
        // (except for unit type functions) so we don't match it
        ExprKind::Match(_, arms, MatchSource::Normal) => {
            for arm in arms.iter() {
                check_final_expr(cx, arm.body, semi_spans.clone(), RetReplacement::Unit);
            }
        },
        // if it's a whole block, check it
        other_expr_kind => check_block_return(cx, other_expr_kind, semi_spans),
    }
}

fn emit_return_lint(
    cx: &LateContext<'_>,
    ret_span: Span,
    semi_spans: Vec<Span>,
    inner_span: Option<Span>,
    replacement: RetReplacement,
) {
    if ret_span.from_expansion() {
        return;
    }
    let mut applicability = Applicability::MachineApplicable;
    let return_replacement = inner_span.map_or_else(
        || replacement.to_string(),
        |inner_span| {
            let (snippet, _) = snippet_with_context(cx, inner_span, ret_span.ctxt(), "..", &mut applicability);
            snippet.to_string()
        },
    );
    let sugg_help = if inner_span.is_some() {
        "remove `return`"
    } else {
        replacement.sugg_help()
    };
    span_lint_and_then(cx, NEEDLESS_RETURN, ret_span, "unneeded `return` statement", |diag| {
        diag.span_suggestion_hidden(ret_span, sugg_help, return_replacement, applicability);
        // for each parent statement, we need to remove the semicolon
        for semi_stmt_span in semi_spans {
            diag.tool_only_span_suggestion(semi_stmt_span, "remove this semicolon", "", applicability);
        }
    });
}

fn last_statement_borrows<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    for_each_expr(expr, |e| {
        if let Some(def_id) = fn_def_id(cx, e)
            && cx
                .tcx
                .fn_sig(def_id)
                .skip_binder()
                .output()
                .walk()
                .any(|arg| matches!(arg.unpack(), GenericArgKind::Lifetime(_)))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(Descend::from(!expr.span.from_expansion()))
        }
    })
    .is_some()
}

// Go backwards while encountering whitespace and extend the given Span to that point.
#[expect(clippy::cast_possible_truncation)]
fn extend_span_to_previous_non_ws(cx: &LateContext<'_>, sp: Span) -> Span {
    if let Ok(prev_source) = cx.sess().source_map().span_to_prev_source(sp) {
        let ws = [' ', '\t', '\n'];
        if let Some(non_ws_pos) = prev_source.rfind(|c| !ws.contains(&c)) {
            let len = prev_source.len() - non_ws_pos - 1;
            return sp.with_lo(BytePos(sp.lo().0 - len as u32));
        }
    }

    sp
}
