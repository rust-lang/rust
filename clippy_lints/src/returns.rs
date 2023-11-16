use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::source::{snippet_opt, snippet_with_context};
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::visitors::{for_each_expr_with_closures, Descend};
use clippy_utils::{fn_def_id, is_from_proc_macro, is_inside_let_else, path_to_local_id, span_find_starting_semi};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    Block, Body, Expr, ExprKind, FnDecl, ItemKind, LangItem, MatchSource, OwnerNode, PatKind, QPath, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, GenericArgKind, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::def_id::LocalDefId;
use rustc_span::{BytePos, Pos, Span};
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `let`-bindings, which are subsequently
    /// returned.
    ///
    /// ### Why is this bad?
    /// It is just extraneous code. Remove it to make your code
    /// more rusty.
    ///
    /// ### Known problems
    /// In the case of some temporaries, e.g. locks, eliding the variable binding could lead
    /// to deadlocks. See [this issue](https://github.com/rust-lang/rust/issues/37612).
    /// This could become relevant if the code is later changed to use the code that would have been
    /// bound without first assigning it to a let-binding.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo() -> String {
    ///     let x = String::new();
    ///     x
    /// }
    /// ```
    /// instead, use
    /// ```no_run
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
    /// ```no_run
    /// fn foo(x: usize) -> usize {
    ///     return x;
    /// }
    /// ```
    /// simplify to
    /// ```no_run
    /// fn foo(x: usize) -> usize {
    ///     x
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_RETURN,
    style,
    "using a return statement like `return expr;` where an expression would suffice"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for return statements on `Err` paired with the `?` operator.
    ///
    /// ### Why is this bad?
    /// The `return` is unnecessary.
    ///
    /// ### Example
    /// ```rust,ignore
    /// fn foo(x: usize) -> Result<(), Box<dyn Error>> {
    ///     if x == 0 {
    ///         return Err(...)?;
    ///     }
    ///     Ok(())
    /// }
    /// ```
    /// simplify to
    /// ```rust,ignore
    /// fn foo(x: usize) -> Result<(), Box<dyn Error>> {
    ///     if x == 0 {
    ///         Err(...)?;
    ///     }
    ///     Ok(())
    /// }
    /// ```
    /// if paired with `try_err`, use instead:
    /// ```rust,ignore
    /// fn foo(x: usize) -> Result<(), Box<dyn Error>> {
    ///     if x == 0 {
    ///         return Err(...);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub NEEDLESS_RETURN_WITH_QUESTION_MARK,
    style,
    "using a return statement like `return Err(expr)?;` where removing it would suffice"
}

#[derive(PartialEq, Eq)]
enum RetReplacement<'tcx> {
    Empty,
    Block,
    Unit,
    IfSequence(Cow<'tcx, str>, Applicability),
    Expr(Cow<'tcx, str>, Applicability),
}

impl<'tcx> RetReplacement<'tcx> {
    fn sugg_help(&self) -> &'static str {
        match self {
            Self::Empty | Self::Expr(..) => "remove `return`",
            Self::Block => "replace `return` with an empty block",
            Self::Unit => "replace `return` with a unit value",
            Self::IfSequence(..) => "remove `return` and wrap the sequence with parentheses",
        }
    }

    fn applicability(&self) -> Applicability {
        match self {
            Self::Expr(_, ap) | Self::IfSequence(_, ap) => *ap,
            _ => Applicability::MachineApplicable,
        }
    }
}

impl<'tcx> ToString for RetReplacement<'tcx> {
    fn to_string(&self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Block => "{}".to_string(),
            Self::Unit => "()".to_string(),
            Self::IfSequence(inner, _) => format!("({inner})"),
            Self::Expr(inner, _) => inner.to_string(),
        }
    }
}

declare_lint_pass!(Return => [LET_AND_RETURN, NEEDLESS_RETURN, NEEDLESS_RETURN_WITH_QUESTION_MARK]);

impl<'tcx> LateLintPass<'tcx> for Return {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if !in_external_macro(cx.sess(), stmt.span)
            && let StmtKind::Semi(expr) = stmt.kind
            && let ExprKind::Ret(Some(ret)) = expr.kind
            && let ExprKind::Match(.., MatchSource::TryDesugar(_)) = ret.kind
            // Ensure this is not the final stmt, otherwise removing it would cause a compile error
            && let OwnerNode::Item(item) = cx.tcx.hir().owner(cx.tcx.hir().get_parent_item(expr.hir_id))
            && let ItemKind::Fn(_, _, body) = item.kind
            && let block = cx.tcx.hir().body(body).value
            && let ExprKind::Block(block, _) = block.kind
            && !is_inside_let_else(cx.tcx, expr)
            && let [.., final_stmt] = block.stmts
            && final_stmt.hir_id != stmt.hir_id
            && !is_from_proc_macro(cx, expr)
        {
            span_lint_and_sugg(
                cx,
                NEEDLESS_RETURN_WITH_QUESTION_MARK,
                expr.span.until(ret.span),
                "unneeded `return` statement with `?` operator",
                "remove it",
                String::new(),
                Applicability::MachineApplicable,
            );
        }
    }

    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        // we need both a let-binding stmt and an expr
        if let Some(retexpr) = block.expr
            && let Some(stmt) = block.stmts.iter().last()
            && let StmtKind::Local(local) = &stmt.kind
            && local.ty.is_none()
            && cx.tcx.hir().attrs(local.hir_id).is_empty()
            && let Some(initexpr) = &local.init
            && let PatKind::Binding(_, local_id, _, _) = local.pat.kind
            && path_to_local_id(retexpr, local_id)
            && !last_statement_borrows(cx, initexpr)
            && !in_external_macro(cx.sess(), initexpr.span)
            && !in_external_macro(cx.sess(), retexpr.span)
            && !local.span.from_expansion()
        {
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
                            if !has_enclosing_paren(&snippet) {
                                snippet = format!("({snippet})");
                            }
                            snippet.push_str(" as _");
                        }
                        err.multipart_suggestion(
                            "return the expression directly",
                            vec![(local.span, String::new()), (retexpr.span, snippet)],
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.span_help(initexpr.span, "this expression can be directly returned");
                    }
                },
            );
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        _: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        sp: Span,
        _: LocalDefId,
    ) {
        if sp.from_expansion() {
            return;
        }

        match kind {
            FnKind::Closure => {
                // when returning without value in closure, replace this `return`
                // with an empty block to prevent invalid suggestion (see #6501)
                let replacement = if let ExprKind::Ret(None) = &body.value.kind {
                    RetReplacement::Block
                } else {
                    RetReplacement::Empty
                };
                check_final_expr(cx, body.value, vec![], replacement, None);
            },
            FnKind::ItemFn(..) | FnKind::Method(..) => {
                check_block_return(cx, &body.value.kind, sp, vec![]);
            },
        }
    }
}

// if `expr` is a block, check if there are needless returns in it
fn check_block_return<'tcx>(cx: &LateContext<'tcx>, expr_kind: &ExprKind<'tcx>, sp: Span, mut semi_spans: Vec<Span>) {
    if let ExprKind::Block(block, _) = expr_kind {
        if let Some(block_expr) = block.expr {
            check_final_expr(cx, block_expr, semi_spans, RetReplacement::Empty, None);
        } else if let Some(stmt) = block.stmts.iter().last() {
            match stmt.kind {
                StmtKind::Expr(expr) => {
                    check_final_expr(cx, expr, semi_spans, RetReplacement::Empty, None);
                },
                StmtKind::Semi(semi_expr) => {
                    // Remove ending semicolons and any whitespace ' ' in between.
                    // Without `return`, the suggestion might not compile if the semicolon is retained
                    if let Some(semi_span) = stmt.span.trim_start(semi_expr.span) {
                        let semi_span_to_remove =
                            span_find_starting_semi(cx.sess().source_map(), semi_span.with_hi(sp.hi()));
                        semi_spans.push(semi_span_to_remove);
                    }
                    check_final_expr(cx, semi_expr, semi_spans, RetReplacement::Empty, None);
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
    replacement: RetReplacement<'tcx>,
    match_ty_opt: Option<Ty<'_>>,
) {
    let peeled_drop_expr = expr.peel_drop_temps();
    match &peeled_drop_expr.kind {
        // simple return is always "bad"
        ExprKind::Ret(ref inner) => {
            // check if expr return nothing
            let ret_span = if inner.is_none() && replacement == RetReplacement::Empty {
                extend_span_to_previous_non_ws(cx, peeled_drop_expr.span)
            } else {
                peeled_drop_expr.span
            };

            let replacement = if let Some(inner_expr) = inner {
                // if desugar of `do yeet`, don't lint
                if let ExprKind::Call(path_expr, _) = inner_expr.kind
                    && let ExprKind::Path(QPath::LangItem(LangItem::TryTraitFromYeet, _, _)) = path_expr.kind
                {
                    return;
                }

                let mut applicability = Applicability::MachineApplicable;
                let (snippet, _) = snippet_with_context(cx, inner_expr.span, ret_span.ctxt(), "..", &mut applicability);
                if expr_contains_conjunctive_ifs(inner_expr) {
                    RetReplacement::IfSequence(snippet, applicability)
                } else {
                    RetReplacement::Expr(snippet, applicability)
                }
            } else {
                match match_ty_opt {
                    Some(match_ty) => {
                        match match_ty.kind() {
                            // If the code got till here with
                            // tuple not getting detected before it,
                            // then we are sure it's going to be Unit
                            // type
                            ty::Tuple(_) => RetReplacement::Unit,
                            // We don't want to anything in this case
                            // cause we can't predict what the user would
                            // want here
                            _ => return,
                        }
                    },
                    None => replacement,
                }
            };

            if !cx.tcx.hir().attrs(expr.hir_id).is_empty() {
                return;
            }
            let borrows = inner.map_or(false, |inner| last_statement_borrows(cx, inner));
            if borrows {
                return;
            }

            emit_return_lint(cx, ret_span, semi_spans, &replacement);
        },
        ExprKind::If(_, then, else_clause_opt) => {
            check_block_return(cx, &then.kind, peeled_drop_expr.span, semi_spans.clone());
            if let Some(else_clause) = else_clause_opt {
                check_block_return(cx, &else_clause.kind, peeled_drop_expr.span, semi_spans);
            }
        },
        // a match expr, check all arms
        // an if/if let expr, check both exprs
        // note, if without else is going to be a type checking error anyways
        // (except for unit type functions) so we don't match it
        ExprKind::Match(_, arms, MatchSource::Normal) => {
            let match_ty = cx.typeck_results().expr_ty(peeled_drop_expr);
            for arm in *arms {
                check_final_expr(cx, arm.body, semi_spans.clone(), RetReplacement::Unit, Some(match_ty));
            }
        },
        // if it's a whole block, check it
        other_expr_kind => check_block_return(cx, other_expr_kind, peeled_drop_expr.span, semi_spans),
    }
}

fn expr_contains_conjunctive_ifs<'tcx>(expr: &'tcx Expr<'tcx>) -> bool {
    fn contains_if(expr: &Expr<'_>, on_if: bool) -> bool {
        match expr.kind {
            ExprKind::If(..) => on_if,
            ExprKind::Binary(_, left, right) => contains_if(left, true) || contains_if(right, true),
            _ => false,
        }
    }

    contains_if(expr, false)
}

fn emit_return_lint(cx: &LateContext<'_>, ret_span: Span, semi_spans: Vec<Span>, replacement: &RetReplacement<'_>) {
    if ret_span.from_expansion() {
        return;
    }

    span_lint_and_then(cx, NEEDLESS_RETURN, ret_span, "unneeded `return` statement", |diag| {
        let suggestions = std::iter::once((ret_span, replacement.to_string()))
            .chain(semi_spans.into_iter().map(|span| (span, String::new())))
            .collect();

        diag.multipart_suggestion_verbose(replacement.sugg_help(), suggestions, replacement.applicability());
    });
}

fn last_statement_borrows<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    for_each_expr_with_closures(cx, expr, |e| {
        if let Some(def_id) = fn_def_id(cx, e)
            && cx
                .tcx
                .fn_sig(def_id)
                .instantiate_identity()
                .skip_binder()
                .output()
                .walk()
                .any(|arg| matches!(arg.unpack(), GenericArgKind::Lifetime(re) if !re.is_static()))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(Descend::from(!e.span.from_expansion()))
        }
    })
    .is_some()
}

// Go backwards while encountering whitespace and extend the given Span to that point.
fn extend_span_to_previous_non_ws(cx: &LateContext<'_>, sp: Span) -> Span {
    if let Ok(prev_source) = cx.sess().source_map().span_to_prev_source(sp) {
        let ws = [' ', '\t', '\n'];
        if let Some(non_ws_pos) = prev_source.rfind(|c| !ws.contains(&c)) {
            let len = prev_source.len() - non_ws_pos - 1;
            return sp.with_lo(sp.lo() - BytePos::from_usize(len));
        }
    }

    sp
}
