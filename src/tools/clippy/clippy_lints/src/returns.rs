use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::source::{SpanRangeExt, snippet_with_context};
use clippy_utils::sugg::has_enclosing_paren;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{
    binary_expr_needs_parentheses, fn_def_id, is_from_proc_macro, is_inside_let_else, is_res_lang_ctor,
    leaks_droppable_temporary_with_limited_lifetime, path_res, path_to_local_id, span_contains_cfg,
    span_find_starting_semi, sym,
};
use core::ops::ControlFlow;
use rustc_ast::MetaItemInner;
use rustc_errors::Applicability;
use rustc_hir::LangItem::ResultErr;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    Block, Body, Expr, ExprKind, FnDecl, HirId, ItemKind, LangItem, MatchSource, Node, OwnerNode, PatKind, QPath, Stmt,
    StmtKind,
};
use rustc_lint::{LateContext, LateLintPass, Level, LintContext};
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::{self, GenericArgKind, Ty};
use rustc_session::declare_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::edition::Edition;
use rustc_span::{BytePos, Pos, Span};
use std::borrow::Cow;
use std::fmt::Display;

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
    // This lint requires some special handling in `check_final_expr` for `#[expect]`.
    // This handling needs to be updated if the group gets changed. This should also
    // be caught by tests.
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
    /// Returns may be used to add attributes to the return expression. Return
    /// statements with attributes are therefore be accepted by this lint.
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
    NeedsPar(Cow<'tcx, str>, Applicability),
    Expr(Cow<'tcx, str>, Applicability),
}

impl RetReplacement<'_> {
    fn sugg_help(&self) -> &'static str {
        match self {
            Self::Empty | Self::Expr(..) => "remove `return`",
            Self::Block => "replace `return` with an empty block",
            Self::Unit => "replace `return` with a unit value",
            Self::NeedsPar(..) => "remove `return` and wrap the sequence with parentheses",
        }
    }

    fn applicability(&self) -> Applicability {
        match self {
            Self::Expr(_, ap) | Self::NeedsPar(_, ap) => *ap,
            _ => Applicability::MachineApplicable,
        }
    }
}

impl Display for RetReplacement<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, ""),
            Self::Block => write!(f, "{{}}"),
            Self::Unit => write!(f, "()"),
            Self::NeedsPar(inner, _) => write!(f, "({inner})"),
            Self::Expr(inner, _) => write!(f, "{inner}"),
        }
    }
}

declare_lint_pass!(Return => [LET_AND_RETURN, NEEDLESS_RETURN, NEEDLESS_RETURN_WITH_QUESTION_MARK]);

/// Checks if a return statement is "needed" in the middle of a block, or if it can be removed. This
/// is the case when the enclosing block expression is coerced to some other type, which only works
/// because of the never-ness of `return` expressions
fn stmt_needs_never_type(cx: &LateContext<'_>, stmt_hir_id: HirId) -> bool {
    cx.tcx
        .hir_parent_iter(stmt_hir_id)
        .find_map(|(_, node)| if let Node::Expr(expr) = node { Some(expr) } else { None })
        .is_some_and(|e| {
            cx.typeck_results()
                .expr_adjustments(e)
                .iter()
                .any(|adjust| adjust.target != cx.tcx.types.unit && matches!(adjust.kind, Adjust::NeverToAny))
        })
}

impl<'tcx> LateLintPass<'tcx> for Return {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if !stmt.span.in_external_macro(cx.sess().source_map())
            && let StmtKind::Semi(expr) = stmt.kind
            && let ExprKind::Ret(Some(ret)) = expr.kind
            // return Err(...)? desugars to a match
            // over a Err(...).branch()
            // which breaks down to a branch call, with the callee being
            // the constructor of the Err variant
            && let ExprKind::Match(maybe_cons, _, MatchSource::TryDesugar(_)) = ret.kind
            && let ExprKind::Call(_, [maybe_result_err]) = maybe_cons.kind
            && let ExprKind::Call(maybe_constr, _) = maybe_result_err.kind
            && is_res_lang_ctor(cx, path_res(cx, maybe_constr), ResultErr)

            // Ensure this is not the final stmt, otherwise removing it would cause a compile error
            && let OwnerNode::Item(item) = cx.tcx.hir_owner_node(cx.tcx.hir_get_parent_item(expr.hir_id))
            && let ItemKind::Fn { body, .. } = item.kind
            && let block = cx.tcx.hir_body(body).value
            && let ExprKind::Block(block, _) = block.kind
            && !is_inside_let_else(cx.tcx, expr)
            && let [.., final_stmt] = block.stmts
            && final_stmt.hir_id != stmt.hir_id
            && !is_from_proc_macro(cx, expr)
            && !stmt_needs_never_type(cx, stmt.hir_id)
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
            && let StmtKind::Let(local) = &stmt.kind
            && local.ty.is_none()
            && cx.tcx.hir_attrs(local.hir_id).is_empty()
            && let Some(initexpr) = &local.init
            && let PatKind::Binding(_, local_id, _, _) = local.pat.kind
            && path_to_local_id(retexpr, local_id)
            && (cx.sess().edition() >= Edition::Edition2024 || !last_statement_borrows(cx, initexpr))
            && !initexpr.span.in_external_macro(cx.sess().source_map())
            && !retexpr.span.in_external_macro(cx.sess().source_map())
            && !local.span.from_expansion()
            && !span_contains_cfg(cx, stmt.span.between(retexpr.span))
        {
            span_lint_hir_and_then(
                cx,
                LET_AND_RETURN,
                retexpr.hir_id,
                retexpr.span,
                "returning the result of a `let` binding from a block",
                |err| {
                    err.span_label(local.span, "unnecessary `let` binding");

                    if let Some(src) = initexpr.span.get_source_text(cx) {
                        let sugg = if binary_expr_needs_parentheses(initexpr) {
                            if has_enclosing_paren(&src) {
                                src.to_owned()
                            } else {
                                format!("({src})")
                            }
                        } else if !cx.typeck_results().expr_adjustments(retexpr).is_empty() {
                            if has_enclosing_paren(&src) {
                                format!("{src} as _")
                            } else {
                                format!("({src}) as _")
                            }
                        } else {
                            src.to_owned()
                        };
                        err.multipart_suggestion(
                            "return the expression directly",
                            vec![(local.span, String::new()), (retexpr.span, sugg)],
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
        ExprKind::Ret(inner) => {
            // check if expr return nothing
            let ret_span = if inner.is_none() && replacement == RetReplacement::Empty {
                extend_span_to_previous_non_ws(cx, peeled_drop_expr.span)
            } else {
                peeled_drop_expr.span
            };

            let replacement = if let Some(inner_expr) = inner {
                // if desugar of `do yeet`, don't lint
                if let ExprKind::Call(path_expr, [_]) = inner_expr.kind
                    && let ExprKind::Path(QPath::LangItem(LangItem::TryTraitFromYeet, ..)) = path_expr.kind
                {
                    return;
                }

                let mut applicability = Applicability::MachineApplicable;
                let (snippet, _) = snippet_with_context(cx, inner_expr.span, ret_span.ctxt(), "..", &mut applicability);
                if binary_expr_needs_parentheses(inner_expr) {
                    RetReplacement::NeedsPar(snippet, applicability)
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

            if inner.is_some_and(|inner| leaks_droppable_temporary_with_limited_lifetime(cx, inner)) {
                return;
            }

            if ret_span.from_expansion() || is_from_proc_macro(cx, expr) {
                return;
            }

            // Returns may be used to turn an expression into a statement in rustc's AST.
            // This allows the addition of attributes, like `#[allow]` (See: clippy#9361)
            // `#[expect(clippy::needless_return)]` needs to be handled separately to
            // actually fulfill the expectation (clippy::#12998)
            match cx.tcx.hir_attrs(expr.hir_id) {
                [] => {},
                [attr] => {
                    if matches!(Level::from_attr(attr), Some((Level::Expect, _)))
                        && let metas = attr.meta_item_list()
                        && let Some(lst) = metas
                        && let [MetaItemInner::MetaItem(meta_item), ..] = lst.as_slice()
                        && let [tool, lint_name] = meta_item.path.segments.as_slice()
                        && tool.ident.name == sym::clippy
                        && matches!(
                            lint_name.ident.name,
                            sym::needless_return | sym::style | sym::all | sym::warnings
                        )
                    {
                        // This is an expectation of the `needless_return` lint
                    } else {
                        return;
                    }
                },
                _ => return,
            }

            emit_return_lint(cx, ret_span, semi_spans, &replacement, expr.hir_id);
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

fn emit_return_lint(
    cx: &LateContext<'_>,
    ret_span: Span,
    semi_spans: Vec<Span>,
    replacement: &RetReplacement<'_>,
    at: HirId,
) {
    span_lint_hir_and_then(
        cx,
        NEEDLESS_RETURN,
        at,
        ret_span,
        "unneeded `return` statement",
        |diag| {
            let suggestions = std::iter::once((ret_span, replacement.to_string()))
                .chain(semi_spans.into_iter().map(|span| (span, String::new())))
                .collect();

            diag.multipart_suggestion_verbose(replacement.sugg_help(), suggestions, replacement.applicability());
        },
    );
}

fn last_statement_borrows<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    for_each_expr(cx, expr, |e| {
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
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

// Go backwards while encountering whitespace and extend the given Span to that point.
fn extend_span_to_previous_non_ws(cx: &LateContext<'_>, sp: Span) -> Span {
    if let Ok(prev_source) = cx.sess().source_map().span_to_prev_source(sp) {
        let ws = [b' ', b'\t', b'\n'];
        if let Some(non_ws_pos) = prev_source.bytes().rposition(|c| !ws.contains(&c)) {
            let len = prev_source.len() - non_ws_pos - 1;
            return sp.with_lo(sp.lo() - BytePos::from_usize(len));
        }
    }

    sp
}
