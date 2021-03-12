mod let_unit_value;
mod utils;

use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, MatchSource, Node, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::hygiene::{ExpnKind, MacroKind};

use if_chain::if_chain;

use crate::utils::diagnostics::{span_lint, span_lint_and_then};
use crate::utils::source::{indent_of, reindent_multiline, snippet_opt};

use utils::{is_unit, is_unit_literal};

declare_clippy_lint! {
    /// **What it does:** Checks for binding a unit value.
    ///
    /// **Why is this bad?** A unit value cannot usefully be used anywhere. So
    /// binding one is kind of pointless.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = {
    ///     1;
    /// };
    /// ```
    pub LET_UNIT_VALUE,
    pedantic,
    "creating a `let` binding to a value of unit type, which usually can't be used afterwards"
}

declare_lint_pass!(UnitTypes => [LET_UNIT_VALUE]);

impl<'tcx> LateLintPass<'tcx> for UnitTypes {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        let_unit_value::check(cx, stmt);
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for comparisons to unit. This includes all binary
    /// comparisons (like `==` and `<`) and asserts.
    ///
    /// **Why is this bad?** Unit is always equal to itself, and thus is just a
    /// clumsily written constant. Mostly this happens when someone accidentally
    /// adds semicolons at the end of the operands.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// # fn baz() {};
    /// if {
    ///     foo();
    /// } == {
    ///     bar();
    /// } {
    ///     baz();
    /// }
    /// ```
    /// is equal to
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// # fn baz() {};
    /// {
    ///     foo();
    ///     bar();
    ///     baz();
    /// }
    /// ```
    ///
    /// For asserts:
    /// ```rust
    /// # fn foo() {};
    /// # fn bar() {};
    /// assert_eq!({ foo(); }, { bar(); });
    /// ```
    /// will always succeed
    pub UNIT_CMP,
    correctness,
    "comparing unit values"
}

declare_lint_pass!(UnitCmp => [UNIT_CMP]);

impl<'tcx> LateLintPass<'tcx> for UnitCmp {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if expr.span.from_expansion() {
            if let Some(callee) = expr.span.source_callee() {
                if let ExpnKind::Macro(MacroKind::Bang, symbol) = callee.kind {
                    if let ExprKind::Binary(ref cmp, ref left, _) = expr.kind {
                        let op = cmp.node;
                        if op.is_comparison() && is_unit(cx.typeck_results().expr_ty(left)) {
                            let result = match &*symbol.as_str() {
                                "assert_eq" | "debug_assert_eq" => "succeed",
                                "assert_ne" | "debug_assert_ne" => "fail",
                                _ => return,
                            };
                            span_lint(
                                cx,
                                UNIT_CMP,
                                expr.span,
                                &format!(
                                    "`{}` of unit values detected. This will always {}",
                                    symbol.as_str(),
                                    result
                                ),
                            );
                        }
                    }
                }
            }
            return;
        }
        if let ExprKind::Binary(ref cmp, ref left, _) = expr.kind {
            let op = cmp.node;
            if op.is_comparison() && is_unit(cx.typeck_results().expr_ty(left)) {
                let result = match op {
                    BinOpKind::Eq | BinOpKind::Le | BinOpKind::Ge => "true",
                    _ => "false",
                };
                span_lint(
                    cx,
                    UNIT_CMP,
                    expr.span,
                    &format!(
                        "{}-comparison of unit values detected. This will always be {}",
                        op.as_str(),
                        result
                    ),
                );
            }
        }
    }
}

declare_clippy_lint! {
    /// **What it does:** Checks for passing a unit value as an argument to a function without using a
    /// unit literal (`()`).
    ///
    /// **Why is this bad?** This is likely the result of an accidental semicolon.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// foo({
    ///     let a = bar();
    ///     baz(a);
    /// })
    /// ```
    pub UNIT_ARG,
    complexity,
    "passing unit to a function"
}

declare_lint_pass!(UnitArg => [UNIT_ARG]);

impl<'tcx> LateLintPass<'tcx> for UnitArg {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        // apparently stuff in the desugaring of `?` can trigger this
        // so check for that here
        // only the calls to `Try::from_error` is marked as desugared,
        // so we need to check both the current Expr and its parent.
        if is_questionmark_desugar_marked_call(expr) {
            return;
        }
        if_chain! {
            let map = &cx.tcx.hir();
            let opt_parent_node = map.find(map.get_parent_node(expr.hir_id));
            if let Some(hir::Node::Expr(parent_expr)) = opt_parent_node;
            if is_questionmark_desugar_marked_call(parent_expr);
            then {
                return;
            }
        }

        match expr.kind {
            ExprKind::Call(_, args) | ExprKind::MethodCall(_, _, args, _) => {
                let args_to_recover = args
                    .iter()
                    .filter(|arg| {
                        if is_unit(cx.typeck_results().expr_ty(arg)) && !is_unit_literal(arg) {
                            !matches!(
                                &arg.kind,
                                ExprKind::Match(.., MatchSource::TryDesugar) | ExprKind::Path(..)
                            )
                        } else {
                            false
                        }
                    })
                    .collect::<Vec<_>>();
                if !args_to_recover.is_empty() {
                    lint_unit_args(cx, expr, &args_to_recover);
                }
            },
            _ => (),
        }
    }
}

fn is_questionmark_desugar_marked_call(expr: &Expr<'_>) -> bool {
    use rustc_span::hygiene::DesugaringKind;
    if let ExprKind::Call(ref callee, _) = expr.kind {
        callee.span.is_desugaring(DesugaringKind::QuestionMark)
    } else {
        false
    }
}

fn lint_unit_args(cx: &LateContext<'_>, expr: &Expr<'_>, args_to_recover: &[&Expr<'_>]) {
    let mut applicability = Applicability::MachineApplicable;
    let (singular, plural) = if args_to_recover.len() > 1 {
        ("", "s")
    } else {
        ("a ", "")
    };
    span_lint_and_then(
        cx,
        UNIT_ARG,
        expr.span,
        &format!("passing {}unit value{} to a function", singular, plural),
        |db| {
            let mut or = "";
            args_to_recover
                .iter()
                .filter_map(|arg| {
                    if_chain! {
                        if let ExprKind::Block(block, _) = arg.kind;
                        if block.expr.is_none();
                        if let Some(last_stmt) = block.stmts.iter().last();
                        if let StmtKind::Semi(last_expr) = last_stmt.kind;
                        if let Some(snip) = snippet_opt(cx, last_expr.span);
                        then {
                            Some((
                                last_stmt.span,
                                snip,
                            ))
                        }
                        else {
                            None
                        }
                    }
                })
                .for_each(|(span, sugg)| {
                    db.span_suggestion(
                        span,
                        "remove the semicolon from the last statement in the block",
                        sugg,
                        Applicability::MaybeIncorrect,
                    );
                    or = "or ";
                    applicability = Applicability::MaybeIncorrect;
                });

            let arg_snippets: Vec<String> = args_to_recover
                .iter()
                .filter_map(|arg| snippet_opt(cx, arg.span))
                .collect();
            let arg_snippets_without_empty_blocks: Vec<String> = args_to_recover
                .iter()
                .filter(|arg| !is_empty_block(arg))
                .filter_map(|arg| snippet_opt(cx, arg.span))
                .collect();

            if let Some(call_snippet) = snippet_opt(cx, expr.span) {
                let sugg = fmt_stmts_and_call(
                    cx,
                    expr,
                    &call_snippet,
                    &arg_snippets,
                    &arg_snippets_without_empty_blocks,
                );

                if arg_snippets_without_empty_blocks.is_empty() {
                    db.multipart_suggestion(
                        &format!("use {}unit literal{} instead", singular, plural),
                        args_to_recover
                            .iter()
                            .map(|arg| (arg.span, "()".to_string()))
                            .collect::<Vec<_>>(),
                        applicability,
                    );
                } else {
                    let plural = arg_snippets_without_empty_blocks.len() > 1;
                    let empty_or_s = if plural { "s" } else { "" };
                    let it_or_them = if plural { "them" } else { "it" };
                    db.span_suggestion(
                        expr.span,
                        &format!(
                            "{}move the expression{} in front of the call and replace {} with the unit literal `()`",
                            or, empty_or_s, it_or_them
                        ),
                        sugg,
                        applicability,
                    );
                }
            }
        },
    );
}

fn is_empty_block(expr: &Expr<'_>) -> bool {
    matches!(
        expr.kind,
        ExprKind::Block(
            Block {
                stmts: &[],
                expr: None,
                ..
            },
            _,
        )
    )
}

fn fmt_stmts_and_call(
    cx: &LateContext<'_>,
    call_expr: &Expr<'_>,
    call_snippet: &str,
    args_snippets: &[impl AsRef<str>],
    non_empty_block_args_snippets: &[impl AsRef<str>],
) -> String {
    let call_expr_indent = indent_of(cx, call_expr.span).unwrap_or(0);
    let call_snippet_with_replacements = args_snippets
        .iter()
        .fold(call_snippet.to_owned(), |acc, arg| acc.replacen(arg.as_ref(), "()", 1));

    let mut stmts_and_call = non_empty_block_args_snippets
        .iter()
        .map(|it| it.as_ref().to_owned())
        .collect::<Vec<_>>();
    stmts_and_call.push(call_snippet_with_replacements);
    stmts_and_call = stmts_and_call
        .into_iter()
        .map(|v| reindent_multiline(v.into(), true, Some(call_expr_indent)).into_owned())
        .collect();

    let mut stmts_and_call_snippet = stmts_and_call.join(&format!("{}{}", ";\n", " ".repeat(call_expr_indent)));
    // expr is not in a block statement or result expression position, wrap in a block
    let parent_node = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(call_expr.hir_id));
    if !matches!(parent_node, Some(Node::Block(_))) && !matches!(parent_node, Some(Node::Stmt(_))) {
        let block_indent = call_expr_indent + 4;
        stmts_and_call_snippet =
            reindent_multiline(stmts_and_call_snippet.into(), true, Some(block_indent)).into_owned();
        stmts_and_call_snippet = format!(
            "{{\n{}{}\n{}}}",
            " ".repeat(block_indent),
            &stmts_and_call_snippet,
            " ".repeat(call_expr_indent)
        );
    }
    stmts_and_call_snippet
}
