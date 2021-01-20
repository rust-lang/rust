use crate::utils;
use crate::utils::eager_or_lazy;
use crate::utils::sugg::Sugg;
use crate::utils::{is_type_diagnostic_item, paths, span_lint_and_sugg};
use if_chain::if_chain;

use rustc_errors::Applicability;
use rustc_hir::{Arm, BindingAnnotation, Block, Expr, ExprKind, MatchSource, Mutability, PatKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:**
    /// Lints usage of `if let Some(v) = ... { y } else { x }` which is more
    /// idiomatically done with `Option::map_or` (if the else bit is a pure
    /// expression) or `Option::map_or_else` (if the else bit is an impure
    /// expression).
    ///
    /// **Why is this bad?**
    /// Using the dedicated functions of the Option type is clearer and
    /// more concise than an `if let` expression.
    ///
    /// **Known problems:**
    /// This lint uses a deliberately conservative metric for checking
    /// if the inside of either body contains breaks or continues which will
    /// cause it to not suggest a fix if either block contains a loop with
    /// continues or breaks contained within the loop.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// # let optional: Option<u32> = Some(0);
    /// # fn do_complicated_function() -> u32 { 5 };
    /// let _ = if let Some(foo) = optional {
    ///     foo
    /// } else {
    ///     5
    /// };
    /// let _ = if let Some(foo) = optional {
    ///     foo
    /// } else {
    ///     let y = do_complicated_function();
    ///     y*y
    /// };
    /// ```
    ///
    /// should be
    ///
    /// ```rust
    /// # let optional: Option<u32> = Some(0);
    /// # fn do_complicated_function() -> u32 { 5 };
    /// let _ = optional.map_or(5, |foo| foo);
    /// let _ = optional.map_or_else(||{
    ///     let y = do_complicated_function();
    ///     y*y
    /// }, |foo| foo);
    /// ```
    pub OPTION_IF_LET_ELSE,
    pedantic,
    "reimplementation of Option::map_or"
}

declare_lint_pass!(OptionIfLetElse => [OPTION_IF_LET_ELSE]);

/// Returns true iff the given expression is the result of calling `Result::ok`
fn is_result_ok(cx: &LateContext<'_>, expr: &'_ Expr<'_>) -> bool {
    if let ExprKind::MethodCall(ref path, _, &[ref receiver], _) = &expr.kind {
        path.ident.name.to_ident_string() == "ok"
            && is_type_diagnostic_item(cx, &cx.typeck_results().expr_ty(&receiver), sym::result_type)
    } else {
        false
    }
}

/// A struct containing information about occurrences of the
/// `if let Some(..) = .. else` construct that this lint detects.
struct OptionIfLetElseOccurence {
    option: String,
    method_sugg: String,
    some_expr: String,
    none_expr: String,
    wrap_braces: bool,
}

/// Extracts the body of a given arm. If the arm contains only an expression,
/// then it returns the expression. Otherwise, it returns the entire block
fn extract_body_from_arm<'a>(arm: &'a Arm<'a>) -> Option<&'a Expr<'a>> {
    if let ExprKind::Block(
        Block {
            stmts: statements,
            expr: Some(expr),
            ..
        },
        _,
    ) = &arm.body.kind
    {
        if let [] = statements {
            Some(&expr)
        } else {
            Some(&arm.body)
        }
    } else {
        None
    }
}

/// If this is the else body of an if/else expression, then we need to wrap
/// it in curly braces. Otherwise, we don't.
fn should_wrap_in_braces(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    utils::get_enclosing_block(cx, expr.hir_id).map_or(false, |parent| {
        if let Some(Expr {
            kind:
                ExprKind::Match(
                    _,
                    arms,
                    MatchSource::IfDesugar {
                        contains_else_clause: true,
                    }
                    | MatchSource::IfLetDesugar {
                        contains_else_clause: true,
                    },
                ),
            ..
        }) = parent.expr
        {
            expr.hir_id == arms[1].body.hir_id
        } else {
            false
        }
    })
}

fn format_option_in_sugg(cx: &LateContext<'_>, cond_expr: &Expr<'_>, as_ref: bool, as_mut: bool) -> String {
    format!(
        "{}{}",
        Sugg::hir(cx, cond_expr, "..").maybe_par(),
        if as_mut {
            ".as_mut()"
        } else if as_ref {
            ".as_ref()"
        } else {
            ""
        }
    )
}

/// If this expression is the option if let/else construct we're detecting, then
/// this function returns an `OptionIfLetElseOccurence` struct with details if
/// this construct is found, or None if this construct is not found.
fn detect_option_if_let_else<'tcx>(
    cx: &'_ LateContext<'tcx>,
    expr: &'_ Expr<'tcx>,
) -> Option<OptionIfLetElseOccurence> {
    if_chain! {
        if !utils::in_macro(expr.span); // Don't lint macros, because it behaves weirdly
        if let ExprKind::Match(cond_expr, arms, MatchSource::IfLetDesugar{contains_else_clause: true}) = &expr.kind;
        if arms.len() == 2;
        if !is_result_ok(cx, cond_expr); // Don't lint on Result::ok because a different lint does it already
        if let PatKind::TupleStruct(struct_qpath, &[inner_pat], _) = &arms[0].pat.kind;
        if utils::match_qpath(struct_qpath, &paths::OPTION_SOME);
        if let PatKind::Binding(bind_annotation, _, id, _) = &inner_pat.kind;
        if !utils::usage::contains_return_break_continue_macro(arms[0].body);
        if !utils::usage::contains_return_break_continue_macro(arms[1].body);
        then {
            let capture_mut = if bind_annotation == &BindingAnnotation::Mutable { "mut " } else { "" };
            let some_body = extract_body_from_arm(&arms[0])?;
            let none_body = extract_body_from_arm(&arms[1])?;
            let method_sugg = if eager_or_lazy::is_eagerness_candidate(cx, none_body) { "map_or" } else { "map_or_else" };
            let capture_name = id.name.to_ident_string();
            let wrap_braces = should_wrap_in_braces(cx, expr);
            let (as_ref, as_mut) = match &cond_expr.kind {
                ExprKind::AddrOf(_, Mutability::Not, _) => (true, false),
                ExprKind::AddrOf(_, Mutability::Mut, _) => (false, true),
                _ => (bind_annotation == &BindingAnnotation::Ref, bind_annotation == &BindingAnnotation::RefMut),
            };
            let cond_expr = match &cond_expr.kind {
                // Pointer dereferencing happens automatically, so we can omit it in the suggestion
                ExprKind::Unary(UnOp::UnDeref, expr) | ExprKind::AddrOf(_, _, expr) => expr,
                _ => cond_expr,
            };
            Some(OptionIfLetElseOccurence {
                option: format_option_in_sugg(cx, cond_expr, as_ref, as_mut),
                method_sugg: method_sugg.to_string(),
                some_expr: format!("|{}{}| {}", capture_mut, capture_name, Sugg::hir(cx, some_body, "..")),
                none_expr: format!("{}{}", if method_sugg == "map_or" { "" } else { "|| " }, Sugg::hir(cx, none_body, "..")),
                wrap_braces,
            })
        } else {
            None
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for OptionIfLetElse {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let Some(detection) = detect_option_if_let_else(cx, expr) {
            span_lint_and_sugg(
                cx,
                OPTION_IF_LET_ELSE,
                expr.span,
                format!("use Option::{} instead of an if let/else", detection.method_sugg).as_str(),
                "try",
                format!(
                    "{}{}.{}({}, {}){}",
                    if detection.wrap_braces { "{ " } else { "" },
                    detection.option,
                    detection.method_sugg,
                    detection.none_expr,
                    detection.some_expr,
                    if detection.wrap_braces { " }" } else { "" },
                ),
                Applicability::MaybeIncorrect,
            );
        }
    }
}
