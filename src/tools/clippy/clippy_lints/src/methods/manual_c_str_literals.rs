use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet;
use clippy_utils::{get_parent_expr, sym};
use rustc_ast::{LitKind, StrStyle};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::edition::Edition::Edition2021;
use rustc_span::{Span, Symbol};

use super::MANUAL_C_STR_LITERALS;

/// Checks:
/// - `b"...".as_ptr()`
/// - `b"...".as_ptr().cast()`
/// - `"...".as_ptr()`
/// - `"...".as_ptr().cast()`
///
/// Iff the parent call of `.cast()` isn't `CStr::from_ptr`, to avoid linting twice.
pub(super) fn check_as_ptr<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    receiver: &'tcx Expr<'tcx>,
    msrv: Msrv,
) {
    if let ExprKind::Lit(lit) = receiver.kind
        && let LitKind::ByteStr(_, StrStyle::Cooked) | LitKind::Str(_, StrStyle::Cooked) = lit.node
        && cx.tcx.sess.edition() >= Edition2021
        && let casts_removed = peel_ptr_cast_ancestors(cx, expr)
        && !get_parent_expr(cx, casts_removed).is_some_and(
            |parent| matches!(parent.kind, ExprKind::Call(func, _) if is_c_str_function(cx, func).is_some()),
        )
        && let Some(sugg) = rewrite_as_cstr(cx, lit.span)
        && msrv.meets(cx, msrvs::C_STR_LITERALS)
    {
        span_lint_and_sugg(
            cx,
            MANUAL_C_STR_LITERALS,
            receiver.span,
            "manually constructing a nul-terminated string",
            r#"use a `c""` literal"#,
            sugg,
            // an additional cast may be needed, since the type of `CStr::as_ptr` and
            // `"".as_ptr()` can differ and is platform dependent
            Applicability::HasPlaceholders,
        );
    }
}

/// Checks if the callee is a "relevant" `CStr` function considered by this lint.
/// Returns the function name.
fn is_c_str_function(cx: &LateContext<'_>, func: &Expr<'_>) -> Option<Symbol> {
    if let ExprKind::Path(QPath::TypeRelative(cstr, fn_name)) = &func.kind
        && let TyKind::Path(QPath::Resolved(_, ty_path)) = &cstr.kind
        && cx.tcx.lang_items().c_str() == ty_path.res.opt_def_id()
    {
        Some(fn_name.ident.name)
    } else {
        None
    }
}

/// Checks calls to the `CStr` constructor functions:
/// - `CStr::from_bytes_with_nul(..)`
/// - `CStr::from_bytes_with_nul_unchecked(..)`
/// - `CStr::from_ptr(..)`
pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, func: &Expr<'_>, args: &[Expr<'_>], msrv: Msrv) {
    if let Some(fn_name) = is_c_str_function(cx, func)
        && let [arg] = args
        && cx.tcx.sess.edition() >= Edition2021
        && msrv.meets(cx, msrvs::C_STR_LITERALS)
    {
        match fn_name {
            sym::from_bytes_with_nul | sym::from_bytes_with_nul_unchecked
                if !arg.span.from_expansion()
                    && let ExprKind::Lit(lit) = arg.kind
                    && let LitKind::ByteStr(_, StrStyle::Cooked) | LitKind::Str(_, StrStyle::Cooked) = lit.node =>
            {
                check_from_bytes(cx, expr, arg, fn_name);
            },
            sym::from_ptr => check_from_ptr(cx, expr, arg),
            _ => {},
        }
    }
}

/// Checks `CStr::from_ptr(b"foo\0".as_ptr().cast())`
fn check_from_ptr(cx: &LateContext<'_>, expr: &Expr<'_>, arg: &Expr<'_>) {
    if let ExprKind::MethodCall(method, lit, [], _) = peel_ptr_cast(arg).kind
        && method.ident.name == sym::as_ptr
        && !lit.span.from_expansion()
        && let ExprKind::Lit(lit) = lit.kind
        && let LitKind::ByteStr(_, StrStyle::Cooked) = lit.node
        && let Some(sugg) = rewrite_as_cstr(cx, lit.span)
    {
        span_lint_and_sugg(
            cx,
            MANUAL_C_STR_LITERALS,
            expr.span,
            "calling `CStr::from_ptr` with a byte string literal",
            r#"use a `c""` literal"#,
            sugg,
            Applicability::MachineApplicable,
        );
    }
}
/// Checks `CStr::from_bytes_with_nul(b"foo\0")`
fn check_from_bytes(cx: &LateContext<'_>, expr: &Expr<'_>, arg: &Expr<'_>, method: Symbol) {
    let (span, applicability) = if let Some(parent) = get_parent_expr(cx, expr)
        && let ExprKind::MethodCall(method, ..) = parent.kind
        && [sym::unwrap, sym::expect].contains(&method.ident.name)
    {
        (parent.span, Applicability::MachineApplicable)
    } else if method == sym::from_bytes_with_nul_unchecked {
        // `*_unchecked` returns `&CStr` directly, nothing needs to be changed
        (expr.span, Applicability::MachineApplicable)
    } else {
        // User needs to remove error handling, can't be machine applicable
        (expr.span, Applicability::HasPlaceholders)
    };

    let Some(sugg) = rewrite_as_cstr(cx, arg.span) else {
        return;
    };

    span_lint_and_sugg(
        cx,
        MANUAL_C_STR_LITERALS,
        span,
        "calling `CStr::new` with a byte string literal",
        r#"use a `c""` literal"#,
        sugg,
        applicability,
    );
}

/// Rewrites a byte string literal to a c-str literal.
/// `b"foo\0"` -> `c"foo"`
///
/// Returns `None` if it doesn't end in a NUL byte.
fn rewrite_as_cstr(cx: &LateContext<'_>, span: Span) -> Option<String> {
    let mut sugg = String::from("c") + snippet(cx, span.source_callsite(), "..").trim_start_matches('b');

    // NUL byte should always be right before the closing quote.
    if let Some(quote_pos) = sugg.rfind('"') {
        // Possible values right before the quote:
        // - literal NUL value
        if sugg.as_bytes()[quote_pos - 1] == b'\0' {
            sugg.remove(quote_pos - 1);
        }
        // - \x00
        else if sugg[..quote_pos].ends_with("\\x00") {
            sugg.replace_range(quote_pos - 4..quote_pos, "");
        }
        // - \0
        else if sugg[..quote_pos].ends_with("\\0") {
            sugg.replace_range(quote_pos - 2..quote_pos, "");
        }
        // No known suffix, so assume it's not a C-string.
        else {
            return None;
        }
    }

    Some(sugg)
}

fn get_cast_target<'tcx>(e: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    match &e.kind {
        ExprKind::MethodCall(method, receiver, [], _) if method.ident.as_str() == "cast" => Some(receiver),
        ExprKind::Cast(expr, _) => Some(expr),
        _ => None,
    }
}

/// `x.cast()` -> `x`
/// `x as *const _` -> `x`
/// `x` -> `x` (returns the same expression for non-cast exprs)
fn peel_ptr_cast<'tcx>(e: &'tcx Expr<'tcx>) -> &'tcx Expr<'tcx> {
    get_cast_target(e).map_or(e, peel_ptr_cast)
}

/// Same as `peel_ptr_cast`, but the other way around, by walking up the ancestor cast expressions:
///
/// `foo(x.cast() as *const _)`
///      ^ given this `x` expression, returns the `foo(...)` expression
fn peel_ptr_cast_ancestors<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'tcx>) -> &'tcx Expr<'tcx> {
    let mut prev = e;
    for (_, node) in cx.tcx.hir_parent_iter(e.hir_id) {
        if let Node::Expr(e) = node
            && get_cast_target(e).is_some()
        {
            prev = e;
        } else {
            break;
        }
    }
    prev
}
