use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{is_type_diagnostic_item, is_type_lang_item, match_type};
use clippy_utils::{is_expr_path_def_path, paths};
use if_chain::if_chain;
use rustc_ast::util::parser::PREC_POSTFIX;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::LateContext;
use rustc_span::symbol::{sym, Symbol};

use super::MANUAL_STR_REPEAT;

enum RepeatKind {
    Str,
    String,
    Char,
}

fn parse_repeat_arg(cx: &LateContext<'_>, e: &Expr<'_>) -> Option<RepeatKind> {
    if let ExprKind::Lit(lit) = &e.kind {
        match lit.node {
            LitKind::Str(..) => Some(RepeatKind::Str),
            LitKind::Char(_) => Some(RepeatKind::Char),
            _ => None,
        }
    } else {
        let ty = cx.typeck_results().expr_ty(e);
        if is_type_diagnostic_item(cx, ty, sym::string_type)
            || is_type_lang_item(cx, ty, LangItem::OwnedBox)
            || match_type(cx, ty, &paths::COW)
        {
            Some(RepeatKind::String)
        } else {
            let ty = ty.peel_refs();
            (ty.is_str() || is_type_diagnostic_item(cx, ty, sym::string_type)).then(|| RepeatKind::Str)
        }
    }
}

pub(super) fn check(
    cx: &LateContext<'_>,
    collect_expr: &Expr<'_>,
    take_expr: &Expr<'_>,
    take_self_arg: &Expr<'_>,
    take_arg: &Expr<'_>,
) {
    if_chain! {
        if let ExprKind::Call(repeat_fn, [repeat_arg]) = take_self_arg.kind;
        if is_expr_path_def_path(cx, repeat_fn, &paths::ITER_REPEAT);
        if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(collect_expr), sym::string_type);
        if let Some(collect_id) = cx.typeck_results().type_dependent_def_id(collect_expr.hir_id);
        if let Some(take_id) = cx.typeck_results().type_dependent_def_id(take_expr.hir_id);
        if let Some(iter_trait_id) = cx.tcx.get_diagnostic_item(sym::Iterator);
        if cx.tcx.trait_of_item(collect_id) == Some(iter_trait_id);
        if cx.tcx.trait_of_item(take_id) == Some(iter_trait_id);
        if let Some(repeat_kind) = parse_repeat_arg(cx, repeat_arg);
        let ctxt = collect_expr.span.ctxt();
        if ctxt == take_expr.span.ctxt();
        if ctxt == take_self_arg.span.ctxt();
        then {
            let mut app = Applicability::MachineApplicable;
            let (val_snip, val_is_mac) = snippet_with_context(cx, repeat_arg.span, ctxt, "..", &mut app);
            let count_snip = snippet_with_context(cx, take_arg.span, ctxt, "..", &mut app).0;

            let val_str = match repeat_kind {
                RepeatKind::String => format!("(&{})", val_snip),
                RepeatKind::Str if !val_is_mac && repeat_arg.precedence().order() < PREC_POSTFIX => {
                    format!("({})", val_snip)
                },
                RepeatKind::Str => val_snip.into(),
                RepeatKind::Char if val_snip == r#"'"'"# => r#""\"""#.into(),
                RepeatKind::Char if val_snip == r#"'\''"# => r#""'""#.into(),
                RepeatKind::Char => format!("\"{}\"", &val_snip[1..val_snip.len() - 1]),
            };

            span_lint_and_sugg(
                cx,
                MANUAL_STR_REPEAT,
                collect_expr.span,
                "manual implementation of `str::repeat` using iterators",
                "try this",
                format!("{}.repeat({})", val_str, count_snip),
                app
            )
        }
    }
}
