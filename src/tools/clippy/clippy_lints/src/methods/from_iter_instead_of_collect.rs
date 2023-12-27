use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::implements_trait;
use clippy_utils::{is_path_diagnostic_item, sugg};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::sym;

use super::FROM_ITER_INSTEAD_OF_COLLECT;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>], func: &hir::Expr<'_>) {
    if is_path_diagnostic_item(cx, func, sym::from_iter_fn)
        && let ty = cx.typeck_results().expr_ty(expr)
        && let arg_ty = cx.typeck_results().expr_ty(&args[0])
        && let Some(iter_id) = cx.tcx.get_diagnostic_item(sym::Iterator)
        && implements_trait(cx, arg_ty, iter_id, &[])
    {
        // `expr` implements `FromIterator` trait
        let iter_expr = sugg::Sugg::hir(cx, &args[0], "..").maybe_par();
        let turbofish = extract_turbofish(cx, expr, ty);
        let sugg = format!("{iter_expr}.collect::<{turbofish}>()");
        span_lint_and_sugg(
            cx,
            FROM_ITER_INSTEAD_OF_COLLECT,
            expr.span,
            "usage of `FromIterator::from_iter`",
            "use `.collect()` instead of `::from_iter()`",
            sugg,
            Applicability::MaybeIncorrect,
        );
    }
}

fn extract_turbofish(cx: &LateContext<'_>, expr: &hir::Expr<'_>, ty: Ty<'_>) -> String {
    fn strip_angle_brackets(s: &str) -> Option<&str> {
        s.strip_prefix('<')?.strip_suffix('>')
    }

    let call_site = expr.span.source_callsite();
    if let Some(snippet) = snippet_opt(cx, call_site)
        && let snippet_split = snippet.split("::").collect::<Vec<_>>()
        && let Some((_, elements)) = snippet_split.split_last()
    {
        if let [type_specifier, _] = snippet_split.as_slice()
            && let Some(type_specifier) = strip_angle_brackets(type_specifier)
            && let Some((type_specifier, ..)) = type_specifier.split_once(" as ")
        {
            type_specifier.to_string()
        } else {
            // is there a type specifier? (i.e.: like `<u32>` in `collections::BTreeSet::<u32>::`)
            if let Some(type_specifier) = snippet_split.iter().find(|e| strip_angle_brackets(e).is_some()) {
                // remove the type specifier from the path elements
                let without_ts = elements
                    .iter()
                    .filter_map(|e| {
                        if e == type_specifier {
                            None
                        } else {
                            Some((*e).to_string())
                        }
                    })
                    .collect::<Vec<_>>();
                // join and add the type specifier at the end (i.e.: `collections::BTreeSet<u32>`)
                format!("{}{type_specifier}", without_ts.join("::"))
            } else {
                // type is not explicitly specified so wildcards are needed
                // i.e.: 2 wildcards in `std::collections::BTreeMap<&i32, &char>`
                let ty_str = ty.to_string();
                let start = ty_str.find('<').unwrap_or(0);
                let end = ty_str.find('>').unwrap_or(ty_str.len());
                let nb_wildcard = ty_str[start..end].split(',').count();
                let wildcards = format!("_{}", ", _".repeat(nb_wildcard - 1));
                format!("{}<{wildcards}>", elements.join("::"))
            }
        }
    } else {
        ty.to_string()
    }
}
