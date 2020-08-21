use crate::utils::{is_type_diagnostic_item, match_qpath, snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::UNNECESSARY_LAZY_EVALUATIONS;

// Return true if the expression is an accessor of any of the arguments
fn expr_uses_argument(expr: &hir::Expr<'_>, params: &[hir::Param<'_>]) -> bool {
    params.iter().any(|arg| {
        if_chain! {
            if let hir::PatKind::Binding(_, _, ident, _) = arg.pat.kind;
            if let hir::ExprKind::Path(hir::QPath::Resolved(_, ref path)) = expr.kind;
            if let [p, ..] = path.segments;
            then {
                ident.name == p.ident.name
            } else {
                false
            }
        }
    })
}

fn match_any_qpath(path: &hir::QPath<'_>, paths: &[&[&str]]) -> bool {
    paths.iter().any(|candidate| match_qpath(path, candidate))
}

fn can_simplify(expr: &hir::Expr<'_>, params: &[hir::Param<'_>], variant_calls: bool) -> bool {
    match expr.kind {
        // Closures returning literals can be unconditionally simplified
        hir::ExprKind::Lit(_) => true,

        hir::ExprKind::Index(ref object, ref index) => {
            // arguments are not being indexed into
            if expr_uses_argument(object, params) {
                false
            } else {
                // arguments are not used as index
                !expr_uses_argument(index, params)
            }
        },

        // Reading fields can be simplified if the object is not an argument of the closure
        hir::ExprKind::Field(ref object, _) => !expr_uses_argument(object, params),

        // Paths can be simplified if the root is not the argument, this also covers None
        hir::ExprKind::Path(_) => !expr_uses_argument(expr, params),

        // Calls to Some, Ok, Err can be considered literals if they don't derive an argument
        hir::ExprKind::Call(ref func, ref args) => if_chain! {
            if variant_calls; // Disable lint when rules conflict with bind_instead_of_map
            if let hir::ExprKind::Path(ref path) = func.kind;
            if match_any_qpath(path, &[&["Some"], &["Ok"], &["Err"]]);
            then {
                // Recursively check all arguments
                args.iter().all(|arg| can_simplify(arg, params, variant_calls))
            } else {
                false
            }
        },

        // For anything more complex than the above, a closure is probably the right solution,
        // or the case is handled by an other lint
        _ => false,
    }
}

/// lint use of `<fn>_else(simple closure)` for `Option`s and `Result`s that can be
/// replaced with `<fn>(return value of simple closure)`
pub(super) fn lint<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    args: &'tcx [hir::Expr<'_>],
    allow_variant_calls: bool,
    simplify_using: &str,
) {
    let is_option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&args[0]), sym!(option_type));
    let is_result = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&args[0]), sym!(result_type));

    if is_option || is_result {
        if let hir::ExprKind::Closure(_, _, eid, _, _) = args[1].kind {
            let body = cx.tcx.hir().body(eid);
            let ex = &body.value;
            let params = &body.params;

            if can_simplify(ex, params, allow_variant_calls) {
                let msg = if is_option {
                    "unnecessary closure used to substitute value for `Option::None`"
                } else {
                    "unnecessary closure used to substitute value for `Result::Err`"
                };

                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_LAZY_EVALUATIONS,
                    expr.span,
                    msg,
                    &format!("Use `{}` instead", simplify_using),
                    format!(
                        "{0}.{1}({2})",
                        snippet(cx, args[0].span, ".."),
                        simplify_using,
                        snippet(cx, ex.span, ".."),
                    ),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
