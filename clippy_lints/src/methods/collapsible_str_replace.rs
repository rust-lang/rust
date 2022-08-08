use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::visitors::for_each_expr;
use core::ops::ControlFlow;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::*;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::source_map::Spanned;
use rustc_span::Span;

use super::method_call;
use super::COLLAPSIBLE_STR_REPLACE;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    name: &str,
    recv: &'tcx hir::Expr<'tcx>,
    args: &'tcx [hir::Expr<'tcx>],
) {
    match (name, args) {
        ("replace", ..) => {
            // The receiver of the method call must be `str` type to lint `collapsible_str_replace`
            let original_recv = find_original_recv(recv);
            let original_recv_ty_kind = cx.typeck_results().expr_ty(original_recv).peel_refs().kind();
            let original_recv_is_str_kind = matches!(original_recv_ty_kind, ty::Str);

            if_chain! {
                if original_recv_is_str_kind;
                if let Some(parent) = get_parent_expr(cx, expr);
                if let Some((name, ..)) = method_call(parent);

                then {
                    match name {
                        // If the parent node is a `str::replace` call, we've already handled the lint, don't lint again
                        "replace" => return,
                        _ => {
                            check_consecutive_replace_calls(cx, expr);
                            return;
                        },
                    }
                }
            }

            match method_call(recv) {
                // Check if there's an earlier `str::replace` call
                Some(("replace", ..)) => {
                    if original_recv_is_str_kind {
                        check_consecutive_replace_calls(cx, expr);
                        return;
                    }
                },
                _ => {},
            }
        },
        _ => {},
    }
}

/// Check a chain of `str::replace` calls for `collapsible_str_replace` lint.
fn check_consecutive_replace_calls<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
    if_chain! {
        if let Some(from_args) = get_replace_call_from_args_if_all_char_ty(cx, expr);
        if let Some(to_arg) = get_replace_call_unique_to_arg_repr(expr);
        then {
            let earliest_replace_call_span = get_earliest_replace_call_span(expr);

            if replace_call_from_args_are_only_lit_chars(&from_args) {
                let from_arg_reprs: Vec<String> = from_args.iter().map(|from_arg| {
                    get_replace_call_char_arg_repr(from_arg).unwrap()
                }).collect();
                let app = Applicability::MachineApplicable;

                span_lint_and_sugg(
                    cx,
                    COLLAPSIBLE_STR_REPLACE,
                    expr.span.with_lo(earliest_replace_call_span.lo()),
                    "used consecutive `str::replace` call",
                    "replace with",
                    format!(
                        "replace(|c| matches!(c, {}), {})",
                        from_arg_reprs.join(" | "),
                        to_arg,
                    ),
                    app,
                );
            } else {
                // Use fallback lint
                let from_arg_reprs: Vec<String> = from_args.iter().map(|from_arg| {
                    get_replace_call_char_arg_repr(from_arg).unwrap()
                }).collect();
                let app = Applicability::MachineApplicable;

                span_lint_and_sugg(
                    cx,
                    COLLAPSIBLE_STR_REPLACE,
                    expr.span.with_lo(earliest_replace_call_span.lo()),
                    "used consecutive `str::replace` call",
                    "replace with",
                    format!(
                        "replace(&[{}], {})",
                        from_arg_reprs.join(" , "),
                        to_arg,
                    ),
                    app,
                );
            }
        }
    }
}

/// Check if all the `from` arguments of a chain of consecutive calls to `str::replace`
/// are all of `ExprKind::Lit` types. If any is not, return false.
fn replace_call_from_args_are_only_lit_chars<'tcx>(from_args: &Vec<&'tcx hir::Expr<'tcx>>) -> bool {
    let mut only_lit_chars = true;

    for from_arg in from_args.iter() {
        match from_arg.kind {
            ExprKind::Lit(..) => {},
            _ => only_lit_chars = false,
        }
    }

    only_lit_chars
}

/// Collect and return all of the `from` arguments of a chain of consecutive `str::replace` calls
/// if these `from` arguments's expressions are of the `ty::Char` kind. Otherwise return `None`.
fn get_replace_call_from_args_if_all_char_ty<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
) -> Option<Vec<&'tcx hir::Expr<'tcx>>> {
    let mut all_from_args_are_chars = true;
    let mut from_args = Vec::new();

    let _: Option<()> = for_each_expr(expr, |e| {
        if let Some((name, [_, args @ ..], _)) = method_call(e) {
            match (name, args) {
                ("replace", [from, _]) => {
                    let from_ty_kind = cx.typeck_results().expr_ty(from).peel_refs().kind();
                    if matches!(from_ty_kind, ty::Char) {
                        from_args.push(from);
                    } else {
                        all_from_args_are_chars = false;
                    }
                    ControlFlow::Continue(())
                },
                _ => ControlFlow::BREAK,
            }
        } else {
            ControlFlow::Continue(())
        }
    });

    if all_from_args_are_chars {
        return Some(from_args);
    } else {
        return None;
    }
}

/// Return a unique String representation of the `to` argument used in a chain of `str::replace`
/// calls if each `str::replace` call's `to` argument is identical to the other `to` arguments in
/// the chain. Otherwise, return `None`.
fn get_replace_call_unique_to_arg_repr<'tcx>(expr: &'tcx hir::Expr<'tcx>) -> Option<String> {
    let mut to_args = Vec::new();

    let _: Option<()> = for_each_expr(expr, |e| {
        if let Some((name, [_, args @ ..], _)) = method_call(e) {
            match (name, args) {
                ("replace", [_, to]) => {
                    to_args.push(to);
                    ControlFlow::Continue(())
                },
                _ => ControlFlow::BREAK,
            }
        } else {
            ControlFlow::Continue(())
        }
    });

    // let mut to_arg_repr_set = FxHashSet::default();
    let mut to_arg_reprs = Vec::new();
    for &to_arg in to_args.iter() {
        if let Some(to_arg_repr) = get_replace_call_char_arg_repr(to_arg) {
            to_arg_reprs.push(to_arg_repr);
        }
    }

    let to_arg_repr_set = FxHashSet::from_iter(to_arg_reprs.iter().cloned());
    // Check if the set of `to` argument representations has more than one unique value
    if to_arg_repr_set.len() != 1 {
        return None;
    }

    // Return the single representation value
    to_arg_reprs.pop()
}

/// Get the representation of an argument of a `str::replace` call either of the literal char value
/// or variable name, i.e. the resolved path segments `ident`.
/// Return:
/// - the str literal with double quotes, e.g. "\"l\""
/// - the char literal with single quotes, e.g. "'l'"
/// - the variable as a String, e.g. "l"
fn get_replace_call_char_arg_repr<'tcx>(arg: &'tcx hir::Expr<'tcx>) -> Option<String> {
    match arg.kind {
        ExprKind::Lit(Spanned {
            node: LitKind::Str(to_arg_val, _),
            ..
        }) => {
            let repr = to_arg_val.as_str();
            let double_quote = "\"";
            Some(double_quote.to_owned() + repr + double_quote)
        },
        ExprKind::Lit(Spanned {
            node: LitKind::Char(to_arg_val),
            ..
        }) => {
            let repr = to_arg_val.to_string();
            let double_quote = "\'";
            Some(double_quote.to_owned() + &repr + double_quote)
        },
        ExprKind::Path(QPath::Resolved(
            _,
            Path {
                segments: path_segments,
                ..
            },
        )) => {
            // join the path_segments values by "::"
            let path_segment_ident_names: Vec<&str> = path_segments
                .iter()
                .map(|path_seg| path_seg.ident.name.as_str())
                .collect();
            Some(path_segment_ident_names.join("::"))
        },
        _ => None,
    }
}

fn get_earliest_replace_call_span<'tcx>(expr: &'tcx hir::Expr<'tcx>) -> Span {
    let mut earliest_replace_call_span = expr.span;

    let _: Option<()> = for_each_expr(expr, |e| {
        if let Some((name, [_, args @ ..], span)) = method_call(e) {
            match (name, args) {
                ("replace", [_, _]) => {
                    earliest_replace_call_span = span;
                    ControlFlow::Continue(())
                },
                _ => ControlFlow::BREAK,
            }
        } else {
            ControlFlow::Continue(())
        }
    });

    earliest_replace_call_span
}

/// Find the original receiver of a chain of `str::replace` method calls.
fn find_original_recv<'tcx>(recv: &'tcx hir::Expr<'tcx>) -> &'tcx hir::Expr<'tcx> {
    let mut original_recv = recv;

    let _: Option<()> = for_each_expr(recv, |e| {
        if let Some((name, [prev_recv, args @ ..], _)) = method_call(e) {
            match (name, args) {
                ("replace", [_, _]) => {
                    original_recv = prev_recv;
                    ControlFlow::Continue(())
                },
                _ => ControlFlow::BREAK,
            }
        } else {
            ControlFlow::Continue(())
        }
    });

    original_recv
}
