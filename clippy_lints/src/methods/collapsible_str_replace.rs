use clippy_utils::diagnostics::span_lint_and_sugg;
// use clippy_utils::source::snippet_with_context;
use clippy_utils::visitors::for_each_expr;
use core::ops::ControlFlow;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::*;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::source_map::Spanned;
use std::unreachable;
// use rustc_span::Span;

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
        ("replace", [from, to]) => {
            // Check for `str::replace` calls with char slice for linting
            let original_recv = find_original_recv(recv);
            let original_recv_ty = cx.typeck_results().expr_ty(original_recv).peel_refs();
            if_chain! {
                // Check the receiver of the method call is `str` type
                if matches!(original_recv_ty.kind(), ty::Str);
                let from_ty = cx.typeck_results().expr_ty(from).peel_refs();
                if let ty::Array(array_ty, _) = from_ty.kind();
                if matches!(array_ty.kind(), ty::Char);
                then {
                    check_replace_call_with_char_slice(cx, from, to);
                }
            }

            match method_call(recv) {
                // Check if there's an earlier `str::replace` call
                Some(("replace", [prev_recv, prev_from, prev_to], prev_span)) => {
                    println!("Consecutive replace calls");
                    // Check that the original receiver is of `ty::Str` type
                    // Check that all the `from` args are char literals
                    // Check that all the `to` args are the same variable or has the same &str value
                    // If so, then lint
                },
                _ => {},
            }
        },
        _ => {},
    }
}

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

fn check_replace_call_with_char_slice<'tcx>(
    cx: &LateContext<'tcx>,
    from_arg: &'tcx hir::Expr<'tcx>,
    to_arg: &'tcx hir::Expr<'tcx>,
) {
    let mut has_no_var = true;
    let mut char_list: Vec<char> = Vec::new();
    // Go through the `from_arg` to collect all char literals
    let _: Option<()> = for_each_expr(from_arg, |e| {
        if let ExprKind::Lit(Spanned {
            node: LitKind::Char(val),
            ..
        }) = e.kind
        {
            char_list.push(val);
            ControlFlow::Continue(())
        } else if let ExprKind::Path(..) = e.kind {
            // If a variable is found in the char slice, no lint for first version of this lint
            has_no_var = false;
            ControlFlow::BREAK
        } else {
            ControlFlow::Continue(())
        }
    });

    if has_no_var {
        let to_arg_repr = match to_arg.kind {
            ExprKind::Lit(Spanned {
                node: LitKind::Str(to_arg_val, _),
                ..
            }) => {
                let repr = to_arg_val.as_str();
                let double_quote = "\"";
                double_quote.to_owned() + repr + double_quote
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

                path_segment_ident_names.join("::")
            },
            _ => unreachable!(),
        };

        let app = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            COLLAPSIBLE_STR_REPLACE,
            from_arg.span,
            "used slice of chars in `str::replace` call",
            "replace with",
            format!(
                "replace(|c| matches!(c, {}), {})",
                format_slice_of_chars_for_sugg(&char_list),
                to_arg_repr,
            ),
            app,
        );
    }
}

fn format_slice_of_chars_for_sugg(chars: &Vec<char>) -> String {
    let single_quoted_chars: Vec<String> = chars
        .iter()
        .map(|c| "'".to_owned() + &c.to_string() + &"'".to_owned())
        .collect();
    single_quoted_chars.join(" | ")
}
