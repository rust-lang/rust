use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::matching_root_macro_call;
use clippy_utils::msrvs::Msrv;
use clippy_utils::source::snippet;
use clippy_utils::visitors::{for_each_expr_without_closures, is_local_used};
use clippy_utils::{is_in_const_context, path_to_local, sym};
use rustc_ast::{BorrowKind, LitKind};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Arm, BinOpKind, Expr, ExprKind, MatchSource, Node, PatKind, UnOp};
use rustc_lint::LateContext;
use rustc_span::symbol::Ident;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;
use std::ops::ControlFlow;

use super::{REDUNDANT_GUARDS, pat_contains_disallowed_or};

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, arms: &'tcx [Arm<'tcx>], msrv: Msrv) {
    for outer_arm in arms {
        let Some(guard) = outer_arm.guard else {
            continue;
        };

        // `Some(x) if matches!(x, y)`
        if let ExprKind::Match(scrutinee, [arm, _], MatchSource::Normal) = guard.kind
            && matching_root_macro_call(cx, guard.span, sym::matches_macro).is_some()
            && let Some(binding) = get_pat_binding(cx, scrutinee, outer_arm)
            && !pat_contains_disallowed_or(cx, arm.pat, msrv)
        {
            let pat_span = match (arm.pat.kind, binding.byref_ident) {
                (PatKind::Ref(pat, _), Some(_)) => pat.span,
                (PatKind::Ref(..), None) | (_, Some(_)) => continue,
                _ => arm.pat.span,
            };
            emit_redundant_guards(
                cx,
                outer_arm,
                guard.span,
                snippet(cx, pat_span, "<binding>"),
                &binding,
                arm.guard,
            );
        }
        // `Some(x) if let Some(2) = x`
        else if let ExprKind::Let(let_expr) = guard.kind
            && let Some(binding) = get_pat_binding(cx, let_expr.init, outer_arm)
            && !pat_contains_disallowed_or(cx, let_expr.pat, msrv)
        {
            let pat_span = match (let_expr.pat.kind, binding.byref_ident) {
                (PatKind::Ref(pat, _), Some(_)) => pat.span,
                (PatKind::Ref(..), None) | (_, Some(_)) => continue,
                _ => let_expr.pat.span,
            };
            emit_redundant_guards(
                cx,
                outer_arm,
                let_expr.span,
                snippet(cx, pat_span, "<binding>"),
                &binding,
                None,
            );
        }
        // `Some(x) if x == Some(2)`
        // `Some(x) if Some(2) == x`
        else if let ExprKind::Binary(bin_op, local, pat) = guard.kind
            && matches!(bin_op.node, BinOpKind::Eq)
            // Ensure they have the same type. If they don't, we'd need deref coercion which isn't
            // possible (currently) in a pattern. In some cases, you can use something like
            // `as_deref` or similar but in general, we shouldn't lint this as it'd create an
            // extraordinary amount of FPs.
            //
            // This isn't necessary in the other two checks, as they must be a pattern already.
            && cx.typeck_results().expr_ty(local) == cx.typeck_results().expr_ty(pat)
            // Since we want to lint on both `x == Some(2)` and `Some(2) == x`, we might have to "swap"
            // `local` and `pat`, depending on which side they are.
            && let Some((binding, pat)) = get_pat_binding(cx, local, outer_arm)
                .map(|binding| (binding, pat))
                .or_else(|| get_pat_binding(cx, pat, outer_arm).map(|binding| (binding, local)))
            && expr_can_be_pat(cx, pat)
        {
            let pat_span = match (pat.kind, binding.byref_ident) {
                (ExprKind::AddrOf(BorrowKind::Ref, _, expr), Some(_)) => expr.span,
                (ExprKind::AddrOf(..), None) | (_, Some(_)) => continue,
                _ => pat.span,
            };
            emit_redundant_guards(
                cx,
                outer_arm,
                guard.span,
                snippet(cx, pat_span, "<binding>"),
                &binding,
                None,
            );
        } else if let ExprKind::MethodCall(path, recv, args, ..) = guard.kind
            && let Some(binding) = get_pat_binding(cx, recv, outer_arm)
        {
            check_method_calls(cx, outer_arm, path.ident.name, recv, args, guard, &binding);
        }
    }
}

fn check_method_calls<'tcx>(
    cx: &LateContext<'tcx>,
    arm: &Arm<'tcx>,
    method: Symbol,
    recv: &Expr<'_>,
    args: &[Expr<'_>],
    if_expr: &Expr<'_>,
    binding: &PatBindingInfo,
) {
    let ty = cx.typeck_results().expr_ty(recv).peel_refs();
    let slice_like = ty.is_slice() || ty.is_array();

    let sugg = if method == sym::is_empty {
        // `s if s.is_empty()` becomes ""
        // `arr if arr.is_empty()` becomes []

        if ty.is_str() && !is_in_const_context(cx) {
            r#""""#.into()
        } else if slice_like {
            "[]".into()
        } else {
            return;
        }
    } else if slice_like
        && let Some(needle) = args.first()
        && let ExprKind::AddrOf(.., needle) = needle.kind
        && let ExprKind::Array(needles) = needle.kind
        && needles.iter().all(|needle| expr_can_be_pat(cx, needle))
    {
        // `arr if arr.starts_with(&[123])` becomes [123, ..]
        // `arr if arr.ends_with(&[123])` becomes [.., 123]
        // `arr if arr.starts_with(&[])` becomes [..]  (why would anyone write this?)

        let mut sugg = snippet(cx, needle.span, "<needle>").into_owned();

        if needles.is_empty() {
            sugg.insert_str(1, "..");
        } else if method == sym::starts_with {
            sugg.insert_str(sugg.len() - 1, ", ..");
        } else if method == sym::ends_with {
            sugg.insert_str(1, ".., ");
        } else {
            return;
        }

        sugg.into()
    } else {
        return;
    };

    emit_redundant_guards(cx, arm, if_expr.span, sugg, binding, None);
}

struct PatBindingInfo {
    span: Span,
    byref_ident: Option<Ident>,
    is_field: bool,
}

fn get_pat_binding<'tcx>(
    cx: &LateContext<'tcx>,
    guard_expr: &Expr<'_>,
    outer_arm: &Arm<'tcx>,
) -> Option<PatBindingInfo> {
    if let Some(local) = path_to_local(guard_expr)
        && !is_local_used(cx, outer_arm.body, local)
    {
        let mut span = None;
        let mut byref_ident = None;
        let mut multiple_bindings = false;
        // `each_binding` gives the `HirId` of the `Pat` itself, not the binding
        outer_arm.pat.walk(|pat| {
            if let PatKind::Binding(bind_annot, hir_id, ident, _) = pat.kind
                && hir_id == local
            {
                if matches!(bind_annot.0, rustc_ast::ByRef::Yes(_)) {
                    let _ = byref_ident.insert(ident);
                }
                // the second call of `replace()` returns a `Some(span)`, meaning a multi-binding pattern
                if span.replace(pat.span).is_some() {
                    multiple_bindings = true;
                    return false;
                }
            }
            true
        });

        // Ignore bindings from or patterns, like `First(x) | Second(x, _) | Third(x, _, _)`
        if !multiple_bindings {
            return span.map(|span| PatBindingInfo {
                span,
                byref_ident,
                is_field: matches!(cx.tcx.parent_hir_node(local), Node::PatField(_)),
            });
        }
    }

    None
}

fn emit_redundant_guards<'tcx>(
    cx: &LateContext<'tcx>,
    outer_arm: &Arm<'tcx>,
    guard_span: Span,
    binding_replacement: Cow<'static, str>,
    pat_binding: &PatBindingInfo,
    inner_guard: Option<&Expr<'_>>,
) {
    span_lint_and_then(
        cx,
        REDUNDANT_GUARDS,
        guard_span.source_callsite(),
        "redundant guard",
        |diag| {
            let suggestion_span = match *pat_binding {
                PatBindingInfo {
                    span,
                    byref_ident: Some(ident),
                    is_field: true,
                } => (span, format!("{ident}: {binding_replacement}")),
                PatBindingInfo {
                    span, is_field: true, ..
                } => (span.shrink_to_hi(), format!(": {binding_replacement}")),
                PatBindingInfo { span, .. } => (span, binding_replacement.into_owned()),
            };
            diag.multipart_suggestion_verbose(
                "try",
                vec![
                    suggestion_span,
                    (
                        guard_span.source_callsite().with_lo(outer_arm.pat.span.hi()),
                        inner_guard.map_or_else(String::new, |guard| {
                            format!(" if {}", snippet(cx, guard.span, "<guard>"))
                        }),
                    ),
                ],
                Applicability::MaybeIncorrect,
            );
        },
    );
}

/// Checks if the given `Expr` can also be represented as a `Pat`.
fn expr_can_be_pat(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    for_each_expr_without_closures(expr, |expr| {
        if match expr.kind {
            ExprKind::Call(c, ..) if let ExprKind::Path(qpath) = c.kind => {
                // Allow ctors
                matches!(cx.qpath_res(&qpath, c.hir_id), Res::Def(DefKind::Ctor(..), ..))
            },
            ExprKind::Path(qpath) => {
                matches!(
                    cx.qpath_res(&qpath, expr.hir_id),
                    Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Ctor(..), ..),
                )
            },
            ExprKind::AddrOf(..)
            | ExprKind::Array(..)
            | ExprKind::Tup(..)
            | ExprKind::Struct(..)
            | ExprKind::Unary(UnOp::Neg, _) => true,
            ExprKind::Lit(lit) if !matches!(lit.node, LitKind::CStr(..)) => true,
            _ => false,
        } {
            return ControlFlow::Continue(());
        }

        ControlFlow::Break(())
    })
    .is_none()
}
