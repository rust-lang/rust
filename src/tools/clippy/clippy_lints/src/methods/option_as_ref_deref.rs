use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{path_to_local_id, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{Symbol, sym};

use super::OPTION_AS_REF_DEREF;

/// lint use of `_.as_ref().map(Deref::deref)` for `Option`s
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    as_ref_recv: &hir::Expr<'_>,
    map_arg: &hir::Expr<'_>,
    is_mut: bool,
    msrv: Msrv,
) {
    let same_mutability = |m| (is_mut && m == &hir::Mutability::Mut) || (!is_mut && m == &hir::Mutability::Not);

    let option_ty = cx.typeck_results().expr_ty(as_ref_recv);
    if !is_type_diagnostic_item(cx, option_ty, sym::Option) {
        return;
    }

    let deref_aliases: [Symbol; 7] = [
        sym::cstring_as_c_str,
        sym::os_string_as_os_str,
        sym::pathbuf_as_path,
        sym::string_as_str,
        sym::string_as_mut_str,
        sym::vec_as_slice,
        sym::vec_as_mut_slice,
    ];

    let is_deref = match map_arg.kind {
        hir::ExprKind::Path(ref expr_qpath) => cx
            .qpath_res(expr_qpath, map_arg.hir_id)
            .opt_def_id()
            .and_then(|fun_def_id| cx.tcx.get_diagnostic_name(fun_def_id))
            .is_some_and(|fun_name| {
                matches!(fun_name, sym::deref_method | sym::deref_mut_method) || deref_aliases.contains(&fun_name)
            }),
        hir::ExprKind::Closure(&hir::Closure { body, .. }) => {
            let closure_body = cx.tcx.hir_body(body);
            let closure_expr = peel_blocks(closure_body.value);

            match &closure_expr.kind {
                hir::ExprKind::MethodCall(_, receiver, [], _) => {
                    if path_to_local_id(receiver, closure_body.params[0].pat.hir_id)
                        && let adj = cx
                            .typeck_results()
                            .expr_adjustments(receiver)
                            .iter()
                            .map(|x| &x.kind)
                            .collect::<Box<[_]>>()
                        && let [ty::adjustment::Adjust::Deref(None), ty::adjustment::Adjust::Borrow(_)] = *adj
                        && let method_did = cx.typeck_results().type_dependent_def_id(closure_expr.hir_id).unwrap()
                        && let Some(method_name) = cx.tcx.get_diagnostic_name(method_did)
                    {
                        matches!(method_name, sym::deref_method | sym::deref_mut_method)
                            || deref_aliases.contains(&method_name)
                    } else {
                        false
                    }
                },
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, m, inner) if same_mutability(m) => {
                    if let hir::ExprKind::Unary(hir::UnOp::Deref, inner1) = inner.kind
                        && let hir::ExprKind::Unary(hir::UnOp::Deref, inner2) = inner1.kind
                    {
                        path_to_local_id(inner2, closure_body.params[0].pat.hir_id)
                    } else {
                        false
                    }
                },
                _ => false,
            }
        },
        _ => false,
    };

    if is_deref && msrv.meets(cx, msrvs::OPTION_AS_DEREF) {
        let current_method = if is_mut {
            format!(".as_mut().map({})", snippet(cx, map_arg.span, ".."))
        } else {
            format!(".as_ref().map({})", snippet(cx, map_arg.span, ".."))
        };
        let method_hint = if is_mut { "as_deref_mut" } else { "as_deref" };
        let hint = format!("{}.{method_hint}()", snippet(cx, as_ref_recv.span, ".."));
        let suggestion = format!("consider using {method_hint}");

        let msg = format!("called `{current_method}` on an `Option` value");
        span_lint_and_sugg(
            cx,
            OPTION_AS_REF_DEREF,
            expr.span,
            msg,
            suggestion,
            hint,
            Applicability::MachineApplicable,
        );
    }
}
