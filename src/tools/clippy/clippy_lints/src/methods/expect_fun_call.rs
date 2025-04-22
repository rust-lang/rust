use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::{FormatArgsStorage, format_args_inputs_span, root_macro_call_first_node};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{is_type_diagnostic_item, is_type_lang_item};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Span;
use rustc_span::symbol::sym;
use std::borrow::Cow;

use super::EXPECT_FUN_CALL;

/// Checks for the `EXPECT_FUN_CALL` lint.
#[allow(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    format_args_storage: &FormatArgsStorage,
    expr: &hir::Expr<'_>,
    method_span: Span,
    name: &str,
    receiver: &'tcx hir::Expr<'tcx>,
    args: &'tcx [hir::Expr<'tcx>],
) {
    // Strip `&`, `as_ref()` and `as_str()` off `arg` until we're left with either a `String` or
    // `&str`
    fn get_arg_root<'a>(cx: &LateContext<'_>, arg: &'a hir::Expr<'a>) -> &'a hir::Expr<'a> {
        let mut arg_root = arg;
        loop {
            arg_root = match &arg_root.kind {
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, expr) => expr,
                hir::ExprKind::MethodCall(method_name, receiver, [], ..) => {
                    if (method_name.ident.name == sym::as_str || method_name.ident.name == sym::as_ref) && {
                        let arg_type = cx.typeck_results().expr_ty(receiver);
                        let base_type = arg_type.peel_refs();
                        base_type.is_str() || is_type_lang_item(cx, base_type, hir::LangItem::String)
                    } {
                        receiver
                    } else {
                        break;
                    }
                },
                _ => break,
            };
        }
        arg_root
    }

    // Only `&'static str` or `String` can be used directly in the `panic!`. Other types should be
    // converted to string.
    fn requires_to_string(cx: &LateContext<'_>, arg: &hir::Expr<'_>) -> bool {
        let arg_ty = cx.typeck_results().expr_ty(arg);
        if is_type_lang_item(cx, arg_ty, hir::LangItem::String) {
            return false;
        }
        if let ty::Ref(_, ty, ..) = arg_ty.kind()
            && ty.is_str()
            && can_be_static_str(cx, arg)
        {
            return false;
        }
        true
    }

    // Check if an expression could have type `&'static str`, knowing that it
    // has type `&str` for some lifetime.
    fn can_be_static_str(cx: &LateContext<'_>, arg: &hir::Expr<'_>) -> bool {
        match arg.kind {
            hir::ExprKind::Lit(_) => true,
            hir::ExprKind::Call(fun, _) => {
                if let hir::ExprKind::Path(ref p) = fun.kind {
                    match cx.qpath_res(p, fun.hir_id) {
                        hir::def::Res::Def(hir::def::DefKind::Fn | hir::def::DefKind::AssocFn, def_id) => matches!(
                            cx.tcx.fn_sig(def_id).instantiate_identity().output().skip_binder().kind(),
                            ty::Ref(re, ..) if re.is_static(),
                        ),
                        _ => false,
                    }
                } else {
                    false
                }
            },
            hir::ExprKind::MethodCall(..) => {
                cx.typeck_results()
                    .type_dependent_def_id(arg.hir_id)
                    .is_some_and(|method_id| {
                        matches!(
                            cx.tcx.fn_sig(method_id).instantiate_identity().output().skip_binder().kind(),
                            ty::Ref(re, ..) if re.is_static()
                        )
                    })
            },
            hir::ExprKind::Path(ref p) => matches!(
                cx.qpath_res(p, arg.hir_id),
                hir::def::Res::Def(hir::def::DefKind::Const | hir::def::DefKind::Static { .. }, _)
            ),
            _ => false,
        }
    }

    fn is_call(node: &hir::ExprKind<'_>) -> bool {
        match node {
            hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, expr) => {
                is_call(&expr.kind)
            },
            hir::ExprKind::Call(..)
            | hir::ExprKind::MethodCall(..)
            // These variants are debatable or require further examination
            | hir::ExprKind::If(..)
            | hir::ExprKind::Match(..)
            | hir::ExprKind::Block{ .. } => true,
            _ => false,
        }
    }

    if args.len() != 1 || name != "expect" || !is_call(&args[0].kind) {
        return;
    }

    let receiver_type = cx.typeck_results().expr_ty_adjusted(receiver);
    let closure_args = if is_type_diagnostic_item(cx, receiver_type, sym::Option) {
        "||"
    } else if is_type_diagnostic_item(cx, receiver_type, sym::Result) {
        "|_|"
    } else {
        return;
    };

    let arg_root = get_arg_root(cx, &args[0]);

    let span_replace_word = method_span.with_hi(expr.span.hi());

    let mut applicability = Applicability::MachineApplicable;

    // Special handling for `format!` as arg_root
    if let Some(macro_call) = root_macro_call_first_node(cx, arg_root) {
        if cx.tcx.is_diagnostic_item(sym::format_macro, macro_call.def_id)
            && let Some(format_args) = format_args_storage.get(cx, arg_root, macro_call.expn)
        {
            let span = format_args_inputs_span(format_args);
            let sugg = snippet_with_applicability(cx, span, "..", &mut applicability);
            span_lint_and_sugg(
                cx,
                EXPECT_FUN_CALL,
                span_replace_word,
                format!("function call inside of `{name}`"),
                "try",
                format!("unwrap_or_else({closure_args} panic!({sugg}))"),
                applicability,
            );
        }
        return;
    }

    let mut arg_root_snippet: Cow<'_, _> = snippet_with_applicability(cx, arg_root.span, "..", &mut applicability);
    if requires_to_string(cx, arg_root) {
        arg_root_snippet.to_mut().push_str(".to_string()");
    }

    span_lint_and_sugg(
        cx,
        EXPECT_FUN_CALL,
        span_replace_word,
        format!("function call inside of `{name}`"),
        "try",
        format!("unwrap_or_else({closure_args} {{ panic!(\"{{}}\", {arg_root_snippet}) }})"),
        applicability,
    );
}
