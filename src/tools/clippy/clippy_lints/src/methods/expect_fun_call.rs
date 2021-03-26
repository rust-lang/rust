use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_expn_of;
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::source_map::Span;
use rustc_span::symbol::sym;
use std::borrow::Cow;

use super::EXPECT_FUN_CALL;

/// Checks for the `EXPECT_FUN_CALL` lint.
#[allow(clippy::too_many_lines)]
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, method_span: Span, name: &str, args: &[hir::Expr<'_>]) {
    // Strip `&`, `as_ref()` and `as_str()` off `arg` until we're left with either a `String` or
    // `&str`
    fn get_arg_root<'a>(cx: &LateContext<'_>, arg: &'a hir::Expr<'a>) -> &'a hir::Expr<'a> {
        let mut arg_root = arg;
        loop {
            arg_root = match &arg_root.kind {
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, expr) => expr,
                hir::ExprKind::MethodCall(method_name, _, call_args, _) => {
                    if call_args.len() == 1
                        && (method_name.ident.name == sym::as_str || method_name.ident.name == sym!(as_ref))
                        && {
                            let arg_type = cx.typeck_results().expr_ty(&call_args[0]);
                            let base_type = arg_type.peel_refs();
                            *base_type.kind() == ty::Str || is_type_diagnostic_item(cx, base_type, sym::string_type)
                        }
                    {
                        &call_args[0]
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
        if is_type_diagnostic_item(cx, arg_ty, sym::string_type) {
            return false;
        }
        if let ty::Ref(_, ty, ..) = arg_ty.kind() {
            if *ty.kind() == ty::Str && can_be_static_str(cx, arg) {
                return false;
            }
        };
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
                            cx.tcx.fn_sig(def_id).output().skip_binder().kind(),
                            ty::Ref(ty::ReStatic, ..)
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
                    .map_or(false, |method_id| {
                        matches!(
                            cx.tcx.fn_sig(method_id).output().skip_binder().kind(),
                            ty::Ref(ty::ReStatic, ..)
                        )
                    })
            },
            hir::ExprKind::Path(ref p) => matches!(
                cx.qpath_res(p, arg.hir_id),
                hir::def::Res::Def(hir::def::DefKind::Const | hir::def::DefKind::Static, _)
            ),
            _ => false,
        }
    }

    fn generate_format_arg_snippet(
        cx: &LateContext<'_>,
        a: &hir::Expr<'_>,
        applicability: &mut Applicability,
    ) -> Vec<String> {
        if_chain! {
            if let hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, ref format_arg) = a.kind;
            if let hir::ExprKind::Match(ref format_arg_expr, _, _) = format_arg.kind;
            if let hir::ExprKind::Tup(ref format_arg_expr_tup) = format_arg_expr.kind;

            then {
                format_arg_expr_tup
                    .iter()
                    .map(|a| snippet_with_applicability(cx, a.span, "..", applicability).into_owned())
                    .collect()
            } else {
                unreachable!()
            }
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

    if args.len() != 2 || name != "expect" || !is_call(&args[1].kind) {
        return;
    }

    let receiver_type = cx.typeck_results().expr_ty_adjusted(&args[0]);
    let closure_args = if is_type_diagnostic_item(cx, receiver_type, sym::option_type) {
        "||"
    } else if is_type_diagnostic_item(cx, receiver_type, sym::result_type) {
        "|_|"
    } else {
        return;
    };

    let arg_root = get_arg_root(cx, &args[1]);

    let span_replace_word = method_span.with_hi(expr.span.hi());

    let mut applicability = Applicability::MachineApplicable;

    //Special handling for `format!` as arg_root
    if_chain! {
        if let hir::ExprKind::Block(block, None) = &arg_root.kind;
        if block.stmts.len() == 1;
        if let hir::StmtKind::Local(local) = &block.stmts[0].kind;
        if let Some(arg_root) = &local.init;
        if let hir::ExprKind::Call(ref inner_fun, ref inner_args) = arg_root.kind;
        if is_expn_of(inner_fun.span, "format").is_some() && inner_args.len() == 1;
        if let hir::ExprKind::Call(_, format_args) = &inner_args[0].kind;
        then {
            let fmt_spec = &format_args[0];
            let fmt_args = &format_args[1];

            let mut args = vec![snippet(cx, fmt_spec.span, "..").into_owned()];

            args.extend(generate_format_arg_snippet(cx, fmt_args, &mut applicability));

            let sugg = args.join(", ");

            span_lint_and_sugg(
                cx,
                EXPECT_FUN_CALL,
                span_replace_word,
                &format!("use of `{}` followed by a function call", name),
                "try this",
                format!("unwrap_or_else({} panic!({}))", closure_args, sugg),
                applicability,
            );

            return;
        }
    }

    let mut arg_root_snippet: Cow<'_, _> = snippet_with_applicability(cx, arg_root.span, "..", &mut applicability);
    if requires_to_string(cx, arg_root) {
        arg_root_snippet.to_mut().push_str(".to_string()");
    }

    span_lint_and_sugg(
        cx,
        EXPECT_FUN_CALL,
        span_replace_word,
        &format!("use of `{}` followed by a function call", name),
        "try this",
        format!(
            "unwrap_or_else({} {{ panic!(\"{{}}\", {}) }})",
            closure_args, arg_root_snippet
        ),
        applicability,
    );
}
