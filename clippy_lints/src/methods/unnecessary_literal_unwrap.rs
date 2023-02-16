use clippy_utils::{diagnostics::span_lint_and_then, is_res_lang_ctor, path_res};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::UNNECESSARY_LITERAL_UNWRAP;

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    recv: &hir::Expr<'_>,
    method: &str,
    args: &[hir::Expr<'_>],
) {
    let init = clippy_utils::expr_or_init(cx, recv);

    let (constructor, call_args) = if let hir::ExprKind::Call(call, call_args) = init.kind {
        if is_res_lang_ctor(cx, path_res(cx, call), hir::LangItem::OptionSome) {
            ("Some", call_args)
        } else if is_res_lang_ctor(cx, path_res(cx, call), hir::LangItem::ResultOk) {
            ("Ok", call_args)
        } else if is_res_lang_ctor(cx, path_res(cx, call), hir::LangItem::ResultErr) {
            ("Err", call_args)
        } else {
            return;
        }
    } else if is_res_lang_ctor(cx, path_res(cx, init), hir::LangItem::OptionNone) {
        let call_args: &[hir::Expr<'_>] = &[];
        ("None", call_args)
    } else {
        return;
    };

    let help_message = format!("used `{method}()` on `{constructor}` value");
    let suggestion_message = format!("remove the `{constructor}` and `{method}()`");

    if init.span == recv.span {
        span_lint_and_then(cx, UNNECESSARY_LITERAL_UNWRAP, expr.span, &help_message, |diag| {
            let suggestions = match (constructor, method) {
                ("None", "unwrap") => vec![(expr.span, "panic!()".to_string())],
                ("None", "expect") => vec![
                    (expr.span.with_hi(args[0].span.lo()), "panic!(".to_string()),
                    (expr.span.with_lo(args[0].span.hi()), ")".to_string()),
                ],
                ("Ok", "unwrap_err") | ("Err", "unwrap") => vec![
                    (
                        recv.span.with_hi(call_args[0].span.lo()),
                        "panic!(\"{:?}\", ".to_string(),
                    ),
                    (expr.span.with_lo(call_args[0].span.hi()), ")".to_string()),
                ],
                ("Ok", "expect_err") | ("Err", "expect") => vec![
                    (
                        recv.span.with_hi(call_args[0].span.lo()),
                        "panic!(\"{1}: {:?}\", ".to_string(),
                    ),
                    (call_args[0].span.with_lo(args[0].span.lo()), ", ".to_string()),
                ],
                _ => vec![
                    (recv.span.with_hi(call_args[0].span.lo()), String::new()),
                    (expr.span.with_lo(call_args[0].span.hi()), String::new()),
                ],
            };

            diag.multipart_suggestion(suggestion_message, suggestions, Applicability::MachineApplicable);
        });
    } else {
        span_lint_and_then(cx, UNNECESSARY_LITERAL_UNWRAP, expr.span, &help_message, |diag| {
            diag.span_help(init.span, suggestion_message);
        });
    }
}
