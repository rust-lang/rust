use clippy_utils::{diagnostics::span_lint_and_then, is_res_lang_ctor, last_path_segment, path_res, MaybePath};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::UNNECESSARY_LITERAL_UNWRAP;

fn get_ty_from_args<'a>(args: Option<&'a [hir::GenericArg<'a>]>, index: usize) -> Option<&'a hir::Ty<'a>> {
    let args = args?;

    if args.len() <= index {
        return None;
    }

    match args[index] {
        hir::GenericArg::Type(ty) => match ty.kind {
            hir::TyKind::Infer => None,
            _ => Some(ty),
        },
        _ => None,
    }
}

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    recv: &hir::Expr<'_>,
    method: &str,
    args: &[hir::Expr<'_>],
) {
    let init = clippy_utils::expr_or_init(cx, recv);

    let (constructor, call_args, ty) = if let hir::ExprKind::Call(call, call_args) = init.kind {
        let Some(qpath) = call.qpath_opt() else { return };

        let args = last_path_segment(qpath).args.map(|args| args.args);
        let res = cx.qpath_res(qpath, call.hir_id());

        if is_res_lang_ctor(cx, res, hir::LangItem::OptionSome) {
            ("Some", call_args, get_ty_from_args(args, 0))
        } else if is_res_lang_ctor(cx, res, hir::LangItem::ResultOk) {
            ("Ok", call_args, get_ty_from_args(args, 0))
        } else if is_res_lang_ctor(cx, res, hir::LangItem::ResultErr) {
            ("Err", call_args, get_ty_from_args(args, 1))
        } else {
            return;
        }
    } else if is_res_lang_ctor(cx, path_res(cx, init), hir::LangItem::OptionNone) {
        let call_args: &[hir::Expr<'_>] = &[];
        ("None", call_args, None)
    } else {
        return;
    };

    let help_message = format!("used `{method}()` on `{constructor}` value");
    let suggestion_message = format!("remove the `{constructor}` and `{method}()`");

    span_lint_and_then(cx, UNNECESSARY_LITERAL_UNWRAP, expr.span, &help_message, |diag| {
        let suggestions = match (constructor, method, ty) {
            ("None", "unwrap", _) => Some(vec![(expr.span, "panic!()".to_string())]),
            ("None", "expect", _) => Some(vec![
                (expr.span.with_hi(args[0].span.lo()), "panic!(".to_string()),
                (expr.span.with_lo(args[0].span.hi()), ")".to_string()),
            ]),
            (_, _, Some(_)) => None,
            ("Ok", "unwrap_err", None) | ("Err", "unwrap", None) => Some(vec![
                (
                    recv.span.with_hi(call_args[0].span.lo()),
                    "panic!(\"{:?}\", ".to_string(),
                ),
                (expr.span.with_lo(call_args[0].span.hi()), ")".to_string()),
            ]),
            ("Ok", "expect_err", None) | ("Err", "expect", None) => Some(vec![
                (
                    recv.span.with_hi(call_args[0].span.lo()),
                    "panic!(\"{1}: {:?}\", ".to_string(),
                ),
                (call_args[0].span.with_lo(args[0].span.lo()), ", ".to_string()),
            ]),
            (_, _, None) => Some(vec![
                (recv.span.with_hi(call_args[0].span.lo()), String::new()),
                (expr.span.with_lo(call_args[0].span.hi()), String::new()),
            ]),
        };

        match (init.span == recv.span, suggestions) {
            (true, Some(suggestions)) => {
                diag.multipart_suggestion(suggestion_message, suggestions, Applicability::MachineApplicable);
            },
            _ => {
                diag.span_help(init.span, suggestion_message);
            },
        }
    });
}
