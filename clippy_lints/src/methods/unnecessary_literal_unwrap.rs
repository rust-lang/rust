use clippy_utils::{diagnostics::span_lint_and_help, is_res_lang_ctor, path_res};
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::UNNECESSARY_LITERAL_UNWRAP;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>, name: &str) {
    let mess = if is_res_lang_ctor(cx, path_res(cx, recv), hir::LangItem::OptionSome) {
        Some((UNNECESSARY_LITERAL_UNWRAP, "Some"))
    } else {
        None
    };

    if let Some((lint, constructor)) = mess {
        let help = String::new();
        span_lint_and_help(
            cx,
            lint,
            expr.span,
            &format!("used `{name}()` on `{constructor}` value"),
            None,
            &help,
        );
    }
}
