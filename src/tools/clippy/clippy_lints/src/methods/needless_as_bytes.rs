use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_lang_item;
use rustc_errors::Applicability;
use rustc_hir::{Expr, LangItem};
use rustc_lint::LateContext;
use rustc_span::{Span, Symbol};

use super::NEEDLESS_AS_BYTES;

pub fn check(cx: &LateContext<'_>, prev_method: Symbol, method: Symbol, prev_recv: &Expr<'_>, span: Span) {
    let ty1 = cx.typeck_results().expr_ty_adjusted(prev_recv).peel_refs();
    if is_type_lang_item(cx, ty1, LangItem::String) || ty1.is_str() {
        let mut app = Applicability::MachineApplicable;
        let sugg = Sugg::hir_with_context(cx, prev_recv, span.ctxt(), "..", &mut app);
        span_lint_and_sugg(
            cx,
            NEEDLESS_AS_BYTES,
            span,
            format!("needless call to `{prev_method}`"),
            format!("`{method}()` can be called directly on strings"),
            format!("{sugg}.{method}()"),
            app,
        );
    }
}
