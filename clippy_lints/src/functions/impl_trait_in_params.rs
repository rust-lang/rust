use clippy_utils::{diagnostics::span_lint_and_then, is_in_test_function};

use rustc_hir::{intravisit::FnKind, Body, HirId};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::IMPL_TRAIT_IN_PARAMS;

pub(super) fn check_fn<'tcx>(cx: &LateContext<'_>, kind: &'tcx FnKind<'_>, body: &'tcx Body<'_>, hir_id: HirId) {
    if cx.tcx.visibility(cx.tcx.hir().body_owner_def_id(body.id())).is_public() && !is_in_test_function(cx.tcx, hir_id)
    {
        if let FnKind::ItemFn(ident, generics, _) = kind {
            for param in generics.params {
                if param.is_impl_trait() {
                    // No generics with nested generics, and no generics like FnMut(x)
                    span_lint_and_then(
                        cx,
                        IMPL_TRAIT_IN_PARAMS,
                        param.span,
                        "'`impl Trait` used as a function parameter'",
                        |diag| {
                            if let Some(gen_span) = generics.span_for_param_suggestion() {
                                diag.span_suggestion_with_style(
                                    gen_span,
                                    "add a type paremeter",
                                    format!(", {{ /* Generic name */ }}: {}", &param.name.ident().as_str()[5..]),
                                    rustc_errors::Applicability::HasPlaceholders,
                                    rustc_errors::SuggestionStyle::ShowAlways,
                                );
                            } else {
                                diag.span_suggestion_with_style(
                                    Span::new(
                                        body.params[0].span.lo() - rustc_span::BytePos(1),
                                        ident.span.hi(),
                                        ident.span.ctxt(),
                                        ident.span.parent(),
                                    ),
                                    "add a type paremeter",
                                    format!("<{{ /* Generic name */ }}: {}>", &param.name.ident().as_str()[5..]),
                                    rustc_errors::Applicability::HasPlaceholders,
                                    rustc_errors::SuggestionStyle::ShowAlways,
                                );
                            }
                        },
                    );
                }
            }
        }
    }
}
