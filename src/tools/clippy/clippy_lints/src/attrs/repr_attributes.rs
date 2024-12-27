use rustc_hir::Attribute;
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs;

use super::REPR_PACKED_WITHOUT_ABI;

pub(super) fn check(cx: &LateContext<'_>, item_span: Span, attrs: &[Attribute], msrv: &msrvs::Msrv) {
    if msrv.meets(msrvs::REPR_RUST) {
        check_packed(cx, item_span, attrs);
    }
}

fn check_packed(cx: &LateContext<'_>, item_span: Span, attrs: &[Attribute]) {
    if let Some(items) = attrs.iter().find_map(|attr| {
        if attr.ident().is_some_and(|ident| matches!(ident.name, sym::repr)) {
            attr.meta_item_list()
        } else {
            None
        }
    }) && let Some(packed) = items
        .iter()
        .find(|item| item.ident().is_some_and(|ident| matches!(ident.name, sym::packed)))
        && !items.iter().any(|item| {
            item.ident()
                .is_some_and(|ident| matches!(ident.name, sym::C | sym::Rust))
        })
    {
        span_lint_and_then(
            cx,
            REPR_PACKED_WITHOUT_ABI,
            item_span,
            "item uses `packed` representation without ABI-qualification",
            |diag| {
                diag.warn("unqualified `#[repr(packed)]` defaults to `#[repr(Rust, packed)]`, which has no stable ABI")
                    .help("qualify the desired ABI explicity via `#[repr(C, packed)]` or `#[repr(Rust, packed)]`")
                    .span_label(packed.span(), "`packed` representation set here");
            },
        );
    }
}
