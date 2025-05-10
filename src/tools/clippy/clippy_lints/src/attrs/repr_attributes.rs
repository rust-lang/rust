use rustc_attr_data_structures::{AttributeKind, ReprAttr, find_attr};
use rustc_hir::Attribute;
use rustc_lint::LateContext;
use rustc_span::Span;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};

use super::REPR_PACKED_WITHOUT_ABI;

pub(super) fn check(cx: &LateContext<'_>, item_span: Span, attrs: &[Attribute], msrv: Msrv) {
    if let Some(reprs) = find_attr!(attrs, AttributeKind::Repr(r) => r) {
        let packed_span = reprs
            .iter()
            .find(|(r, _)| matches!(r, ReprAttr::ReprPacked(..)))
            .map(|(_, s)| *s);

        if let Some(packed_span) = packed_span
            && !reprs
                .iter()
                .any(|(x, _)| *x == ReprAttr::ReprC || *x == ReprAttr::ReprRust)
            && msrv.meets(cx, msrvs::REPR_RUST)
        {
            span_lint_and_then(
                cx,
                REPR_PACKED_WITHOUT_ABI,
                item_span,
                "item uses `packed` representation without ABI-qualification",
                |diag| {
                    diag.warn(
                        "unqualified `#[repr(packed)]` defaults to `#[repr(Rust, packed)]`, which has no stable ABI",
                    )
                    .help("qualify the desired ABI explicitly via `#[repr(C, packed)]` or `#[repr(Rust, packed)]`")
                    .span_label(packed_span, "`packed` representation set here");
                },
            );
        }
    }
}
