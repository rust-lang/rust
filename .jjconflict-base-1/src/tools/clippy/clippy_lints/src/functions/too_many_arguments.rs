use rustc_abi::ExternAbi;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_lint::LateContext;
use rustc_span::Span;

use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_trait_impl_item;

use super::TOO_MANY_ARGUMENTS;

pub(super) fn check_fn(
    cx: &LateContext<'_>,
    kind: FnKind<'_>,
    decl: &hir::FnDecl<'_>,
    hir_id: hir::HirId,
    def_id: LocalDefId,
    too_many_arguments_threshold: u64,
) {
    // don't warn for implementations, it's not their fault
    if !is_trait_impl_item(cx, hir_id)
        // don't lint extern functions decls, it's not their fault either
        && kind.header().is_some_and(|header| header.abi == ExternAbi::Rust)
    {
        check_arg_number(cx, decl, cx.tcx.def_span(def_id), too_many_arguments_threshold);
    }
}

pub(super) fn check_trait_item(cx: &LateContext<'_>, item: &hir::TraitItem<'_>, too_many_arguments_threshold: u64) {
    if let hir::TraitItemKind::Fn(ref sig, _) = item.kind
        // don't lint extern functions decls, it's not their fault
        && sig.header.abi == ExternAbi::Rust
    {
        check_arg_number(
            cx,
            sig.decl,
            item.span.with_hi(sig.decl.output.span().hi()),
            too_many_arguments_threshold,
        );
    }
}

fn check_arg_number(cx: &LateContext<'_>, decl: &hir::FnDecl<'_>, fn_span: Span, too_many_arguments_threshold: u64) {
    let args = decl.inputs.len() as u64;
    if args > too_many_arguments_threshold {
        span_lint(
            cx,
            TOO_MANY_ARGUMENTS,
            fn_span,
            format!("this function has too many arguments ({args}/{too_many_arguments_threshold})"),
        );
    }
}
