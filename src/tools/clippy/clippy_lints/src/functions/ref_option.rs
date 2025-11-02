use crate::functions::REF_OPTION;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::ty::option_arg_ty;
use clippy_utils::{is_from_proc_macro, is_trait_impl_item};
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{self as hir, FnDecl, HirId};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::{self, Mutability, Ty};
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

fn check_ty<'a>(cx: &LateContext<'a>, param: &hir::Ty<'a>, param_ty: Ty<'a>, fixes: &mut Vec<(Span, String)>) {
    if !param.span.in_external_macro(cx.sess().source_map())
        && let ty::Ref(_, opt_ty, Mutability::Not) = param_ty.kind()
        && let Some(gen_ty) = option_arg_ty(cx, *opt_ty)
        && !gen_ty.is_ref()
        // Need to gen the original spans, so first parsing mid, and hir parsing afterward
        && let hir::TyKind::Ref(lifetime, hir::MutTy { ty, .. }) = param.kind
        && let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = ty.kind
        && let (Some(first), Some(last)) = (path.segments.first(), path.segments.last())
        && let Some(hir::GenericArgs {
            args: [hir::GenericArg::Type(opt_ty)],
            ..
        }) = last.args
        && !is_from_proc_macro(cx, param)
    {
        let lifetime = snippet(cx, lifetime.ident.span, "..");
        fixes.push((
            param.span,
            format!(
                "{}<&{lifetime}{}{}>",
                snippet(cx, first.ident.span.to(last.ident.span), ".."),
                if lifetime.is_empty() { "" } else { " " },
                snippet(cx, opt_ty.span, "..")
            ),
        ));
    }
}

fn check_fn_sig<'a>(cx: &LateContext<'a>, decl: &FnDecl<'a>, span: Span, sig: ty::FnSig<'a>) {
    let mut fixes = Vec::new();
    // Check function arguments' types
    for (param, param_ty) in decl.inputs.iter().zip(sig.inputs()) {
        check_ty(cx, param, *param_ty, &mut fixes);
    }
    // Check return type
    if let hir::FnRetTy::Return(ty) = &decl.output {
        check_ty(cx, ty, sig.output(), &mut fixes);
    }
    if !fixes.is_empty() {
        span_lint_and_then(
            cx,
            REF_OPTION,
            span,
            "it is more idiomatic to use `Option<&T>` instead of `&Option<T>`",
            |diag| {
                diag.multipart_suggestion("change this to", fixes, Applicability::Unspecified);
            },
        );
    }
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn check_fn<'a>(
    cx: &LateContext<'a>,
    kind: FnKind<'a>,
    decl: &FnDecl<'a>,
    span: Span,
    hir_id: HirId,
    def_id: LocalDefId,
    body: &hir::Body<'a>,
    avoid_breaking_exported_api: bool,
) {
    if avoid_breaking_exported_api && cx.effective_visibilities.is_exported(def_id) {
        return;
    }
    if span.in_external_macro(cx.sess().source_map()) {
        return;
    }

    if let FnKind::Closure = kind {
        // Compute the span of the closure parameters + return type if set
        let inputs_output_span = if let hir::FnRetTy::Return(out_ty) = &decl.output {
            if decl.inputs.is_empty() {
                out_ty.span
            } else {
                span.with_hi(out_ty.span.hi())
            }
        } else if let (Some(first), Some(last)) = (decl.inputs.first(), decl.inputs.last()) {
            first.span.to(last.span)
        } else {
            // No parameters - no point in checking
            return;
        };

        // Figure out the signature of the closure
        let ty::Closure(_, args) = cx.typeck_results().expr_ty(body.value).kind() else {
            return;
        };
        let sig = args.as_closure().sig().skip_binder();

        if is_from_proc_macro(cx, &(&kind, body, hir_id, span)) {
            return;
        }

        check_fn_sig(cx, decl, inputs_output_span, sig);
    } else if !is_trait_impl_item(cx, hir_id) {
        let sig = cx.tcx.fn_sig(def_id).instantiate_identity().skip_binder();

        if is_from_proc_macro(cx, &(&kind, body, hir_id, span)) {
            return;
        }

        check_fn_sig(cx, decl, span, sig);
    }
}

pub(super) fn check_trait_item<'a>(
    cx: &LateContext<'a>,
    trait_item: &hir::TraitItem<'a>,
    avoid_breaking_exported_api: bool,
) {
    if !trait_item.span.in_external_macro(cx.sess().source_map())
        && let hir::TraitItemKind::Fn(ref sig, _) = trait_item.kind
        && !(avoid_breaking_exported_api && cx.effective_visibilities.is_exported(trait_item.owner_id.def_id))
        && !is_from_proc_macro(cx, trait_item)
    {
        let def_id = trait_item.owner_id.def_id;
        let ty_sig = cx.tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        check_fn_sig(cx, sig.decl, sig.span, ty_sig);
    }
}
