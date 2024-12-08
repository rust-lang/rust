use hir::FnSig;
use rustc_ast::ast::Attribute;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefIdSet;
use rustc_hir::{self as hir, QPath};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, sym};

use clippy_utils::attrs::is_proc_macro;
use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::is_must_use_ty;
use clippy_utils::visitors::for_each_expr_without_closures;
use clippy_utils::{return_ty, trait_ref_of_method};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;

use core::ops::ControlFlow;

use super::{DOUBLE_MUST_USE, MUST_USE_CANDIDATE, MUST_USE_UNIT};

pub(super) fn check_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
    let attrs = cx.tcx.hir().attrs(item.hir_id());
    let attr = cx.tcx.get_attr(item.owner_id, sym::must_use);
    if let hir::ItemKind::Fn(ref sig, _generics, ref body_id) = item.kind {
        let is_public = cx.effective_visibilities.is_exported(item.owner_id.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
        if let Some(attr) = attr {
            check_needless_must_use(cx, sig.decl, item.owner_id, item.span, fn_header_span, attr, sig);
        } else if is_public && !is_proc_macro(attrs) && !attrs.iter().any(|a| a.has_name(sym::no_mangle)) {
            check_must_use_candidate(
                cx,
                sig.decl,
                cx.tcx.hir().body(*body_id),
                item.span,
                item.owner_id,
                item.span.with_hi(sig.decl.output.span().hi()),
                "this function could have a `#[must_use]` attribute",
            );
        }
    }
}

pub(super) fn check_impl_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
    if let hir::ImplItemKind::Fn(ref sig, ref body_id) = item.kind {
        let is_public = cx.effective_visibilities.is_exported(item.owner_id.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
        let attrs = cx.tcx.hir().attrs(item.hir_id());
        let attr = cx.tcx.get_attr(item.owner_id, sym::must_use);
        if let Some(attr) = attr {
            check_needless_must_use(cx, sig.decl, item.owner_id, item.span, fn_header_span, attr, sig);
        } else if is_public && !is_proc_macro(attrs) && trait_ref_of_method(cx, item.owner_id.def_id).is_none() {
            check_must_use_candidate(
                cx,
                sig.decl,
                cx.tcx.hir().body(*body_id),
                item.span,
                item.owner_id,
                item.span.with_hi(sig.decl.output.span().hi()),
                "this method could have a `#[must_use]` attribute",
            );
        }
    }
}

pub(super) fn check_trait_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
    if let hir::TraitItemKind::Fn(ref sig, ref eid) = item.kind {
        let is_public = cx.effective_visibilities.is_exported(item.owner_id.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());

        let attrs = cx.tcx.hir().attrs(item.hir_id());
        let attr = cx.tcx.get_attr(item.owner_id, sym::must_use);
        if let Some(attr) = attr {
            check_needless_must_use(cx, sig.decl, item.owner_id, item.span, fn_header_span, attr, sig);
        } else if let hir::TraitFn::Provided(eid) = *eid {
            let body = cx.tcx.hir().body(eid);
            if attr.is_none() && is_public && !is_proc_macro(attrs) {
                check_must_use_candidate(
                    cx,
                    sig.decl,
                    body,
                    item.span,
                    item.owner_id,
                    item.span.with_hi(sig.decl.output.span().hi()),
                    "this method could have a `#[must_use]` attribute",
                );
            }
        }
    }
}

fn check_needless_must_use(
    cx: &LateContext<'_>,
    decl: &hir::FnDecl<'_>,
    item_id: hir::OwnerId,
    item_span: Span,
    fn_header_span: Span,
    attr: &Attribute,
    sig: &FnSig<'_>,
) {
    if in_external_macro(cx.sess(), item_span) {
        return;
    }
    if returns_unit(decl) {
        span_lint_and_then(
            cx,
            MUST_USE_UNIT,
            fn_header_span,
            "this unit-returning function has a `#[must_use]` attribute",
            |diag| {
                diag.span_suggestion(attr.span, "remove the attribute", "", Applicability::MachineApplicable);
            },
        );
    } else if attr.value_str().is_none() && is_must_use_ty(cx, return_ty(cx, item_id)) {
        // Ignore async functions unless Future::Output type is a must_use type
        if sig.header.is_async() {
            let infcx = cx.tcx.infer_ctxt().build(cx.typing_mode());
            if let Some(future_ty) = infcx.err_ctxt().get_impl_future_output_ty(return_ty(cx, item_id))
                && !is_must_use_ty(cx, future_ty)
            {
                return;
            };
        }

        span_lint_and_help(
            cx,
            DOUBLE_MUST_USE,
            fn_header_span,
            "this function has a `#[must_use]` attribute with no message, but returns a type already marked as `#[must_use]`",
            None,
            "either add some descriptive message or remove the attribute",
        );
    }
}

fn check_must_use_candidate<'tcx>(
    cx: &LateContext<'tcx>,
    decl: &'tcx hir::FnDecl<'_>,
    body: &'tcx hir::Body<'_>,
    item_span: Span,
    item_id: hir::OwnerId,
    fn_span: Span,
    msg: &'static str,
) {
    if has_mutable_arg(cx, body)
        || mutates_static(cx, body)
        || in_external_macro(cx.sess(), item_span)
        || returns_unit(decl)
        || !cx.effective_visibilities.is_exported(item_id.def_id)
        || is_must_use_ty(cx, return_ty(cx, item_id))
    {
        return;
    }
    span_lint_and_then(cx, MUST_USE_CANDIDATE, fn_span, msg, |diag| {
        if let Some(snippet) = fn_span.get_source_text(cx) {
            diag.span_suggestion(
                fn_span,
                "add the attribute",
                format!("#[must_use] {snippet}"),
                Applicability::MachineApplicable,
            );
        }
    });
}

fn returns_unit(decl: &hir::FnDecl<'_>) -> bool {
    match decl.output {
        hir::FnRetTy::DefaultReturn(_) => true,
        hir::FnRetTy::Return(ty) => match ty.kind {
            hir::TyKind::Tup(tys) => tys.is_empty(),
            hir::TyKind::Never => true,
            _ => false,
        },
    }
}

fn has_mutable_arg(cx: &LateContext<'_>, body: &hir::Body<'_>) -> bool {
    let mut tys = DefIdSet::default();
    body.params.iter().any(|param| is_mutable_pat(cx, param.pat, &mut tys))
}

fn is_mutable_pat(cx: &LateContext<'_>, pat: &hir::Pat<'_>, tys: &mut DefIdSet) -> bool {
    if let hir::PatKind::Wild = pat.kind {
        return false; // ignore `_` patterns
    }
    if cx.tcx.has_typeck_results(pat.hir_id.owner.def_id) {
        is_mutable_ty(cx, cx.tcx.typeck(pat.hir_id.owner.def_id).pat_ty(pat), tys)
    } else {
        false
    }
}

fn is_mutable_ty<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, tys: &mut DefIdSet) -> bool {
    match *ty.kind() {
        // primitive types are never mutable
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Str => false,
        ty::Adt(adt, args) => {
            tys.insert(adt.did()) && !ty.is_freeze(cx.tcx, cx.typing_env())
                || matches!(cx.tcx.get_diagnostic_name(adt.did()), Some(sym::Rc | sym::Arc))
                    && args.types().any(|ty| is_mutable_ty(cx, ty, tys))
        },
        ty::Tuple(args) => args.iter().any(|ty| is_mutable_ty(cx, ty, tys)),
        ty::Array(ty, _) | ty::Slice(ty) => is_mutable_ty(cx, ty, tys),
        ty::RawPtr(ty, mutbl) | ty::Ref(_, ty, mutbl) => mutbl == hir::Mutability::Mut || is_mutable_ty(cx, ty, tys),
        // calling something constitutes a side effect, so return true on all callables
        // also never calls need not be used, so return true for them, too
        _ => true,
    }
}

fn is_mutated_static(e: &hir::Expr<'_>) -> bool {
    use hir::ExprKind::{Field, Index, Path};

    match e.kind {
        Path(QPath::Resolved(_, path)) => !matches!(path.res, Res::Local(_)),
        Path(_) => true,
        Field(inner, _) | Index(inner, _, _) => is_mutated_static(inner),
        _ => false,
    }
}

fn mutates_static<'tcx>(cx: &LateContext<'tcx>, body: &'tcx hir::Body<'_>) -> bool {
    for_each_expr_without_closures(body.value, |e| {
        use hir::ExprKind::{AddrOf, Assign, AssignOp, Call, MethodCall};

        match e.kind {
            Call(_, args) => {
                let mut tys = DefIdSet::default();
                for arg in args {
                    if cx.tcx.has_typeck_results(arg.hir_id.owner.def_id)
                        && is_mutable_ty(cx, cx.tcx.typeck(arg.hir_id.owner.def_id).expr_ty(arg), &mut tys)
                        && is_mutated_static(arg)
                    {
                        return ControlFlow::Break(());
                    }
                    tys.clear();
                }
                ControlFlow::Continue(())
            },
            MethodCall(_, receiver, args, _) => {
                let mut tys = DefIdSet::default();
                for arg in std::iter::once(receiver).chain(args.iter()) {
                    if cx.tcx.has_typeck_results(arg.hir_id.owner.def_id)
                        && is_mutable_ty(cx, cx.tcx.typeck(arg.hir_id.owner.def_id).expr_ty(arg), &mut tys)
                        && is_mutated_static(arg)
                    {
                        return ControlFlow::Break(());
                    }
                    tys.clear();
                }
                ControlFlow::Continue(())
            },
            Assign(target, ..) | AssignOp(_, target, _) | AddrOf(_, hir::Mutability::Mut, target)
                if is_mutated_static(target) =>
            {
                ControlFlow::Break(())
            },
            _ => ControlFlow::Continue(()),
        }
    })
    .is_some()
}
