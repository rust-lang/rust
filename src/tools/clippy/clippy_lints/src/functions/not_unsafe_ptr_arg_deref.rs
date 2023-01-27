use rustc_hir::{self as hir, intravisit, HirIdSet};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::def_id::LocalDefId;

use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::type_is_unsafe_function;
use clippy_utils::visitors::for_each_expr_with_closures;
use clippy_utils::{iter_input_pats, path_to_local};

use core::ops::ControlFlow;

use super::NOT_UNSAFE_PTR_ARG_DEREF;

pub(super) fn check_fn<'tcx>(
    cx: &LateContext<'tcx>,
    kind: intravisit::FnKind<'tcx>,
    decl: &'tcx hir::FnDecl<'tcx>,
    body: &'tcx hir::Body<'tcx>,
    hir_id: hir::HirId,
) {
    let unsafety = match kind {
        intravisit::FnKind::ItemFn(_, _, hir::FnHeader { unsafety, .. }) => unsafety,
        intravisit::FnKind::Method(_, sig) => sig.header.unsafety,
        intravisit::FnKind::Closure => return,
    };

    check_raw_ptr(cx, unsafety, decl, body, cx.tcx.hir().local_def_id(hir_id));
}

pub(super) fn check_trait_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
    if let hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(eid)) = item.kind {
        let body = cx.tcx.hir().body(eid);
        check_raw_ptr(cx, sig.header.unsafety, sig.decl, body, item.owner_id.def_id);
    }
}

fn check_raw_ptr<'tcx>(
    cx: &LateContext<'tcx>,
    unsafety: hir::Unsafety,
    decl: &'tcx hir::FnDecl<'tcx>,
    body: &'tcx hir::Body<'tcx>,
    def_id: LocalDefId,
) {
    if unsafety == hir::Unsafety::Normal && cx.effective_visibilities.is_exported(def_id) {
        let raw_ptrs = iter_input_pats(decl, body)
            .filter_map(|arg| raw_ptr_arg(cx, arg))
            .collect::<HirIdSet>();

        if !raw_ptrs.is_empty() {
            let typeck = cx.tcx.typeck_body(body.id());
            let _: Option<!> = for_each_expr_with_closures(cx, body.value, |e| {
                match e.kind {
                    hir::ExprKind::Call(f, args) if type_is_unsafe_function(cx, typeck.expr_ty(f)) => {
                        for arg in args {
                            check_arg(cx, &raw_ptrs, arg);
                        }
                    },
                    hir::ExprKind::MethodCall(_, recv, args, _) => {
                        let def_id = typeck.type_dependent_def_id(e.hir_id).unwrap();
                        if cx.tcx.fn_sig(def_id).skip_binder().skip_binder().unsafety == hir::Unsafety::Unsafe {
                            check_arg(cx, &raw_ptrs, recv);
                            for arg in args {
                                check_arg(cx, &raw_ptrs, arg);
                            }
                        }
                    },
                    hir::ExprKind::Unary(hir::UnOp::Deref, ptr) => check_arg(cx, &raw_ptrs, ptr),
                    _ => (),
                }
                ControlFlow::Continue(())
            });
        }
    }
}

fn raw_ptr_arg(cx: &LateContext<'_>, arg: &hir::Param<'_>) -> Option<hir::HirId> {
    if let (&hir::PatKind::Binding(_, id, _, _), Some(&ty::RawPtr(_))) = (
        &arg.pat.kind,
        cx.maybe_typeck_results()
            .map(|typeck_results| typeck_results.pat_ty(arg.pat).kind()),
    ) {
        Some(id)
    } else {
        None
    }
}

fn check_arg(cx: &LateContext<'_>, raw_ptrs: &HirIdSet, arg: &hir::Expr<'_>) {
    if path_to_local(arg).map_or(false, |id| raw_ptrs.contains(&id)) {
        span_lint(
            cx,
            NOT_UNSAFE_PTR_ARG_DEREF,
            arg.span,
            "this public function might dereference a raw pointer but is not marked `unsafe`",
        );
    }
}
