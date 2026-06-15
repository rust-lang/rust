use clippy_utils::res::MaybeResPath;
use rustc_hir::{self as hir, HirId, HirIdSet, intravisit};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::def_id::LocalDefId;

use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::is_unsafe_fn;
use clippy_utils::visitors::for_each_expr;

use core::ops::ControlFlow;

use super::NOT_UNSAFE_PTR_ARG_DEREF;

pub(super) fn check_fn<'tcx>(
    cx: &LateContext<'tcx>,
    kind: intravisit::FnKind<'tcx>,
    _decl: &'tcx hir::FnDecl<'tcx>,
    body: &'tcx hir::Body<'tcx>,
    def_id: LocalDefId,
) {
    let safety = match kind {
        intravisit::FnKind::ItemFn(_, _, header) => header.safety(),
        intravisit::FnKind::Method(_, sig) => sig.header.safety(),
        intravisit::FnKind::Closure => return,
    };

    check_raw_ptr(cx, safety, body, def_id);
}

pub(super) fn check_trait_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
    if let hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(eid)) = item.kind {
        let body = cx.tcx.hir_body(eid);
        check_raw_ptr(cx, sig.header.safety(), body, item.owner_id.def_id);
    }
}

fn check_raw_ptr<'tcx>(
    cx: &LateContext<'tcx>,
    safety: hir::Safety,
    body: &'tcx hir::Body<'tcx>,
    def_id: LocalDefId,
) {
    if safety.is_safe() && cx.effective_visibilities.is_exported(def_id) {
        let sig = cx.tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        let raw_ptrs = body.params.iter().zip(sig.inputs_and_output)
            .filter_map(|(param, ty)| raw_ptr_arg(param, ty))
            .collect::<HirIdSet>();

        if !raw_ptrs.is_empty() {
            let typeck = cx.tcx.typeck_body(body.id());
            let _: Option<!> = for_each_expr(cx.tcx, body.value, |e| {
                match e.kind {
                    hir::ExprKind::Call(f, args) if is_unsafe_fn(cx, typeck.expr_ty(f)) => {
                        for arg in args {
                            check_arg(cx, &raw_ptrs, arg);
                        }
                    },
                    hir::ExprKind::MethodCall(_, recv, args, _) => {
                        let def_id = typeck.type_dependent_def_id(e.hir_id).unwrap();
                        if cx.tcx.fn_sig(def_id).skip_binder().skip_binder().safety().is_unsafe() {
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

fn raw_ptr_arg(param: &hir::Param<'_>, ty: Ty<'_>) -> Option<HirId> {
    if let (&hir::PatKind::Binding(_, id, _, _), &ty::RawPtr(_, _)) = (
        &param.pat.kind,
        ty.kind(),
    ) {
        Some(id)
    } else {
        None
    }
}

fn check_arg(cx: &LateContext<'_>, raw_ptrs: &HirIdSet, arg: &hir::Expr<'_>) {
    if arg.res_local_id().is_some_and(|id| raw_ptrs.contains(&id)) {
        span_lint(
            cx,
            NOT_UNSAFE_PTR_ARG_DEREF,
            arg.span,
            "this public function might dereference a raw pointer but is not marked `unsafe`",
        );
    }
}
