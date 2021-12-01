use rustc_hir::{self as hir, intravisit, HirIdSet};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::def_id::LocalDefId;

use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::type_is_unsafe_function;
use clippy_utils::{iter_input_pats, path_to_local};

use super::NOT_UNSAFE_PTR_ARG_DEREF;

pub(super) fn check_fn<'tcx>(
    cx: &LateContext<'tcx>,
    kind: intravisit::FnKind<'tcx>,
    decl: &'tcx hir::FnDecl<'tcx>,
    body: &'tcx hir::Body<'tcx>,
    hir_id: hir::HirId,
) {
    let unsafety = match kind {
        intravisit::FnKind::ItemFn(_, _, hir::FnHeader { unsafety, .. }, _) => unsafety,
        intravisit::FnKind::Method(_, sig, _) => sig.header.unsafety,
        intravisit::FnKind::Closure => return,
    };

    check_raw_ptr(cx, unsafety, decl, body, cx.tcx.hir().local_def_id(hir_id));
}

pub(super) fn check_trait_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
    if let hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(eid)) = item.kind {
        let body = cx.tcx.hir().body(eid);
        check_raw_ptr(cx, sig.header.unsafety, sig.decl, body, item.def_id);
    }
}

fn check_raw_ptr<'tcx>(
    cx: &LateContext<'tcx>,
    unsafety: hir::Unsafety,
    decl: &'tcx hir::FnDecl<'tcx>,
    body: &'tcx hir::Body<'tcx>,
    def_id: LocalDefId,
) {
    let expr = &body.value;
    if unsafety == hir::Unsafety::Normal && cx.access_levels.is_exported(def_id) {
        let raw_ptrs = iter_input_pats(decl, body)
            .filter_map(|arg| raw_ptr_arg(cx, arg))
            .collect::<HirIdSet>();

        if !raw_ptrs.is_empty() {
            let typeck_results = cx.tcx.typeck_body(body.id());
            let mut v = DerefVisitor {
                cx,
                ptrs: raw_ptrs,
                typeck_results,
            };

            intravisit::walk_expr(&mut v, expr);
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

struct DerefVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    ptrs: HirIdSet,
    typeck_results: &'a ty::TypeckResults<'tcx>,
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for DerefVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        match expr.kind {
            hir::ExprKind::Call(f, args) => {
                let ty = self.typeck_results.expr_ty(f);

                if type_is_unsafe_function(self.cx, ty) {
                    for arg in args {
                        self.check_arg(arg);
                    }
                }
            },
            hir::ExprKind::MethodCall(_, args, _) => {
                let def_id = self.typeck_results.type_dependent_def_id(expr.hir_id).unwrap();
                let base_type = self.cx.tcx.type_of(def_id);

                if type_is_unsafe_function(self.cx, base_type) {
                    for arg in args {
                        self.check_arg(arg);
                    }
                }
            },
            hir::ExprKind::Unary(hir::UnOp::Deref, ptr) => self.check_arg(ptr),
            _ => (),
        }

        intravisit::walk_expr(self, expr);
    }
}

impl<'a, 'tcx> DerefVisitor<'a, 'tcx> {
    fn check_arg(&self, ptr: &hir::Expr<'_>) {
        if let Some(id) = path_to_local(ptr) {
            if self.ptrs.contains(&id) {
                span_lint(
                    self.cx,
                    NOT_UNSAFE_PTR_ARG_DEREF,
                    ptr.span,
                    "this public function might dereference a raw pointer but is not marked `unsafe`",
                );
            }
        }
    }
}
