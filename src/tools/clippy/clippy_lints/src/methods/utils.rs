use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{get_parent_expr, path_to_local_id, usage};
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{BorrowKind, Expr, ExprKind, HirId, Mutability, Pat, QPath, Stmt, StmtKind};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_span::symbol::sym;

/// Checks if `expr`, of type `ty`, corresponds to a slice or can be dereferenced to a slice, or if
/// `expr` is a method call to `.iter()` on such a type. In these cases, return the slice-like
/// expression.
pub(super) fn derefs_to_slice<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    ty: Ty<'tcx>,
) -> Option<&'tcx Expr<'tcx>> {
    fn may_slice<'a>(cx: &LateContext<'a>, ty: Ty<'a>) -> bool {
        match ty.kind() {
            ty::Slice(_) => true,
            ty::Adt(..) if let Some(boxed) = ty.boxed_ty() => may_slice(cx, boxed),
            ty::Adt(..) => is_type_diagnostic_item(cx, ty, sym::Vec),
            ty::Array(_, size) => size.try_to_target_usize(cx.tcx).is_some(),
            ty::Ref(_, inner, _) => may_slice(cx, *inner),
            _ => false,
        }
    }

    if let ExprKind::MethodCall(path, self_arg, ..) = &expr.kind {
        if path.ident.name == sym::iter && may_slice(cx, cx.typeck_results().expr_ty(self_arg)) {
            Some(self_arg)
        } else {
            None
        }
    } else {
        match ty.kind() {
            ty::Slice(_) => Some(expr),
            _ if ty.boxed_ty().is_some_and(|boxed| may_slice(cx, boxed)) => Some(expr),
            ty::Ref(_, inner, _) => {
                if may_slice(cx, *inner) {
                    Some(expr)
                } else {
                    None
                }
            },
            _ => None,
        }
    }
}

/// The core logic of `check_for_loop_iter` in `unnecessary_iter_cloned.rs`, this function wraps a
/// use of `CloneOrCopyVisitor`.
pub(super) fn clone_or_copy_needed<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &Pat<'tcx>,
    body: &'tcx Expr<'tcx>,
) -> (bool, Vec<(Span, String)>) {
    let mut visitor = CloneOrCopyVisitor {
        cx,
        binding_hir_ids: pat_bindings(pat),
        clone_or_copy_needed: false,
        references_to_binding: Vec::new(),
    };
    visitor.visit_expr(body);
    (visitor.clone_or_copy_needed, visitor.references_to_binding)
}

/// Returns a vector of all `HirId`s bound by the pattern.
fn pat_bindings(pat: &Pat<'_>) -> Vec<HirId> {
    let mut collector = usage::ParamBindingIdCollector {
        binding_hir_ids: Vec::new(),
    };
    collector.visit_pat(pat);
    collector.binding_hir_ids
}

/// `clone_or_copy_needed` will be false when `CloneOrCopyVisitor` is done visiting if the only
/// operations performed on `binding_hir_ids` are:
/// * to take non-mutable references to them
/// * to use them as non-mutable `&self` in method calls
///
/// If any of `binding_hir_ids` is used in any other way, then `clone_or_copy_needed` will be true
/// when `CloneOrCopyVisitor` is done visiting.
struct CloneOrCopyVisitor<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    binding_hir_ids: Vec<HirId>,
    clone_or_copy_needed: bool,
    references_to_binding: Vec<(Span, String)>,
}

impl<'tcx> Visitor<'tcx> for CloneOrCopyVisitor<'_, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        walk_expr(self, expr);
        if self.is_binding(expr) {
            if let Some(parent) = get_parent_expr(self.cx, expr) {
                match parent.kind {
                    ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, referent) => {
                        if !parent.span.from_expansion() {
                            self.references_to_binding
                                .push((parent.span.until(referent.span), String::new()));
                        }
                        return;
                    },
                    ExprKind::MethodCall(.., args, _) => {
                        if args.iter().all(|arg| !self.is_binding(arg))
                            && let Some(method_def_id) = self.cx.typeck_results().type_dependent_def_id(parent.hir_id)
                            && let method_ty = self.cx.tcx.type_of(method_def_id).instantiate_identity()
                            && let self_ty = method_ty.fn_sig(self.cx.tcx).input(0).skip_binder()
                            && matches!(self_ty.kind(), ty::Ref(_, _, Mutability::Not))
                        {
                            return;
                        }
                    },
                    _ => {},
                }
            }
            self.clone_or_copy_needed = true;
        }
    }
}

impl<'tcx> CloneOrCopyVisitor<'_, 'tcx> {
    fn is_binding(&self, expr: &Expr<'tcx>) -> bool {
        self.binding_hir_ids
            .iter()
            .any(|hir_id| path_to_local_id(expr, *hir_id))
    }
}

pub(super) fn get_last_chain_binding_hir_id(mut hir_id: HirId, statements: &[Stmt<'_>]) -> Option<HirId> {
    for stmt in statements {
        if let StmtKind::Let(local) = stmt.kind
            && let Some(init) = local.init
            && let ExprKind::Path(QPath::Resolved(_, path)) = init.kind
            && let rustc_hir::def::Res::Local(local_hir_id) = path.res
            && local_hir_id == hir_id
        {
            hir_id = local.pat.hir_id;
        } else {
            return None;
        }
    }
    Some(hir_id)
}
