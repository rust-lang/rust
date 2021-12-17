use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::ForLoop;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{get_associated_type, get_iterator_item_ty, implements_trait};
use clippy_utils::{fn_def_id, get_parent_expr, path_to_local_id, usage};
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{def_id::DefId, BorrowKind, Expr, ExprKind, HirId, LangItem, Mutability, Pat};
use rustc_lint::LateContext;
use rustc_middle::{hir::map::Map, ty};
use rustc_span::{sym, Symbol};

use super::UNNECESSARY_TO_OWNED;

pub fn check(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, method_name: Symbol, receiver: &'tcx Expr<'tcx>) -> bool {
    if_chain! {
        if let Some(parent) = get_parent_expr(cx, expr);
        if let Some(callee_def_id) = fn_def_id(cx, parent);
        if is_into_iter(cx, callee_def_id);
        then {
            check_for_loop_iter(cx, parent, method_name, receiver)
        } else {
            false
        }
    }
}

/// Checks whether `expr` is an iterator in a `for` loop and, if so, determines whether the
/// iterated-over items could be iterated over by reference. The reason why `check` above does not
/// include this code directly is so that it can be called from
/// `unnecessary_into_owned::check_into_iter_call_arg`.
pub fn check_for_loop_iter(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    method_name: Symbol,
    receiver: &'tcx Expr<'tcx>,
) -> bool {
    if_chain! {
        if let Some(grandparent) = get_parent_expr(cx, expr).and_then(|parent| get_parent_expr(cx, parent));
        if let Some(ForLoop { pat, body, .. }) = ForLoop::hir(grandparent);
        let (clone_or_copy_needed, addr_of_exprs) = clone_or_copy_needed(cx, pat, body);
        if !clone_or_copy_needed;
        if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
        then {
            let snippet = if_chain! {
                if let ExprKind::MethodCall(maybe_iter_method_name, _, [collection], _) = receiver.kind;
                if maybe_iter_method_name.ident.name == sym::iter;

                if let Some(iterator_trait_id) = cx.tcx.get_diagnostic_item(sym::Iterator);
                let receiver_ty = cx.typeck_results().expr_ty(receiver);
                if implements_trait(cx, receiver_ty, iterator_trait_id, &[]);
                if let Some(iter_item_ty) = get_iterator_item_ty(cx, receiver_ty);

                if let Some(into_iterator_trait_id) = cx.tcx.get_diagnostic_item(sym::IntoIterator);
                let collection_ty = cx.typeck_results().expr_ty(collection);
                if implements_trait(cx, collection_ty, into_iterator_trait_id, &[]);
                if let Some(into_iter_item_ty) = get_associated_type(cx, collection_ty, into_iterator_trait_id, "Item");

                if iter_item_ty == into_iter_item_ty;
                if let Some(collection_snippet) = snippet_opt(cx, collection.span);
                then {
                    collection_snippet
                } else {
                    receiver_snippet
                }
            };
            span_lint_and_then(
                cx,
                UNNECESSARY_TO_OWNED,
                expr.span,
                &format!("unnecessary use of `{}`", method_name),
                |diag| {
                    diag.span_suggestion(expr.span, "use", snippet, Applicability::MachineApplicable);
                    for addr_of_expr in addr_of_exprs {
                        match addr_of_expr.kind {
                            ExprKind::AddrOf(_, _, referent) => {
                                let span = addr_of_expr.span.with_hi(referent.span.lo());
                                diag.span_suggestion(span, "remove this `&`", String::new(), Applicability::MachineApplicable);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            );
            return true;
        }
    }
    false
}

/// The core logic of `check_for_loop_iter` above, this function wraps a use of
/// `CloneOrCopyVisitor`.
fn clone_or_copy_needed(
    cx: &LateContext<'tcx>,
    pat: &Pat<'tcx>,
    body: &'tcx Expr<'tcx>,
) -> (bool, Vec<&'tcx Expr<'tcx>>) {
    let mut visitor = CloneOrCopyVisitor {
        cx,
        binding_hir_ids: pat_bindings(pat),
        clone_or_copy_needed: false,
        addr_of_exprs: Vec::new(),
    };
    visitor.visit_expr(body);
    (visitor.clone_or_copy_needed, visitor.addr_of_exprs)
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
/// If any of `binding_hir_ids` is used in any other way, then `clone_or_copy_needed` will be true
/// when `CloneOrCopyVisitor` is done visiting.
struct CloneOrCopyVisitor<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    binding_hir_ids: Vec<HirId>,
    clone_or_copy_needed: bool,
    addr_of_exprs: Vec<&'tcx Expr<'tcx>>,
}

impl<'cx, 'tcx> Visitor<'tcx> for CloneOrCopyVisitor<'cx, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        walk_expr(self, expr);
        if self.is_binding(expr) {
            if let Some(parent) = get_parent_expr(self.cx, expr) {
                match parent.kind {
                    ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, _) => {
                        self.addr_of_exprs.push(parent);
                        return;
                    },
                    ExprKind::MethodCall(_, _, args, _) => {
                        if_chain! {
                            if args.iter().skip(1).all(|arg| !self.is_binding(arg));
                            if let Some(method_def_id) = self.cx.typeck_results().type_dependent_def_id(parent.hir_id);
                            let method_ty = self.cx.tcx.type_of(method_def_id);
                            let self_ty = method_ty.fn_sig(self.cx.tcx).input(0).skip_binder();
                            if matches!(self_ty.kind(), ty::Ref(_, _, Mutability::Not));
                            then {
                                return;
                            }
                        }
                    },
                    _ => {},
                }
            }
            self.clone_or_copy_needed = true;
        }
    }
}

impl<'cx, 'tcx> CloneOrCopyVisitor<'cx, 'tcx> {
    fn is_binding(&self, expr: &Expr<'tcx>) -> bool {
        self.binding_hir_ids
            .iter()
            .any(|hir_id| path_to_local_id(expr, *hir_id))
    }
}

/// Returns true if the named method is `IntoIterator::into_iter`.
pub fn is_into_iter(cx: &LateContext<'_>, callee_def_id: DefId) -> bool {
    cx.tcx.lang_items().require(LangItem::IntoIterIntoIter) == Ok(callee_def_id)
}
