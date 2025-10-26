use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{fn_def_id, is_from_proc_macro, is_lint_allowed};
use hir::intravisit::{Visitor, walk_expr};
use rustc_ast::Label;
use rustc_errors::Applicability;
use rustc_hir::{
    self as hir, Closure, ClosureKind, CoroutineDesugaring, CoroutineKind, CoroutineSource, Expr, ExprKind, FnRetTy,
    FnSig, Node, TyKind,
};
use rustc_lint::{LateContext, LintContext};
use rustc_span::sym;

use super::INFINITE_LOOP;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    loop_block: &'tcx hir::Block<'_>,
    label: Option<Label>,
) {
    if is_lint_allowed(cx, INFINITE_LOOP, expr.hir_id) {
        return;
    }

    // Skip check if this loop is not in a function/method/closure. (In some weird case)
    let Some(parent_fn_ret) = get_parent_fn_ret_ty(cx, expr) else {
        return;
    };
    // Or, its parent function is already returning `Never`
    if is_never_return(parent_fn_ret) {
        return;
    }

    if is_inside_unawaited_async_block(cx, expr) {
        return;
    }

    if expr.span.in_external_macro(cx.sess().source_map()) || is_from_proc_macro(cx, expr) {
        return;
    }

    let mut loop_visitor = LoopVisitor {
        cx,
        label,
        inner_labels: label.into_iter().collect(),
        loop_depth: 0,
        is_finite: false,
    };
    loop_visitor.visit_block(loop_block);

    let is_finite_loop = loop_visitor.is_finite;

    if !is_finite_loop {
        span_lint_and_then(cx, INFINITE_LOOP, expr.span, "infinite loop detected", |diag| {
            if let FnRetTy::DefaultReturn(ret_span) = parent_fn_ret {
                diag.span_suggestion(
                    ret_span,
                    "if this is intentional, consider specifying `!` as function return",
                    " -> !",
                    Applicability::MaybeIncorrect,
                );
            } else {
                diag.help("if this is not intended, try adding a `break` or `return` condition in the loop");
            }
        });
    }
}

/// Check if the given expression is inside an async block that is not being awaited.
/// This helps avoid false positives when async blocks are spawned or assigned to variables.
fn is_inside_unawaited_async_block(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let current_hir_id = expr.hir_id;
    for (_, parent_node) in cx.tcx.hir_parent_iter(current_hir_id) {
        if let Node::Expr(Expr {
            kind:
                ExprKind::Closure(Closure {
                    kind:
                        ClosureKind::Coroutine(CoroutineKind::Desugared(
                            CoroutineDesugaring::Async,
                            CoroutineSource::Block | CoroutineSource::Closure,
                        )),
                    ..
                }),
            ..
        }) = parent_node
        {
            return !is_async_block_awaited(cx, expr);
        }
    }
    false
}

fn is_async_block_awaited(cx: &LateContext<'_>, async_expr: &Expr<'_>) -> bool {
    for (_, parent_node) in cx.tcx.hir_parent_iter(async_expr.hir_id) {
        if let Node::Expr(Expr {
            kind: ExprKind::Match(_, _, hir::MatchSource::AwaitDesugar),
            ..
        }) = parent_node
        {
            return true;
        }
    }
    false
}

fn get_parent_fn_ret_ty<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>) -> Option<FnRetTy<'tcx>> {
    for (_, parent_node) in cx.tcx.hir_parent_iter(expr.hir_id) {
        match parent_node {
            // Skip `Coroutine` closures, these are the body of `async fn`, not async closures.
            // This is because we still need to backtrack one parent node to get the `OpaqueDef` ty.
            Node::Expr(Expr {
                kind:
                    ExprKind::Closure(Closure {
                        kind: ClosureKind::Coroutine(_),
                        ..
                    }),
                ..
            }) => (),
            Node::Item(hir::Item {
                kind:
                    hir::ItemKind::Fn {
                        sig: FnSig { decl, .. },
                        ..
                    },
                ..
            })
            | Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(FnSig { decl, .. }, _),
                ..
            })
            | Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(FnSig { decl, .. }, _),
                ..
            })
            | Node::Expr(Expr {
                kind: ExprKind::Closure(Closure { fn_decl: decl, .. }),
                ..
            }) => return Some(decl.output),
            _ => (),
        }
    }
    None
}

struct LoopVisitor<'hir, 'tcx> {
    cx: &'hir LateContext<'tcx>,
    label: Option<Label>,
    inner_labels: Vec<Label>,
    loop_depth: usize,
    is_finite: bool,
}

impl<'hir> Visitor<'hir> for LoopVisitor<'hir, '_> {
    fn visit_expr(&mut self, ex: &'hir Expr<'_>) {
        match &ex.kind {
            ExprKind::Break(hir::Destination { label, .. }, ..) => {
                // Assuming breaks the loop when `loop_depth` is 0,
                // as it could only means this `break` breaks current loop or any of its upper loop.
                // Or, the depth is not zero but the label is matched.
                if self.loop_depth == 0 || (label.is_some() && *label == self.label) {
                    self.is_finite = true;
                }
            },
            ExprKind::Continue(hir::Destination { label, .. }) => {
                // Check whether we are leaving this loop by continuing into an outer loop
                // whose label we did not encounter.
                if label.is_some_and(|label| !self.inner_labels.contains(&label)) {
                    self.is_finite = true;
                }
            },
            ExprKind::Ret(..) => self.is_finite = true,
            ExprKind::Loop(_, label, _, _) => {
                if let Some(label) = label {
                    self.inner_labels.push(*label);
                }
                self.loop_depth += 1;
                walk_expr(self, ex);
                self.loop_depth -= 1;
                if label.is_some() {
                    self.inner_labels.pop();
                }
            },
            _ => {
                // Calls to a function that never return
                if let Some(did) = fn_def_id(self.cx, ex) {
                    let fn_ret_ty = self.cx.tcx.fn_sig(did).skip_binder().output().skip_binder();
                    if fn_ret_ty.is_never() {
                        self.is_finite = true;
                        return;
                    }
                }
                walk_expr(self, ex);
            },
        }
    }
}

/// Return `true` if the given [`FnRetTy`] is never (!).
///
/// Note: This function also take care of return type of async fn,
/// as the actual type is behind an [`OpaqueDef`](TyKind::OpaqueDef).
fn is_never_return(ret_ty: FnRetTy<'_>) -> bool {
    let FnRetTy::Return(hir_ty) = ret_ty else { return false };

    match hir_ty.kind {
        TyKind::Never => true,
        TyKind::OpaqueDef(hir::OpaqueTy {
            origin: hir::OpaqueTyOrigin::AsyncFn { .. },
            bounds,
            ..
        }) => {
            if let Some(trait_ref) = bounds.iter().find_map(|b| b.trait_ref())
                && let Some(segment) = trait_ref
                    .path
                    .segments
                    .iter()
                    .find(|seg| seg.ident.name == sym::future_trait)
                && let Some(args) = segment.args
                && let Some(cst_kind) = args
                    .constraints
                    .iter()
                    .find_map(|cst| (cst.ident.name == sym::Output).then_some(cst.kind))
                && let hir::AssocItemConstraintKind::Equality {
                    term: hir::Term::Ty(ty),
                } = cst_kind
            {
                matches!(ty.kind, TyKind::Never)
            } else {
                false
            }
        },
        _ => false,
    }
}
