use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_ast::BindingMode;
use rustc_errors::Applicability;
use rustc_hir::{Body, Expr, ExprKind, HirId, HirIdSet, PatKind};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};
use rustc_lint::LateContext;
use rustc_middle::mir::{FakeReadCause, Mutability};
use rustc_middle::ty::{self, BorrowKind};
use rustc_span::{Symbol, sym};

use super::ITER_OVEREAGER_CLONED;
use crate::redundant_clone::REDUNDANT_CLONE;

#[derive(Clone, Copy)]
pub(super) enum Op<'a> {
    // rm `.cloned()`
    // e.g. `count`
    RmCloned,

    // rm `.cloned()`
    // e.g. `map` `for_each` `all` `any`
    NeedlessMove(&'a Expr<'a>),

    // later `.cloned()`
    // and add `&` to the parameter of closure parameter
    // e.g. `find` `filter`
    FixClosure(Symbol, &'a Expr<'a>),

    // later `.cloned()`
    // e.g. `skip` `take`
    LaterCloned,
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    cloned_call: &'tcx Expr<'_>,
    cloned_recv: &'tcx Expr<'_>,
    op: Op<'tcx>,
    needs_into_iter: bool,
) {
    let typeck = cx.typeck_results();
    if let Some(iter_id) = cx.tcx.get_diagnostic_item(sym::Iterator)
        && let Some(method_id) = typeck.type_dependent_def_id(expr.hir_id)
        && cx.tcx.trait_of_item(method_id) == Some(iter_id)
        && let Some(method_id) = typeck.type_dependent_def_id(cloned_call.hir_id)
        && cx.tcx.trait_of_item(method_id) == Some(iter_id)
        && let cloned_recv_ty = typeck.expr_ty_adjusted(cloned_recv)
        && let Some(iter_assoc_ty) = cx.get_associated_type(cloned_recv_ty, iter_id, sym::Item)
        && matches!(*iter_assoc_ty.kind(), ty::Ref(_, ty, _) if !is_copy(cx, ty))
    {
        if needs_into_iter
            && let Some(into_iter_id) = cx.tcx.get_diagnostic_item(sym::IntoIterator)
            && !implements_trait(cx, iter_assoc_ty, into_iter_id, &[])
        {
            return;
        }

        if let Op::NeedlessMove(expr) = op {
            let ExprKind::Closure(closure) = expr.kind else {
                return;
            };
            let body @ Body { params: [p], .. } = cx.tcx.hir_body(closure.body) else {
                return;
            };
            let mut delegate = MoveDelegate {
                used_move: HirIdSet::default(),
            };

            ExprUseVisitor::for_clippy(cx, closure.def_id, &mut delegate)
                .consume_body(body)
                .into_ok();

            let mut to_be_discarded = false;

            p.pat.walk(|it| {
                if delegate.used_move.contains(&it.hir_id) {
                    to_be_discarded = true;
                    return false;
                }

                match it.kind {
                    PatKind::Binding(BindingMode(_, Mutability::Mut), _, _, _) | PatKind::Ref(_, Mutability::Mut) => {
                        to_be_discarded = true;
                        false
                    },
                    _ => true,
                }
            });

            if to_be_discarded {
                return;
            }
        }

        let (lint, msg, trailing_clone) = match op {
            Op::RmCloned | Op::NeedlessMove(_) => (REDUNDANT_CLONE, "unneeded cloning of iterator items", ""),
            Op::LaterCloned | Op::FixClosure(_, _) => (
                ITER_OVEREAGER_CLONED,
                "unnecessarily eager cloning of iterator items",
                ".cloned()",
            ),
        };

        span_lint_and_then(cx, lint, expr.span, msg, |diag| match op {
            Op::RmCloned | Op::LaterCloned => {
                let method_span = expr.span.with_lo(cloned_call.span.hi());
                if let Some(mut snip) = snippet_opt(cx, method_span) {
                    snip.push_str(trailing_clone);
                    let replace_span = expr.span.with_lo(cloned_recv.span.hi());
                    diag.span_suggestion(replace_span, "try", snip, Applicability::MachineApplicable);
                }
            },
            Op::FixClosure(name, predicate_expr) => {
                if let Some(predicate) = snippet_opt(cx, predicate_expr.span) {
                    let new_closure = if let ExprKind::Closure(_) = predicate_expr.kind {
                        predicate.replacen('|', "|&", 1)
                    } else {
                        format!("|&x| {predicate}(x)")
                    };
                    let snip = format!(".{name}({new_closure}).cloned()");
                    let replace_span = expr.span.with_lo(cloned_recv.span.hi());
                    diag.span_suggestion(replace_span, "try", snip, Applicability::MachineApplicable);
                }
            },
            Op::NeedlessMove(_) => {
                let method_span = expr.span.with_lo(cloned_call.span.hi());
                if let Some(snip) = snippet_opt(cx, method_span) {
                    let replace_span = expr.span.with_lo(cloned_recv.span.hi());
                    diag.span_suggestion(replace_span, "try", snip, Applicability::MaybeIncorrect);
                }
            },
        });
    }
}

struct MoveDelegate {
    used_move: HirIdSet,
}

impl<'tcx> Delegate<'tcx> for MoveDelegate {
    fn consume(&mut self, place_with_id: &PlaceWithHirId<'tcx>, _: HirId) {
        if let PlaceBase::Local(l) = place_with_id.place.base {
            self.used_move.insert(l);
        }
    }

    fn use_cloned(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn borrow(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId, _: BorrowKind) {}

    fn mutate(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn fake_read(&mut self, _: &PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}
