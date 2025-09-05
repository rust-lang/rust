use crate::macros::root_macro_call_first_node;
use crate::visitors::{Descend, Visitable, for_each_expr, for_each_expr_without_closures};
use crate::{self as utils, get_enclosing_loop_or_multi_call_closure};
use core::ops::ControlFlow;
use hir::def::Res;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self as hir, Expr, ExprKind, HirId, HirIdSet};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, Place, PlaceBase, PlaceWithHirId};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty;
use rustc_span::sym;

/// Returns a set of mutated local variable IDs, or `None` if mutations could not be determined.
pub fn mutated_variables<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) -> Option<HirIdSet> {
    let mut delegate = MutVarsDelegate {
        used_mutably: HirIdSet::default(),
        skip: false,
    };
    ExprUseVisitor::for_clippy(cx, expr.hir_id.owner.def_id, &mut delegate)
        .walk_expr(expr)
        .into_ok();

    if delegate.skip {
        return None;
    }
    Some(delegate.used_mutably)
}

pub fn is_potentially_mutated<'tcx>(variable: HirId, expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) -> bool {
    mutated_variables(expr, cx).is_none_or(|mutated| mutated.contains(&variable))
}

pub fn is_potentially_local_place(local_id: HirId, place: &Place<'_>) -> bool {
    match place.base {
        PlaceBase::Local(id) => id == local_id,
        PlaceBase::Upvar(_) => {
            // Conservatively assume yes.
            true
        },
        _ => false,
    }
}

struct MutVarsDelegate {
    used_mutably: HirIdSet,
    skip: bool,
}

impl MutVarsDelegate {
    fn update(&mut self, cat: &PlaceWithHirId<'_>) {
        match cat.place.base {
            PlaceBase::Local(id) => {
                self.used_mutably.insert(id);
            },
            PlaceBase::Upvar(_) => {
                //FIXME: This causes false negatives. We can't get the `NodeId` from
                //`Categorization::Upvar(_)`. So we search for any `Upvar`s in the
                //`while`-body, not just the ones in the condition.
                self.skip = true;
            },
            _ => {},
        }
    }
}

impl<'tcx> Delegate<'tcx> for MutVarsDelegate {
    fn consume(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn use_cloned(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId, bk: ty::BorrowKind) {
        if bk == ty::BorrowKind::Mutable {
            self.update(cmt);
        }
    }

    fn mutate(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId) {
        self.update(cmt);
    }

    fn fake_read(&mut self, _: &PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}

pub struct ParamBindingIdCollector {
    pub binding_hir_ids: Vec<HirId>,
}
impl<'tcx> ParamBindingIdCollector {
    fn collect_binding_hir_ids(body: &'tcx hir::Body<'tcx>) -> Vec<HirId> {
        let mut hir_ids: Vec<HirId> = Vec::new();
        for param in body.params {
            let mut finder = ParamBindingIdCollector {
                binding_hir_ids: Vec::new(),
            };
            finder.visit_param(param);
            for hir_id in &finder.binding_hir_ids {
                hir_ids.push(*hir_id);
            }
        }
        hir_ids
    }
}
impl<'tcx> Visitor<'tcx> for ParamBindingIdCollector {
    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
        if let hir::PatKind::Binding(_, hir_id, ..) = pat.kind {
            self.binding_hir_ids.push(hir_id);
        }
        intravisit::walk_pat(self, pat);
    }
}

pub struct BindingUsageFinder<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    binding_ids: Vec<HirId>,
}
impl<'a, 'tcx> BindingUsageFinder<'a, 'tcx> {
    pub fn are_params_used(cx: &'a LateContext<'tcx>, body: &'tcx hir::Body<'tcx>) -> bool {
        let mut finder = BindingUsageFinder {
            cx,
            binding_ids: ParamBindingIdCollector::collect_binding_hir_ids(body),
        };
        finder.visit_body(body).is_break()
    }
}
impl<'tcx> Visitor<'tcx> for BindingUsageFinder<'_, 'tcx> {
    type Result = ControlFlow<()>;
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_path(&mut self, path: &hir::Path<'tcx>, _: HirId) -> Self::Result {
        if let Res::Local(id) = path.res
            && self.binding_ids.contains(&id)
        {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}

/// Checks if the given expression is a macro call to `todo!()` or `unimplemented!()`.
pub fn is_todo_unimplemented_macro(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    root_macro_call_first_node(cx, expr)
        .and_then(|macro_call| cx.tcx.get_diagnostic_name(macro_call.def_id))
        .is_some_and(|macro_name| matches!(macro_name, sym::todo_macro | sym::unimplemented_macro))
}

/// Checks if the given expression is a stub, i.e., a `todo!()` or `unimplemented!()` expression,
/// or a block whose last expression is a `todo!()` or `unimplemented!()`.
pub fn is_todo_unimplemented_stub(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ExprKind::Block(block, _) = expr.kind {
        if let Some(last_expr) = block.expr {
            return is_todo_unimplemented_macro(cx, last_expr);
        }

        return block.stmts.last().is_some_and(|stmt| {
            if let hir::StmtKind::Expr(expr) | hir::StmtKind::Semi(expr) = stmt.kind {
                return is_todo_unimplemented_macro(cx, expr);
            }
            false
        });
    }

    is_todo_unimplemented_macro(cx, expr)
}

/// Checks if the given expression contains macro call to `todo!()` or `unimplemented!()`.
pub fn contains_todo_unimplement_macro(cx: &LateContext<'_>, expr: &'_ Expr<'_>) -> bool {
    for_each_expr_without_closures(expr, |e| {
        if is_todo_unimplemented_macro(cx, e) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

pub fn contains_return_break_continue_macro(expression: &Expr<'_>) -> bool {
    for_each_expr_without_closures(expression, |e| {
        match e.kind {
            ExprKind::Ret(..) | ExprKind::Break(..) | ExprKind::Continue(..) => ControlFlow::Break(()),
            // Something special could be done here to handle while or for loop
            // desugaring, as this will detect a break if there's a while loop
            // or a for loop inside the expression.
            _ if e.span.from_expansion() => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        }
    })
    .is_some()
}

pub fn local_used_in<'tcx>(cx: &LateContext<'tcx>, local_id: HirId, v: impl Visitable<'tcx>) -> bool {
    for_each_expr(cx, v, |e| {
        if utils::path_to_local_id(e, local_id) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

pub fn local_used_after_expr(cx: &LateContext<'_>, local_id: HirId, after: &Expr<'_>) -> bool {
    let Some(block) = utils::get_enclosing_block(cx, local_id) else {
        return false;
    };

    // for _ in 1..3 {
    //    local
    // }
    //
    // let closure = || local;
    // closure();
    // closure();
    let loop_start = get_enclosing_loop_or_multi_call_closure(cx, after).map(|e| e.hir_id);

    let mut past_expr = false;
    for_each_expr(cx, block, |e| {
        if past_expr {
            if utils::path_to_local_id(e, local_id) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(Descend::Yes)
            }
        } else if e.hir_id == after.hir_id {
            past_expr = true;
            ControlFlow::Continue(Descend::No)
        } else {
            past_expr = Some(e.hir_id) == loop_start;
            ControlFlow::Continue(Descend::Yes)
        }
    })
    .is_some()
}
