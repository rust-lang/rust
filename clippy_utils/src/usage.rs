use crate as utils;
use rustc_hir as hir;
use rustc_hir::intravisit;
use rustc_hir::intravisit::{NestedVisitorMap, Visitor};
use rustc_hir::HirIdSet;
use rustc_hir::{Expr, ExprKind, HirId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty;
use rustc_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};

/// Returns a set of mutated local variable IDs, or `None` if mutations could not be determined.
pub fn mutated_variables<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) -> Option<HirIdSet> {
    let mut delegate = MutVarsDelegate {
        used_mutably: HirIdSet::default(),
        skip: false,
    };
    cx.tcx.infer_ctxt().enter(|infcx| {
        ExprUseVisitor::new(
            &mut delegate,
            &infcx,
            expr.hir_id.owner,
            cx.param_env,
            cx.typeck_results(),
        )
        .walk_expr(expr);
    });

    if delegate.skip {
        return None;
    }
    Some(delegate.used_mutably)
}

pub fn is_potentially_mutated<'tcx>(variable: HirId, expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) -> bool {
    mutated_variables(expr, cx).map_or(true, |mutated| mutated.contains(&variable))
}

struct MutVarsDelegate {
    used_mutably: HirIdSet,
    skip: bool,
}

impl<'tcx> MutVarsDelegate {
    #[allow(clippy::similar_names)]
    fn update(&mut self, cat: &PlaceWithHirId<'tcx>) {
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

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId, bk: ty::BorrowKind) {
        if let ty::BorrowKind::MutBorrow = bk {
            self.update(cmt);
        }
    }

    fn mutate(&mut self, cmt: &PlaceWithHirId<'tcx>, _: HirId) {
        self.update(cmt);
    }

    fn fake_read(&mut self, _: rustc_typeck::expr_use_visitor::Place<'tcx>, _: FakeReadCause, _: HirId) {}
}

pub struct ParamBindingIdCollector {
    binding_hir_ids: Vec<hir::HirId>,
}
impl<'tcx> ParamBindingIdCollector {
    fn collect_binding_hir_ids(body: &'tcx hir::Body<'tcx>) -> Vec<hir::HirId> {
        let mut hir_ids: Vec<hir::HirId> = Vec::new();
        for param in body.params.iter() {
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
impl<'tcx> intravisit::Visitor<'tcx> for ParamBindingIdCollector {
    type Map = Map<'tcx>;

    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
        if let hir::PatKind::Binding(_, hir_id, ..) = pat.kind {
            self.binding_hir_ids.push(hir_id);
        }
        intravisit::walk_pat(self, pat);
    }

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::None
    }
}

pub struct BindingUsageFinder<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    binding_ids: Vec<hir::HirId>,
    usage_found: bool,
}
impl<'a, 'tcx> BindingUsageFinder<'a, 'tcx> {
    pub fn are_params_used(cx: &'a LateContext<'tcx>, body: &'tcx hir::Body<'tcx>) -> bool {
        let mut finder = BindingUsageFinder {
            cx,
            binding_ids: ParamBindingIdCollector::collect_binding_hir_ids(body),
            usage_found: false,
        };
        finder.visit_body(body);
        finder.usage_found
    }
}
impl<'a, 'tcx> intravisit::Visitor<'tcx> for BindingUsageFinder<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if !self.usage_found {
            intravisit::walk_expr(self, expr);
        }
    }

    fn visit_path(&mut self, path: &'tcx hir::Path<'tcx>, _: hir::HirId) {
        if let hir::def::Res::Local(id) = path.res {
            if self.binding_ids.contains(&id) {
                self.usage_found = true;
            }
        }
    }

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}

struct ReturnBreakContinueMacroVisitor {
    seen_return_break_continue: bool,
}

impl ReturnBreakContinueMacroVisitor {
    fn new() -> ReturnBreakContinueMacroVisitor {
        ReturnBreakContinueMacroVisitor {
            seen_return_break_continue: false,
        }
    }
}

impl<'tcx> Visitor<'tcx> for ReturnBreakContinueMacroVisitor {
    type Map = Map<'tcx>;
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        if self.seen_return_break_continue {
            // No need to look farther if we've already seen one of them
            return;
        }
        match &ex.kind {
            ExprKind::Ret(..) | ExprKind::Break(..) | ExprKind::Continue(..) => {
                self.seen_return_break_continue = true;
            },
            // Something special could be done here to handle while or for loop
            // desugaring, as this will detect a break if there's a while loop
            // or a for loop inside the expression.
            _ => {
                if utils::in_macro(ex.span) {
                    self.seen_return_break_continue = true;
                } else {
                    rustc_hir::intravisit::walk_expr(self, ex);
                }
            },
        }
    }
}

pub fn contains_return_break_continue_macro(expression: &Expr<'_>) -> bool {
    let mut recursive_visitor = ReturnBreakContinueMacroVisitor::new();
    recursive_visitor.visit_expr(expression);
    recursive_visitor.seen_return_break_continue
}

pub struct UsedAfterExprVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    definition: HirId,
    past_expr: bool,
    used_after_expr: bool,
}
impl<'a, 'tcx> UsedAfterExprVisitor<'a, 'tcx> {
    pub fn is_found(cx: &'a LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
        utils::path_to_local(expr).map_or(false, |definition| {
            let mut visitor = UsedAfterExprVisitor {
                cx,
                expr,
                definition,
                past_expr: false,
                used_after_expr: false,
            };
            utils::get_enclosing_block(cx, definition).map_or(false, |block| {
                visitor.visit_block(block);
                visitor.used_after_expr
            })
        })
    }
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for UsedAfterExprVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if self.used_after_expr {
            return;
        }

        if expr.hir_id == self.expr.hir_id {
            self.past_expr = true;
        } else if self.past_expr && utils::path_to_local_id(expr, self.definition) {
            self.used_after_expr = true;
        } else {
            intravisit::walk_expr(self, expr);
        }
    }
}
