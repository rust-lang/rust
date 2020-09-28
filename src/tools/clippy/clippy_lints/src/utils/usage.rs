use crate::utils::match_var;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::intravisit;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Expr, HirId, Path};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_middle::ty;
use rustc_span::symbol::{Ident, Symbol};
use rustc_typeck::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};

/// Returns a set of mutated local variable IDs, or `None` if mutations could not be determined.
pub fn mutated_variables<'tcx>(expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) -> Option<FxHashSet<HirId>> {
    let mut delegate = MutVarsDelegate {
        used_mutably: FxHashSet::default(),
        skip: false,
    };
    let def_id = expr.hir_id.owner.to_def_id();
    cx.tcx.infer_ctxt().enter(|infcx| {
        ExprUseVisitor::new(
            &mut delegate,
            &infcx,
            def_id.expect_local(),
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

pub fn is_potentially_mutated<'tcx>(variable: &'tcx Path<'_>, expr: &'tcx Expr<'_>, cx: &LateContext<'tcx>) -> bool {
    if let Res::Local(id) = variable.res {
        mutated_variables(expr, cx).map_or(true, |mutated| mutated.contains(&id))
    } else {
        true
    }
}

struct MutVarsDelegate {
    used_mutably: FxHashSet<HirId>,
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
                self.skip = true
            },
            _ => {},
        }
    }
}

impl<'tcx> Delegate<'tcx> for MutVarsDelegate {
    fn consume(&mut self, _: &PlaceWithHirId<'tcx>, _: ConsumeMode) {}

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, bk: ty::BorrowKind) {
        if let ty::BorrowKind::MutBorrow = bk {
            self.update(&cmt)
        }
    }

    fn mutate(&mut self, cmt: &PlaceWithHirId<'tcx>) {
        self.update(&cmt)
    }
}

pub struct UsedVisitor {
    pub var: Symbol, // var to look for
    pub used: bool,  // has the var been used otherwise?
}

impl<'tcx> Visitor<'tcx> for UsedVisitor {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if match_var(expr, self.var) {
            self.used = true;
        } else {
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

pub fn is_unused<'tcx>(ident: &'tcx Ident, body: &'tcx Expr<'_>) -> bool {
    let mut visitor = UsedVisitor {
        var: ident.name,
        used: false,
    };
    walk_expr(&mut visitor, body);
    !visitor.used
}

pub struct ParamBindingIdCollector {
    binding_hir_ids: Vec<hir::HirId>,
}
impl<'tcx> ParamBindingIdCollector {
    fn collect_binding_hir_ids(body: &'tcx hir::Body<'tcx>) -> Vec<hir::HirId> {
        let mut finder = ParamBindingIdCollector {
            binding_hir_ids: Vec::new(),
        };
        finder.visit_body(body);
        finder.binding_hir_ids
    }
}
impl<'tcx> intravisit::Visitor<'tcx> for ParamBindingIdCollector {
    type Map = Map<'tcx>;

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        if let hir::PatKind::Binding(_, hir_id, ..) = param.pat.kind {
            self.binding_hir_ids.push(hir_id);
        }
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
