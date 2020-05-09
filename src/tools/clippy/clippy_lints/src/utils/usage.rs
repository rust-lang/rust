use crate::utils::match_var;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Expr, HirId, Path};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_middle::ty;
use rustc_span::symbol::{Ident, Symbol};
use rustc_typeck::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor, Place, PlaceBase};

/// Returns a set of mutated local variable IDs, or `None` if mutations could not be determined.
pub fn mutated_variables<'a, 'tcx>(expr: &'tcx Expr<'_>, cx: &'a LateContext<'a, 'tcx>) -> Option<FxHashSet<HirId>> {
    let mut delegate = MutVarsDelegate {
        used_mutably: FxHashSet::default(),
        skip: false,
    };
    let def_id = expr.hir_id.owner.to_def_id();
    cx.tcx.infer_ctxt().enter(|infcx| {
        ExprUseVisitor::new(&mut delegate, &infcx, def_id.expect_local(), cx.param_env, cx.tables).walk_expr(expr);
    });

    if delegate.skip {
        return None;
    }
    Some(delegate.used_mutably)
}

pub fn is_potentially_mutated<'a, 'tcx>(
    variable: &'tcx Path<'_>,
    expr: &'tcx Expr<'_>,
    cx: &'a LateContext<'a, 'tcx>,
) -> bool {
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
    fn update(&mut self, cat: &Place<'tcx>) {
        match cat.base {
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
    fn consume(&mut self, _: &Place<'tcx>, _: ConsumeMode) {}

    fn borrow(&mut self, cmt: &Place<'tcx>, bk: ty::BorrowKind) {
        if let ty::BorrowKind::MutBorrow = bk {
            self.update(&cmt)
        }
    }

    fn mutate(&mut self, cmt: &Place<'tcx>) {
        self.update(&cmt)
    }
}

pub struct UsedVisitor {
    pub var: Symbol,    // var to look for
    pub used: bool,     // has the var been used otherwise?
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
