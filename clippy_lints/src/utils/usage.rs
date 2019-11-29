use rustc::hir::def::Res;
use rustc::hir::*;
use rustc::lint::LateContext;
use rustc::ty;
use rustc_data_structures::fx::FxHashSet;
use rustc_typeck::expr_use_visitor::*;

/// Returns a set of mutated local variable IDs, or `None` if mutations could not be determined.
pub fn mutated_variables<'a, 'tcx>(expr: &'tcx Expr, cx: &'a LateContext<'a, 'tcx>) -> Option<FxHashSet<HirId>> {
    let mut delegate = MutVarsDelegate {
        used_mutably: FxHashSet::default(),
        skip: false,
    };
    let def_id = def_id::DefId::local(expr.hir_id.owner);
    cx.tcx.infer_ctxt().enter(|infcx| {
        ExprUseVisitor::new(&mut delegate, &infcx, def_id, cx.param_env, cx.tables).walk_expr(expr);
    });

    if delegate.skip {
        return None;
    }
    Some(delegate.used_mutably)
}

pub fn is_potentially_mutated<'a, 'tcx>(variable: &'tcx Path, expr: &'tcx Expr, cx: &'a LateContext<'a, 'tcx>) -> bool {
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
