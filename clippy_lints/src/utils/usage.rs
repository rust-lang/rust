use rustc::hir::def::Res;
use rustc::hir::*;
use rustc::lint::LateContext;
use rustc::middle::expr_use_visitor::*;
use rustc::middle::mem_categorization::cmt_;
use rustc::middle::mem_categorization::Categorization;
use rustc::ty;
use rustc_data_structures::fx::FxHashSet;
use syntax::source_map::Span;

/// Returns a set of mutated local variable IDs, or `None` if mutations could not be determined.
pub fn mutated_variables<'a, 'tcx>(expr: &'tcx Expr, cx: &'a LateContext<'a, 'tcx>) -> Option<FxHashSet<HirId>> {
    let mut delegate = MutVarsDelegate {
        used_mutably: FxHashSet::default(),
        skip: false,
    };
    let def_id = def_id::DefId::local(expr.hir_id.owner);
    let region_scope_tree = &cx.tcx.region_scope_tree(def_id);
    ExprUseVisitor::new(
        &mut delegate,
        cx.tcx,
        def_id,
        cx.param_env,
        region_scope_tree,
        cx.tables,
        None,
    )
    .walk_expr(expr);

    if delegate.skip {
        return None;
    }
    Some(delegate.used_mutably)
}

pub fn is_potentially_mutated<'a, 'tcx>(variable: &'tcx Path, expr: &'tcx Expr, cx: &'a LateContext<'a, 'tcx>) -> bool {
    if let Res::Local(id) = variable.res {
        mutated_variables(expr, cx).map_or(true, |mutated| mutated.contains(&id))
    } else {
        return true;
    }
}

struct MutVarsDelegate {
    used_mutably: FxHashSet<HirId>,
    skip: bool,
}

impl<'tcx> MutVarsDelegate {
    #[allow(clippy::similar_names)]
    fn update(&mut self, cat: &'tcx Categorization<'_>) {
        match *cat {
            Categorization::Local(id) => {
                self.used_mutably.insert(id);
            },
            Categorization::Upvar(_) => {
                //FIXME: This causes false negatives. We can't get the `NodeId` from
                //`Categorization::Upvar(_)`. So we search for any `Upvar`s in the
                //`while`-body, not just the ones in the condition.
                self.skip = true
            },
            Categorization::Deref(ref cmt, _) | Categorization::Interior(ref cmt, _) => self.update(&cmt.cat),
            _ => {},
        }
    }
}

impl<'tcx> Delegate<'tcx> for MutVarsDelegate {
    fn consume(&mut self, _: HirId, _: Span, _: &cmt_<'tcx>, _: ConsumeMode) {}

    fn matched_pat(&mut self, _: &Pat, _: &cmt_<'tcx>, _: MatchMode) {}

    fn consume_pat(&mut self, _: &Pat, _: &cmt_<'tcx>, _: ConsumeMode) {}

    fn borrow(&mut self, _: HirId, _: Span, cmt: &cmt_<'tcx>, _: ty::Region<'_>, bk: ty::BorrowKind, _: LoanCause) {
        if let ty::BorrowKind::MutBorrow = bk {
            self.update(&cmt.cat)
        }
    }

    fn mutate(&mut self, _: HirId, _: Span, cmt: &cmt_<'tcx>, _: MutateMode) {
        self.update(&cmt.cat)
    }

    fn decl_without_init(&mut self, _: HirId, _: Span) {}
}
