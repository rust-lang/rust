use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit;
use rustc_hir::intravisit::Visitor;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};

fn stalled_generators_within<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: LocalDefId,
) -> &'tcx ty::List<LocalDefId> {
    if !tcx.next_trait_solver_globally() {
        return ty::List::empty();
    }

    let body = tcx.hir_body_owned_by(item);
    let mut collector =
        StalledGeneratorVisitor { tcx, root_def_id: item.to_def_id(), stalled_coroutines: vec![] };
    collector.visit_body(body);
    tcx.mk_local_def_ids(&collector.stalled_coroutines)
}

struct StalledGeneratorVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    root_def_id: DefId,
    stalled_coroutines: Vec<LocalDefId>,
}

impl<'tcx> Visitor<'tcx> for StalledGeneratorVisitor<'tcx> {
    fn visit_nested_body(&mut self, id: hir::BodyId) {
        if self.tcx.typeck_root_def_id(self.tcx.hir_body_owner_def_id(id).to_def_id())
            == self.root_def_id
        {
            let body = self.tcx.hir_body(id);
            self.visit_body(body);
        }
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Closure(&hir::Closure {
            def_id,
            kind: hir::ClosureKind::Coroutine(_),
            ..
        }) = ex.kind
        {
            self.stalled_coroutines.push(def_id);
        }
        intravisit::walk_expr(self, ex);
    }
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers { stalled_generators_within, ..*providers };
}
