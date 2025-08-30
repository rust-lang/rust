use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};

fn nested_bodies_within<'tcx>(tcx: TyCtxt<'tcx>, item: LocalDefId) -> &'tcx ty::List<LocalDefId> {
    let body = tcx.hir_body_owned_by(item);
    let mut collector =
        NestedBodiesVisitor { tcx, root_def_id: item.to_def_id(), nested_bodies: vec![] };
    collector.visit_body(body);
    tcx.mk_local_def_ids(&collector.nested_bodies)
}

struct NestedBodiesVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    root_def_id: DefId,
    nested_bodies: Vec<LocalDefId>,
}

impl<'tcx> Visitor<'tcx> for NestedBodiesVisitor<'tcx> {
    fn visit_nested_body(&mut self, id: hir::BodyId) {
        let body_def_id = self.tcx.hir_body_owner_def_id(id);
        if self.tcx.typeck_root_def_id(body_def_id.to_def_id()) == self.root_def_id {
            // We visit nested bodies before adding the current body. This
            // means that nested bodies are always stored before their parent.
            let body = self.tcx.hir_body(id);
            self.visit_body(body);
            self.nested_bodies.push(body_def_id);
        }
    }
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers { nested_bodies_within, ..*providers };
}
