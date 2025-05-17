use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::Visitor;
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, TyCtxt};

fn nested_bodies_within<'tcx>(tcx: TyCtxt<'tcx>, item: LocalDefId) -> &'tcx ty::List<LocalDefId> {
    let children = if let Some(children) = tcx.resolutions(()).typeck_children.get(&item) {
        // We re-intern this as a `List` since `List` comparisons are cheap pointer equality.
        tcx.mk_local_def_ids_from_iter(
            children
                .iter()
                .copied()
                // We need to filter out the `Node::Err` bodies which were unsuccessfully
                // lowered in HIR lowering, like items that are contained in never pattern
                // match arms, which don't get lowered.
                .filter(|def_id| !matches!(tcx.hir_node_by_def_id(*def_id), hir::Node::Err(_))),
        )
    } else {
        ty::List::empty()
    };

    // Double check that the list of children we're collecting here is consistent
    // with what we see in the HIR.
    if cfg!(debug_assertions) {
        let body = tcx.hir_body_owned_by(item);
        let mut collector =
            NestedBodiesVisitor { tcx, root_def_id: item.to_def_id(), nested_bodies: vec![] };
        collector.visit_body(body);

        let mut expected = FxIndexSet::from_iter(children);
        for found in collector.nested_bodies {
            if !expected.shift_remove(&found) {
                span_bug!(
                    tcx.def_span(found),
                    "did not expect {found:#?} to be collected as a nested \
                    body child of {item:#?}, but it appeared in the HIR",
                );
            }
        }

        for expected in expected {
            span_bug!(
                tcx.def_span(expected),
                "expected {expected:#?} to be collected as a \
                nested body child of {item:#?}",
            );
        }
    }

    children
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
            self.nested_bodies.push(body_def_id);
            let body = self.tcx.hir_body(id);
            self.visit_body(body);
        }
    }
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers { nested_bodies_within, ..*providers };
}
