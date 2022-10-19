use hir::intravisit::Visitor;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, Node};
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, TyCtxt};

use crate::errors;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { impl_restriction, check_impl_restriction, ..*providers };
}

struct ImplOfRestrictedTraitVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'v> Visitor<'v> for ImplOfRestrictedTraitVisitor<'v> {
    type NestedFilter = rustc_middle::hir::nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'v hir::Item<'v>) {
        if let hir::ItemKind::Trait(..) = item.kind {
            let restriction = self.tcx.impl_restriction(item.owner_id.def_id);

            self.tcx.for_each_impl(item.owner_id.to_def_id(), |impl_| {
                if restriction.is_restricted_in(impl_, self.tcx) {
                    self.tcx.sess.emit_err(errors::ImplOfRestrictedTrait {
                        impl_span: self.tcx.span_of_impl(impl_).expect("impl should be local"),
                        restriction_span: restriction.expect_span(),
                        restriction_path: restriction
                            .expect_restriction_path(self.tcx, hir::def_id::LOCAL_CRATE),
                    });
                }
            });
        };

        hir::intravisit::walk_item(self, item)
    }
}

pub(crate) fn impl_restriction(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::Restriction {
    match tcx.resolutions(()).impl_restrictions.get(&def_id) {
        Some(restriction) => *restriction,
        None => {
            let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
            match tcx.hir().get(hir_id) {
                Node::Item(hir::Item { kind: hir::ItemKind::Trait(..), .. }) => {
                    span_bug!(
                        tcx.def_span(def_id),
                        "impl restriction table unexpectedly missing a def-id: {def_id:?}",
                    )
                }
                _ => {
                    span_bug!(
                        tcx.def_span(def_id),
                        "called `impl_restriction` on non-trait: {def_id:?}",
                    )
                }
            }
        }
    }
}

pub(crate) fn check_impl_restriction(tcx: TyCtxt<'_>, _: ()) {
    tcx.hir().walk_toplevel_module(&mut ImplOfRestrictedTraitVisitor { tcx });
}
