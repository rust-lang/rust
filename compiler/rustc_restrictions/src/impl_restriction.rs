use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, intravisit::Visitor};
use rustc_middle::bug;
use rustc_middle::query::Providers;
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
        if let hir::ItemKind::Impl(hir::Impl { of_trait: Some(trait_ref), .. }) = &item.kind {
            let trait_def_id = trait_ref.trait_def_id().expect("item is known to be a trait");

            let restriction = self.tcx.impl_restriction(trait_def_id);

            if restriction.is_restricted_in(item.owner_id.to_def_id(), self.tcx) {
                let impl_span =
                    self.tcx.span_of_impl(item.owner_id.to_def_id()).expect("impl should be local");
                let restriction_span = restriction.expect_span();
                let restriction_path =
                    restriction.expect_restriction_path(self.tcx, hir::def_id::LOCAL_CRATE);
                let diag =
                    errors::ImplOfRestrictedTrait { impl_span, restriction_span, restriction_path };
                self.tcx.sess.emit_err(diag);
            }
        }

        hir::intravisit::walk_item(self, item)
    }
}

fn impl_restriction(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::Restriction {
    match tcx.resolutions(()).impl_restrictions.get(&def_id.to_def_id()) {
        Some(restriction) => *restriction,
        None => bug!("impl restriction not found for {def_id:?}"),
    }
}

fn check_impl_restriction(tcx: TyCtxt<'_>, _: ()) {
    tcx.hir().walk_toplevel_module(&mut ImplOfRestrictedTraitVisitor { tcx });
}
