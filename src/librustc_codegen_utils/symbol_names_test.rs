//! Walks the crate looking for items/impl-items/trait-items that have
//! either a `rustc_symbol_name` or `rustc_item_path` attribute and
//! generates an error giving, respectively, the symbol name or
//! item-path. This is used for unit testing the code that generates
//! paths etc in all kinds of annoying scenarios.

use rustc::hir;
use rustc::ty::TyCtxt;

use rustc_mir::monomorphize::Instance;

const SYMBOL_NAME: &'static str = "rustc_symbol_name";
const ITEM_PATH: &'static str = "rustc_item_path";

pub fn report_symbol_names<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    // if the `rustc_attrs` feature is not enabled, then the
    // attributes we are interested in cannot be present anyway, so
    // skip the walk.
    if !tcx.features().rustc_attrs {
        return;
    }

    tcx.dep_graph.with_ignore(|| {
        let mut visitor = SymbolNamesTest { tcx };
        tcx.hir().krate().visit_all_item_likes(&mut visitor);
    })
}

struct SymbolNamesTest<'a, 'tcx:'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> SymbolNamesTest<'a, 'tcx> {
    fn process_attrs(&mut self,
                     hir_id: hir::HirId) {
        let tcx = self.tcx;
        let def_id = tcx.hir().local_def_id_from_hir_id(hir_id);
        for attr in tcx.get_attrs(def_id).iter() {
            if attr.check_name(SYMBOL_NAME) {
                // for now, can only use on monomorphic names
                let instance = Instance::mono(tcx, def_id);
                let name = self.tcx.symbol_name(instance);
                tcx.sess.span_err(attr.span, &format!("symbol-name({})", name));
            } else if attr.check_name(ITEM_PATH) {
                let path = tcx.item_path_str(def_id);
                tcx.sess.span_err(attr.span, &format!("item-path({})", path));
            }

            // (*) The formatting of `tag({})` is chosen so that tests can elect
            // to test the entirety of the string, if they choose, or else just
            // some subset.
        }
    }
}

impl<'a, 'tcx> hir::itemlikevisit::ItemLikeVisitor<'tcx> for SymbolNamesTest<'a, 'tcx> {
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        self.process_attrs(item.hir_id);
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem) {
        self.process_attrs(trait_item.hir_id);
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem) {
        self.process_attrs(impl_item.hir_id);
    }
}
