//! Walks the crate looking for items/impl-items/trait-items that have
//! either a `rustc_symbol_name` or `rustc_def_path` attribute and
//! generates an error giving, respectively, the symbol name or
//! def-path. This is used for unit testing the code that generates
//! paths etc in all kinds of annoying scenarios.

use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{GenericArgs, Instance, TyCtxt};
use rustc_span::{Symbol, sym};

use crate::errors::{Kind, TestOutput};

const SYMBOL_NAME: Symbol = sym::rustc_symbol_name;
const DEF_PATH: Symbol = sym::rustc_def_path;

pub fn report_symbol_names(tcx: TyCtxt<'_>) {
    // if the `rustc_attrs` feature is not enabled, then the
    // attributes we are interested in cannot be present anyway, so
    // skip the walk.
    if !tcx.features().rustc_attrs() {
        return;
    }

    tcx.dep_graph.with_ignore(|| {
        let mut symbol_names = SymbolNamesTest { tcx };
        let crate_items = tcx.hir_crate_items(());

        for id in crate_items.free_items() {
            symbol_names.process_attrs(id.owner_id.def_id);
        }

        for id in crate_items.trait_items() {
            symbol_names.process_attrs(id.owner_id.def_id);
        }

        for id in crate_items.impl_items() {
            symbol_names.process_attrs(id.owner_id.def_id);
        }

        for id in crate_items.foreign_items() {
            symbol_names.process_attrs(id.owner_id.def_id);
        }
    })
}

struct SymbolNamesTest<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl SymbolNamesTest<'_> {
    fn process_attrs(&mut self, def_id: LocalDefId) {
        let tcx = self.tcx;
        // The formatting of `tag({})` is chosen so that tests can elect
        // to test the entirety of the string, if they choose, or else just
        // some subset.
        for attr in tcx.get_attrs(def_id, SYMBOL_NAME) {
            let def_id = def_id.to_def_id();
            let instance = Instance::new_raw(
                def_id,
                tcx.erase_and_anonymize_regions(GenericArgs::identity_for_item(tcx, def_id)),
            );
            let mangled = tcx.symbol_name(instance);
            tcx.dcx().emit_err(TestOutput {
                span: attr.span(),
                kind: Kind::SymbolName,
                content: format!("{mangled}"),
            });
            if let Ok(demangling) = rustc_demangle::try_demangle(mangled.name) {
                tcx.dcx().emit_err(TestOutput {
                    span: attr.span(),
                    kind: Kind::Demangling,
                    content: format!("{demangling}"),
                });
                tcx.dcx().emit_err(TestOutput {
                    span: attr.span(),
                    kind: Kind::DemanglingAlt,
                    content: format!("{demangling:#}"),
                });
            }
        }

        for attr in tcx.get_attrs(def_id, DEF_PATH) {
            tcx.dcx().emit_err(TestOutput {
                span: attr.span(),
                kind: Kind::DefPath,
                content: with_no_trimmed_paths!(tcx.def_path_str(def_id)),
            });
        }
    }
}
