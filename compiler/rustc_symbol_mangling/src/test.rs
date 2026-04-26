//! Walks the crate looking for items/impl-items/trait-items that have
//! either a `rustc_dump_symbol_name` or `rustc_dump_def_path` attribute and
//! generates an error giving, respectively, the symbol name or
//! def-path. This is used for unit testing the code that generates
//! paths etc in all kinds of annoying scenarios.

use rustc_hir::{CRATE_OWNER_ID, find_attr};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{GenericArgs, Instance, TyCtxt};

pub fn dump_symbol_names_and_def_paths(tcx: TyCtxt<'_>) {
    // if the `rustc_attrs` feature is not enabled, then the
    // attributes we are interested in cannot be present anyway, so
    // skip the walk.
    if !tcx.features().rustc_attrs() {
        return;
    }

    tcx.dep_graph.with_ignore(|| {
        for id in tcx.hir_crate_items(()).owners() {
            if id == CRATE_OWNER_ID {
                continue;
            }

            // The format `$tag($value)` is chosen so that tests can elect to test the
            // entirety of the string, if they choose, or else just some subset.

            if let Some(&span) = find_attr!(tcx, id.def_id, RustcDumpSymbolName(span) => span) {
                let def_id = id.def_id.to_def_id();
                let args = GenericArgs::identity_for_item(tcx, id.def_id);
                let args = tcx.erase_and_anonymize_regions(args);
                let instance = Instance::new_raw(def_id, args);
                let mangled = tcx.symbol_name(instance);

                tcx.dcx().span_err(span, format!("symbol-name({mangled})"));

                if let Ok(demangling) = rustc_demangle::try_demangle(mangled.name) {
                    tcx.dcx().span_err(span, format!("demangling({demangling})"));
                    tcx.dcx().span_err(span, format!("demangling-alt({demangling:#})"));
                }
            }

            if let Some(&span) = find_attr!(tcx, id.def_id, RustcDumpDefPath(span) => span) {
                let def_path = with_no_trimmed_paths!(tcx.def_path_str(id.def_id));
                tcx.dcx().span_err(span, format!("def-path({def_path})"));
            }
        }
    })
}
