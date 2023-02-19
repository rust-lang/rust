//! Detecting diagnostic items.
//!
//! Diagnostic items are items that are not language-inherent, but can reasonably be expected to
//! exist for diagnostic purposes. This allows diagnostic authors to refer to specific items
//! directly, without having to guess module paths and crates.
//! Examples are:
//!
//! * Traits like `Debug`, that have no bearing on language semantics
//!
//! * Compiler internal types like `Ty` and `TyCtxt`

use rustc_ast as ast;
use rustc_hir::diagnostic_items::DiagnosticItems;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_span::symbol::{kw::Empty, sym, Symbol};

use crate::errors::{DuplicateDiagnosticItem, DuplicateDiagnosticItemInCrate};

fn observe_item(tcx: TyCtxt<'_>, diagnostic_items: &mut DiagnosticItems, def_id: LocalDefId) {
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let attrs = tcx.hir().attrs(hir_id);
    if let Some(name) = extract(attrs) {
        // insert into our table
        collect_item(tcx, diagnostic_items, name, def_id.to_def_id());
    }
}

fn collect_item(tcx: TyCtxt<'_>, items: &mut DiagnosticItems, name: Symbol, item_def_id: DefId) {
    items.id_to_name.insert(item_def_id, name);
    if let Some(original_def_id) = items.name_to_id.insert(name, item_def_id) {
        if original_def_id != item_def_id {
            let orig_span = tcx.hir().span_if_local(original_def_id);
            let orig_crate_name =
                orig_span.is_none().then(|| tcx.crate_name(original_def_id.krate));
            match tcx.hir().span_if_local(item_def_id) {
                Some(span) => tcx.sess.emit_err(DuplicateDiagnosticItem { span, name }),
                None => tcx.sess.emit_err(DuplicateDiagnosticItemInCrate {
                    span: orig_span,
                    orig_crate_name: orig_crate_name.unwrap_or(Empty),
                    have_orig_crate_name: orig_crate_name.map(|_| ()),
                    crate_name: tcx.crate_name(item_def_id.krate),
                    name,
                }),
            };
        }
    }
}

/// Extract the first `rustc_diagnostic_item = "$name"` out of a list of attributes.
fn extract(attrs: &[ast::Attribute]) -> Option<Symbol> {
    attrs.iter().find_map(|attr| {
        if attr.has_name(sym::rustc_diagnostic_item) { attr.value_str() } else { None }
    })
}

/// Traverse and collect the diagnostic items in the current
fn diagnostic_items(tcx: TyCtxt<'_>, cnum: CrateNum) -> DiagnosticItems {
    assert_eq!(cnum, LOCAL_CRATE);

    // Initialize the collector.
    let mut diagnostic_items = DiagnosticItems::default();

    // Collect diagnostic items in this crate.
    let crate_items = tcx.hir_crate_items(());

    for id in crate_items.items() {
        observe_item(tcx, &mut diagnostic_items, id.owner_id.def_id);
    }

    for id in crate_items.trait_items() {
        observe_item(tcx, &mut diagnostic_items, id.owner_id.def_id);
    }

    for id in crate_items.impl_items() {
        observe_item(tcx, &mut diagnostic_items, id.owner_id.def_id);
    }

    for id in crate_items.foreign_items() {
        observe_item(tcx, &mut diagnostic_items, id.owner_id.def_id);
    }

    diagnostic_items
}

/// Traverse and collect all the diagnostic items in all crates.
fn all_diagnostic_items(tcx: TyCtxt<'_>, (): ()) -> DiagnosticItems {
    // Initialize the collector.
    let mut items = DiagnosticItems::default();

    // Collect diagnostic items in other crates.
    for &cnum in tcx.crates(()).iter().chain(std::iter::once(&LOCAL_CRATE)) {
        for (&name, &def_id) in &tcx.diagnostic_items(cnum).name_to_id {
            collect_item(tcx, &mut items, name, def_id);
        }
    }

    items
}

pub fn provide(providers: &mut Providers) {
    providers.diagnostic_items = diagnostic_items;
    providers.all_diagnostic_items = all_diagnostic_items;
}
