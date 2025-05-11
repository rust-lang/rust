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

use rustc_hir::diagnostic_items::DiagnosticItems;
use rustc_hir::{Attribute, OwnerId};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_span::{Symbol, sym};

use crate::errors::DuplicateDiagnosticItemInCrate;

fn observe_item<'tcx>(tcx: TyCtxt<'tcx>, diagnostic_items: &mut DiagnosticItems, owner: OwnerId) {
    let attrs = tcx.hir_attrs(owner.into());
    if let Some(name) = extract(attrs) {
        // insert into our table
        collect_item(tcx, diagnostic_items, name, owner.to_def_id());
    }
}

fn collect_item(tcx: TyCtxt<'_>, items: &mut DiagnosticItems, name: Symbol, item_def_id: DefId) {
    items.id_to_name.insert(item_def_id, name);
    if let Some(original_def_id) = items.name_to_id.insert(name, item_def_id) {
        if original_def_id != item_def_id {
            report_duplicate_item(tcx, name, original_def_id, item_def_id);
        }
    }
}

fn report_duplicate_item(
    tcx: TyCtxt<'_>,
    name: Symbol,
    original_def_id: DefId,
    item_def_id: DefId,
) {
    let orig_span = tcx.hir_span_if_local(original_def_id);
    let duplicate_span = tcx.hir_span_if_local(item_def_id);
    tcx.dcx().emit_err(DuplicateDiagnosticItemInCrate {
        duplicate_span,
        orig_span,
        crate_name: tcx.crate_name(item_def_id.krate),
        orig_crate_name: tcx.crate_name(original_def_id.krate),
        different_crates: (item_def_id.krate != original_def_id.krate),
        name,
    });
}

/// Extract the first `rustc_diagnostic_item = "$name"` out of a list of attributes.
fn extract(attrs: &[Attribute]) -> Option<Symbol> {
    attrs.iter().find_map(|attr| {
        if attr.has_name(sym::rustc_diagnostic_item) { attr.value_str() } else { None }
    })
}

/// Traverse and collect the diagnostic items in the current
fn diagnostic_items(tcx: TyCtxt<'_>, _: LocalCrate) -> DiagnosticItems {
    // Initialize the collector.
    let mut diagnostic_items = DiagnosticItems::default();

    // Collect diagnostic items in this crate.
    let crate_items = tcx.hir_crate_items(());
    for id in crate_items.owners() {
        observe_item(tcx, &mut diagnostic_items, id);
    }

    diagnostic_items
}

/// Traverse and collect all the diagnostic items in all crates.
fn all_diagnostic_items(tcx: TyCtxt<'_>, (): ()) -> DiagnosticItems {
    // Initialize the collector.
    let mut items = DiagnosticItems::default();

    // Collect diagnostic items in visible crates.
    for cnum in tcx
        .crates(())
        .iter()
        .copied()
        .filter(|cnum| tcx.is_user_visible_dep(*cnum))
        .chain(std::iter::once(LOCAL_CRATE))
    {
        for (&name, &def_id) in &tcx.diagnostic_items(cnum).name_to_id {
            collect_item(tcx, &mut items, name, def_id);
        }
    }

    items
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.diagnostic_items = diagnostic_items;
    providers.all_diagnostic_items = all_diagnostic_items;
}
