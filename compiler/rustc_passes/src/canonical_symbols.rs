use rustc_hir::{CanonicalSymbols, ForeignItemId, find_attr};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::{Instance, List, TyCtxt};
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_span::{Symbol, sym};

use crate::diagnostics::DuplicateCanonicalSymbolInCrate;

fn observe_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    canonical_symbols: &mut CanonicalSymbols,
    fid: ForeignItemId,
) {
    let attrs = tcx.hir_attrs(fid.owner_id.into());
    if find_attr!(attrs, RustcCanonicalSymbol) {
        let did = fid.owner_id.def_id;
        let instance = Instance::new_raw(did.to_def_id(), List::identity_for_item(tcx, did));
        let symbol_name = tcx.symbol_name(instance);
        let symbol_name = Symbol::intern(symbol_name.name);

        // insert into our table
        collect_item(tcx, canonical_symbols, symbol_name, fid.owner_id.to_def_id());
    }
}

fn collect_item(
    tcx: TyCtxt<'_>,
    canonical_symbols: &mut CanonicalSymbols,
    symbol: Symbol,
    item_def_id: DefId,
) {
    if let Some(original_def_id) = canonical_symbols.set(symbol, item_def_id) {
        report_duplicate_item(tcx, symbol, original_def_id, item_def_id);
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
    tcx.dcx().emit_err(DuplicateCanonicalSymbolInCrate {
        duplicate_span,
        orig_span,
        crate_name: tcx.crate_name(item_def_id.krate),
        orig_crate_name: tcx.crate_name(original_def_id.krate),
        different_crates: (item_def_id.krate != original_def_id.krate),
        name,
    });
}

/// Traverse and collect the canonical symbols in the current crate
fn canonical_symbols(tcx: TyCtxt<'_>, _: LocalCrate) -> CanonicalSymbols {
    // Initialize the collector.
    let mut canonical_symbols = CanonicalSymbols::new();

    // Optimization: can this crate even define canonical items?
    // (But do not mark `rustc_attrs` as used while doing so)
    if tcx.features().enabled_features().contains(&sym::rustc_attrs) {
        // Collect canonical symbols in this crate.
        let crate_items = tcx.hir_crate_items(());
        for id in crate_items.foreign_items() {
            observe_item(tcx, &mut canonical_symbols, id);
        }
    }
    canonical_symbols
}

/// Traverse and collect all the canonical symbols in all crates.
fn all_canonical_symbols(tcx: TyCtxt<'_>, (): ()) -> CanonicalSymbols {
    // Initialize the collector.
    let mut items = CanonicalSymbols::new();

    // Collect all canonical symbols
    for cnum in tcx.crates(()).iter().copied().chain(std::iter::once(LOCAL_CRATE)) {
        for cs in tcx.canonical_symbols(cnum).iter() {
            collect_item(tcx, &mut items, cs.symbol, cs.def_id);
        }
    }

    items
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.canonical_symbols = canonical_symbols;
    providers.all_canonical_symbols = all_canonical_symbols;
}
