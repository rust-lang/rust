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

use crate::hir::def_id::{DefId, LOCAL_CRATE};
use crate::ty::TyCtxt;
use crate::util::nodemap::FxHashMap;

use syntax::ast;
use syntax::symbol::{Symbol, sym};
use crate::hir::itemlikevisit::ItemLikeVisitor;
use crate::hir;

struct DiagnosticItemCollector<'tcx> {
    // items from this crate
    items: FxHashMap<Symbol, DefId>,
    tcx: TyCtxt<'tcx>,
}

impl<'v, 'tcx> ItemLikeVisitor<'v> for DiagnosticItemCollector<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        self.observe_item(&item.attrs, item.hir_id);
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem<'_>) {
        self.observe_item(&trait_item.attrs, trait_item.hir_id);
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem<'_>) {
        self.observe_item(&impl_item.attrs, impl_item.hir_id);
    }
}

impl<'tcx> DiagnosticItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> DiagnosticItemCollector<'tcx> {
        DiagnosticItemCollector {
            tcx,
            items: Default::default(),
        }
    }

    fn observe_item(&mut self, attrs: &[ast::Attribute], hir_id: hir::HirId) {
        if let Some(name) = extract(attrs) {
            let def_id = self.tcx.hir().local_def_id(hir_id);
            // insert into our table
            collect_item(self.tcx, &mut self.items, name, def_id);
        }
    }
}

fn collect_item(
    tcx: TyCtxt<'_>,
    items: &mut FxHashMap<Symbol, DefId>,
    name: Symbol,
    item_def_id: DefId,
) {
    // Check for duplicates.
    if let Some(original_def_id) = items.insert(name, item_def_id) {
        if original_def_id != item_def_id {
            let mut err = match tcx.hir().span_if_local(item_def_id) {
                Some(span) => tcx.sess.struct_span_err(
                    span,
                    &format!("duplicate diagnostic item found: `{}`.", name)),
                None => tcx.sess.struct_err(&format!(
                        "duplicate diagnostic item in crate `{}`: `{}`.",
                        tcx.crate_name(item_def_id.krate),
                        name)),
            };
            if let Some(span) = tcx.hir().span_if_local(original_def_id) {
                span_note!(&mut err, span, "first defined here.");
            } else {
                err.note(&format!("first defined in crate `{}`.",
                                    tcx.crate_name(original_def_id.krate)));
            }
            err.emit();
        }
    }
}

/// Extract the first `rustc_diagnostic_item = "$name"` out of a list of attributes.
fn extract(attrs: &[ast::Attribute]) -> Option<Symbol> {
    attrs.iter().find_map(|attr| {
        if attr.check_name(sym::rustc_diagnostic_item) {
            attr.value_str()
        } else {
            None
        }
    })
}

/// Traverse and collect the diagnostic items in the current
pub fn collect<'tcx>(tcx: TyCtxt<'tcx>) -> &'tcx FxHashMap<Symbol, DefId> {
    // Initialize the collector.
    let mut collector = DiagnosticItemCollector::new(tcx);

    // Collect diagnostic items in this crate.
    tcx.hir().krate().visit_all_item_likes(&mut collector);

    tcx.arena.alloc(collector.items)
}


/// Traverse and collect all the diagnostic items in all crates.
pub fn collect_all<'tcx>(tcx: TyCtxt<'tcx>) -> &'tcx FxHashMap<Symbol, DefId> {
    // Initialize the collector.
    let mut collector = FxHashMap::default();

    // Collect diagnostic items in other crates.
    for &cnum in tcx.crates().iter().chain(std::iter::once(&LOCAL_CRATE)) {
        for (&name, &def_id) in tcx.diagnostic_items(cnum).iter() {
            collect_item(tcx, &mut collector, name, def_id);
        }
    }

    tcx.arena.alloc(collector)
}
