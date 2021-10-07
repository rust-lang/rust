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
use rustc_hir as hir;
use rustc_hir::diagnostic_items::DiagnosticItems;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_span::symbol::{sym, Symbol};

struct DiagnosticItemCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    diagnostic_items: DiagnosticItems,
}

impl<'v, 'tcx> ItemLikeVisitor<'v> for DiagnosticItemCollector<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        self.observe_item(item.def_id);
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem<'_>) {
        self.observe_item(trait_item.def_id);
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem<'_>) {
        self.observe_item(impl_item.def_id);
    }

    fn visit_foreign_item(&mut self, foreign_item: &hir::ForeignItem<'_>) {
        self.observe_item(foreign_item.def_id);
    }
}

impl<'tcx> DiagnosticItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> DiagnosticItemCollector<'tcx> {
        DiagnosticItemCollector { tcx, diagnostic_items: DiagnosticItems::default() }
    }

    fn observe_item(&mut self, def_id: LocalDefId) {
        let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
        let attrs = self.tcx.hir().attrs(hir_id);
        if let Some(name) = extract(attrs) {
            // insert into our table
            collect_item(self.tcx, &mut self.diagnostic_items, name, def_id.to_def_id());
        }
    }
}

fn collect_item(tcx: TyCtxt<'_>, items: &mut DiagnosticItems, name: Symbol, item_def_id: DefId) {
    items.id_to_name.insert(item_def_id, name);
    if let Some(original_def_id) = items.name_to_id.insert(name, item_def_id) {
        if original_def_id != item_def_id {
            let mut err = match tcx.hir().span_if_local(item_def_id) {
                Some(span) => tcx.sess.struct_span_err(
                    span,
                    &format!("duplicate diagnostic item found: `{}`.", name),
                ),
                None => tcx.sess.struct_err(&format!(
                    "duplicate diagnostic item in crate `{}`: `{}`.",
                    tcx.crate_name(item_def_id.krate),
                    name
                )),
            };
            if let Some(span) = tcx.hir().span_if_local(original_def_id) {
                err.span_note(span, "the diagnostic item is first defined here");
            } else {
                err.note(&format!(
                    "the diagnostic item is first defined in crate `{}`.",
                    tcx.crate_name(original_def_id.krate)
                ));
            }
            err.emit();
        }
    }
}

/// Extract the first `rustc_diagnostic_item = "$name"` out of a list of attributes.p
fn extract(attrs: &[ast::Attribute]) -> Option<Symbol> {
    attrs.iter().find_map(|attr| {
        if attr.has_name(sym::rustc_diagnostic_item) { attr.value_str() } else { None }
    })
}

/// Traverse and collect the diagnostic items in the current
fn diagnostic_items<'tcx>(tcx: TyCtxt<'tcx>, cnum: CrateNum) -> DiagnosticItems {
    assert_eq!(cnum, LOCAL_CRATE);

    // Initialize the collector.
    let mut collector = DiagnosticItemCollector::new(tcx);

    // Collect diagnostic items in this crate.
    tcx.hir().visit_all_item_likes(&mut collector);

    collector.diagnostic_items
}

/// Traverse and collect all the diagnostic items in all crates.
fn all_diagnostic_items<'tcx>(tcx: TyCtxt<'tcx>, (): ()) -> DiagnosticItems {
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
