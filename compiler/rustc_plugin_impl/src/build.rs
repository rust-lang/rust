//! Used by `rustc` when compiling a plugin crate.

use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;
use rustc_span::Span;

struct RegistrarFinder<'tcx> {
    tcx: TyCtxt<'tcx>,
    registrars: Vec<(LocalDefId, Span)>,
}

impl<'v, 'tcx> ItemLikeVisitor<'v> for RegistrarFinder<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        if let hir::ItemKind::Fn(..) = item.kind {
            let attrs = self.tcx.hir().attrs(item.hir_id());
            if self.tcx.sess.contains_name(attrs, sym::plugin_registrar) {
                self.registrars.push((item.def_id, item.span));
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {}

    fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {}
}

/// Finds the function marked with `#[plugin_registrar]`, if any.
fn plugin_registrar_fn(tcx: TyCtxt<'_>, (): ()) -> Option<LocalDefId> {
    let mut finder = RegistrarFinder { tcx, registrars: Vec::new() };
    tcx.hir().krate().visit_all_item_likes(&mut finder);

    let (def_id, span) = finder.registrars.pop()?;

    if !finder.registrars.is_empty() {
        let diagnostic = tcx.sess.diagnostic();
        let mut e = diagnostic.struct_err("multiple plugin registration functions found");
        e.span_note(span, "one is here");
        for &(_, span) in &finder.registrars {
            e.span_note(span, "one is here");
        }
        e.emit();
        diagnostic.abort_if_errors();
        unreachable!();
    }

    Some(def_id)
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { plugin_registrar_fn, ..*providers };
}
