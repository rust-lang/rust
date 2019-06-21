//! Used by `rustc` when compiling a plugin crate.

use syntax::attr;
use syntax::symbol::sym;
use syntax_pos::Span;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;

struct RegistrarFinder {
    registrars: Vec<(hir::HirId, Span)> ,
}

impl<'v> ItemLikeVisitor<'v> for RegistrarFinder {
    fn visit_item(&mut self, item: &hir::Item) {
        if let hir::ItemKind::Fn(..) = item.node {
            if attr::contains_name(&item.attrs, sym::plugin_registrar) {
                self.registrars.push((item.hir_id, item.span));
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

/// Finds the function marked with `#[plugin_registrar]`, if any.
pub fn find_plugin_registrar(tcx: TyCtxt<'_>) -> Option<DefId> {
    tcx.plugin_registrar_fn(LOCAL_CRATE)
}

fn plugin_registrar_fn(tcx: TyCtxt<'_>, cnum: CrateNum) -> Option<DefId> {
    assert_eq!(cnum, LOCAL_CRATE);

    let mut finder = RegistrarFinder { registrars: Vec::new() };
    tcx.hir().krate().visit_all_item_likes(&mut finder);

    match finder.registrars.len() {
        0 => None,
        1 => {
            let (hir_id, _) = finder.registrars.pop().unwrap();
            Some(tcx.hir().local_def_id_from_hir_id(hir_id))
        },
        _ => {
            let diagnostic = tcx.sess.diagnostic();
            let mut e = diagnostic.struct_err("multiple plugin registration functions found");
            for &(_, span) in &finder.registrars {
                e.span_note(span, "one is here");
            }
            e.emit();
            diagnostic.abort_if_errors();
            unreachable!();
        }
    }
}


pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        plugin_registrar_fn,
        ..*providers
    };
}
