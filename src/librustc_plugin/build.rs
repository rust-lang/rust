//! Used by `rustc` when compiling a plugin crate.

use syntax::ast;
use syntax::attr;
use errors;
use syntax_pos::Span;
use rustc::hir::map::Map;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;

struct RegistrarFinder {
    registrars: Vec<(ast::NodeId, Span)> ,
}

impl<'v> ItemLikeVisitor<'v> for RegistrarFinder {
    fn visit_item(&mut self, item: &hir::Item) {
        if let hir::ItemKind::Fn(..) = item.node {
            if attr::contains_name(&item.attrs,
                                   "plugin_registrar") {
                self.registrars.push((item.id, item.span));
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

/// Find the function marked with `#[plugin_registrar]`, if any.
pub fn find_plugin_registrar(diagnostic: &errors::Handler,
                             hir_map: &Map)
                             -> Option<ast::NodeId> {
    let krate = hir_map.krate();

    let mut finder = RegistrarFinder { registrars: Vec::new() };
    krate.visit_all_item_likes(&mut finder);

    match finder.registrars.len() {
        0 => None,
        1 => {
            let (node_id, _) = finder.registrars.pop().unwrap();
            Some(node_id)
        },
        _ => {
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
