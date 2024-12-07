//! Parsing and validation of builtin attributes

use rustc_ast::attr::AttributeExt;
use rustc_ast::MetaItemInner;
use rustc_span::symbol::Symbol;


/// Read the content of a `rustc_confusables` attribute, and return the list of candidate names.
pub fn parse_confusables(attr: &impl AttributeExt) -> Option<Vec<Symbol>> {
    let metas = attr.meta_item_list()?;

    let mut candidates = Vec::new();

    for meta in metas {
        let MetaItemInner::Lit(meta_lit) = meta else {
            return None;
        };
        candidates.push(meta_lit.symbol);
    }

    Some(candidates)
}
