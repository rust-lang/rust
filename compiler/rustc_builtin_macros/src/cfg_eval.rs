use crate::util::check_builtin_macro_attribute;

use rustc_ast::{self as ast, AstLike};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_expand::config::StripUnconfigured;
use rustc_span::symbol::sym;
use rustc_span::Span;

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::cfg_eval);

    let mut visitor =
        StripUnconfigured { sess: ecx.sess, features: ecx.ecfg.features, modified: false };
    let mut item = visitor.fully_configure(item);
    if visitor.modified {
        // Erase the tokens if cfg-stripping modified the item
        // This will cause us to synthesize fake tokens
        // when `nt_to_tokenstream` is called on this item.
        if let Some(tokens) = item.tokens_mut() {
            *tokens = None;
        }
    }
    vec![item]
}
