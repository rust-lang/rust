use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;

pub fn expand(
    _ecx: &mut ExtCtxt<'_>,
    _span: Span,
    _meta_item: &rustc_ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {

    vec![item.clone()]
}

