use rustc_ast as ast;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Span, sym};

pub(crate) fn expand_deriving_from(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &ast::MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
}
