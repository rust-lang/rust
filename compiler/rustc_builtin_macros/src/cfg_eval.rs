use crate::util::check_builtin_macro_attribute;

use rustc_ast as ast;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_expand::config::cfg_eval;
use rustc_span::symbol::sym;
use rustc_span::Span;

crate fn expand(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    annotatable: Annotatable,
) -> Vec<Annotatable> {
    check_builtin_macro_attribute(ecx, meta_item, sym::cfg_eval);
    cfg_eval(ecx, annotatable)
}
