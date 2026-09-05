use rustc_ast::{MetaItem, Safety};
use rustc_expand::base::ExtCtxt;
use rustc_span::Span;

use crate::deriving::generic::*;
use crate::deriving::path_std;

pub(crate) fn expand_deriving_copy(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    let trait_def = TraitDef {
        span,
        path: path_std!(marker::Copy),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: false,
        additional_bounds: SmallVec::new(),
        supports_unions: true,
        methods: SmallVec::new(),
        associated_types: SmallVec::new(),
        is_const,
        safety: Safety::Default,
        document: true,
    };

    trait_def.expand(cx, mitem, item, push);
}
