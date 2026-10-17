use rustc_expand::base::ExtCtxt;
use rustc_span::Span;

use crate::deriving::generic::*;
use crate::util::path_std;

pub(crate) fn expand_deriving_copy(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    let trait_def = TraitDef {
        span,
        path: path_std!(cx, span, marker::Copy),
        needs_copy_as_bound_if_packed: false,
        additional_bounds: SmallVec::new(),
        supports_unions: true,
        methods: SmallVec::new(),
        is_const,
        ..
    };

    trait_def.expand(cx, item, push);
}
