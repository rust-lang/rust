use crate::deriving::path_std;
use crate::deriving::generic::*;
use crate::deriving::generic::ty::*;

use syntax::ast::MetaItem;
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax_pos::Span;

pub fn expand_deriving_copy(cx: &mut ExtCtxt<'_>,
                            span: Span,
                            mitem: &MetaItem,
                            item: &Annotatable,
                            push: &mut dyn FnMut(Annotatable)) {
    let trait_def = TraitDef {
        span,
        attributes: vec![],
        path: path_std!(cx, marker::Copy),
        additional_bounds: vec![],
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: true,
        methods: vec![],
        associated_types: vec![],
    };

    trait_def.expand(cx, mitem, item, push);
}
