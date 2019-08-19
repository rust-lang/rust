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
        attributes: Vec::new(),
        path: path_std!(cx, marker::Copy),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: true,
        methods: Vec::new(),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push);
}
