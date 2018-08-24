use deriving::path_std;
use deriving::generic::*;
use deriving::generic::ty::*;
use syntax::ast::MetaItem;
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax_pos::Span;

pub fn expand_deriving_unsafe_bound(cx: &mut ExtCtxt,
                                    span: Span,
                                    _: &MetaItem,
                                    _: &Annotatable,
                                    _: &mut dyn FnMut(Annotatable)) {
    cx.span_err(span, "this unsafe trait should be implemented explicitly");
}

pub fn expand_deriving_copy(cx: &mut ExtCtxt,
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
