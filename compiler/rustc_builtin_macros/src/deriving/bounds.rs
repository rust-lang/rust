use crate::deriving::generic::*;
use crate::deriving::path_std;

use rustc_ast::MetaItem;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;

pub fn expand_deriving_copy(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let trait_def = TraitDef {
        span,
        path: path_std!(marker::Copy),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: false,
        additional_bounds: Vec::new(),
        supports_unions: true,
        methods: Vec::new(),
        associated_types: Vec::new(),
        is_const,
    };

    trait_def.expand(cx, mitem, item, push);
}

pub fn expand_deriving_const_param_ty(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let trait_def = TraitDef {
        span,
        path: path_std!(marker::ConstParamTy),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: false,
        additional_bounds: vec![ty::Ty::Path(path_std!(cmp::Eq))],
        supports_unions: false,
        methods: Vec::new(),
        associated_types: Vec::new(),
        is_const,
    };

    trait_def.expand(cx, mitem, item, push);
}
