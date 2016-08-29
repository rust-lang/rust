// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use deriving::generic::*;
use deriving::generic::ty::*;

use syntax::ast::MetaItem;
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax_pos::Span;

pub fn expand_deriving_unsafe_bound(cx: &mut ExtCtxt,
                                    span: Span,
                                    _: &MetaItem,
                                    _: &Annotatable,
                                    _: &mut FnMut(Annotatable)) {
    cx.span_err(span, "this unsafe trait should be implemented explicitly");
}

pub fn expand_deriving_copy(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: &MetaItem,
                            item: &Annotatable,
                            push: &mut FnMut(Annotatable)) {
    let mut v = cx.crate_root.map(|s| vec![s]).unwrap_or(Vec::new());
    v.push("marker");
    v.push("Copy");
    let path = Path::new(v);

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path,
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: true,
        methods: Vec::new(),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push);
}
