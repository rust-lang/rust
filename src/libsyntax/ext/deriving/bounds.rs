// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{MetaItem, Item};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use ptr::P;

pub fn expand_deriving_unsafe_bound(cx: &mut ExtCtxt,
                                    span: Span,
                                    _: &MetaItem,
                                    _: &Item,
                                    _: &mut FnMut(P<Item>)) {
    cx.span_err(span, "this unsafe trait should be implemented explicitly");
}

pub fn expand_deriving_copy(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: &MetaItem,
                            item: &Item,
                            push: &mut FnMut(P<Item>)) {
    let pod_path = Path::new(vec![
        if cx.use_std { "std" } else { "core" },
        "marker",
        "Pod",
    ]);

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: pod_path.clone(),
        additional_bounds: Vec::new(),
        bound_self: true,
        generics: LifetimeBounds::empty(),
        methods: Vec::new(),
        associated_types: Vec::new(),
    };

    // impl<T: Pod> Pod for Foo<T> {}
    trait_def.expand(cx, mitem, item, push);

    let copy_path = Path::new(vec![
        if cx.use_std { "std" } else { "core" },
        "marker",
        "Copy",
    ]);

    let trait_def = TraitDef {
        path: copy_path,
        additional_bounds: vec![Ty::Literal(pod_path)],
        bound_self: false,
        .. trait_def
    };

    // impl<T: Pod> Copy for Foo<T> {}
    trait_def.expand(cx, mitem, item, push);
}

pub fn expand_deriving_pod(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: &MetaItem,
                            item: &Item,
                            push: &mut FnMut(P<Item>)) {
    let path = Path::new(vec![
        if cx.use_std { "std" } else { "core" },
        "marker",
        "Pod",
    ]);

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path,
        additional_bounds: Vec::new(),
        bound_self: true,
        generics: LifetimeBounds::empty(),
        methods: Vec::new(),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push);
}
