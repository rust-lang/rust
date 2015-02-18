// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{MetaItem, MetaWord, Item};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use ptr::P;

pub fn expand_deriving_bound<F>(cx: &mut ExtCtxt,
                                span: Span,
                                mitem: &MetaItem,
                                item: &Item,
                                push: F) where
    F: FnOnce(P<Item>),
{
    let name = match mitem.node {
        MetaWord(ref tname) => {
            match &tname[..] {
                "Copy" => "Copy",
                "Send" | "Sync" => {
                    return cx.span_err(span,
                                       &format!("{} is an unsafe trait and it \
                                                 should be implemented explicitly",
                                                *tname))
                }
                ref tname => {
                    cx.span_bug(span,
                                &format!("expected built-in trait name but \
                                          found {}", *tname))
                }
            }
        },
        _ => {
            return cx.span_err(span, "unexpected value in deriving, expected \
                                      a trait")
        }
    };

    let path = Path::new(vec![
        if cx.use_std { "std" } else { "core" },
        "marker",
        name
    ]);

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path,
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: Vec::new(),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push)
}
