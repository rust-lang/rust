// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use ast::{meta_item, item, expr};
use codemap::span;
use ext::base::ext_ctxt;
use ext::build;
use ext::deriving::generic::*;

use core::option::Some;

pub fn expand_deriving_totaleq(cx: @ext_ctxt,
                          span: span,
                          mitem: @meta_item,
                          in_items: ~[@item]) -> ~[@item] {

    fn cs_equals(cx: @ext_ctxt, span: span, substr: &Substructure) -> @expr {
        cs_and(|cx, span, _| build::mk_bool(cx, span, false),
               cx, span, substr)
    }

    let trait_def = TraitDef {
        path: ~[~"core", ~"cmp", ~"TotalEq"],
        additional_bounds: ~[],
        methods: ~[
            MethodDef {
                name: ~"equals",
                output_type: Some(~[~"bool"]),
                nargs: 1,
                const_nonmatching: true,
                combine_substructure: cs_equals
            }
        ]
    };

    expand_deriving_generic(cx, span, mitem, in_items,
                            &trait_def)
}
