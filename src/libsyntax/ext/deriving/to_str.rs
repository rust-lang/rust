// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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

pub fn expand_deriving_to_str(cx: @ext_ctxt,
                              span: span,
                              mitem: @meta_item,
                              in_items: ~[@item])
    -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~[~"core", ~"to_str", ~"ToStr"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: ~"to_str",
                generics: LifetimeBounds::empty(),
                self_ty: borrowed_explicit_self(),
                args: ~[],
                ret_ty: Ptr(~Literal(Path::new_local(~"str")), Owned),
                const_nonmatching: false,
                combine_substructure: to_str_substructure
            }
        ]
    };

    expand_deriving_generic(cx, span, mitem, in_items, &trait_def)
}

fn to_str_substructure(cx: @ext_ctxt, span: span, substr: &Substructure) -> @expr {
    match substr.self_args {
        [self_obj] => {
            let self_addr = build::mk_addr_of(cx, span, self_obj);
            build::mk_call_global(cx, span,
                                  ~[cx.ident_of("core"),
                                    cx.ident_of("sys"),
                                    cx.ident_of("log_str")],
                                  ~[self_addr])
        }
        _ => cx.span_bug(span, ~"Invalid number of arguments in `deriving(ToStr)`")
    }
}
