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
use core::option::{None,Some};


pub fn expand_deriving_clone(cx: @ext_ctxt,
                             span: span,
                             mitem: @meta_item,
                             in_items: ~[@item])
                          -> ~[@item] {
    let trait_def = TraitDef {
        path: ~[~"core", ~"clone", ~"Clone"],
        additional_bounds: ~[],
        methods: ~[
            MethodDef {
                name: ~"clone",
                nargs: 0,
                output_type: None, // return Self
                const_nonmatching: false,
                combine_substructure: cs_clone
            }
        ]
    };

    expand_deriving_generic(cx, span,
                            mitem, in_items,
                            &trait_def)
}

pub fn expand_deriving_obsolete(cx: @ext_ctxt,
                                span: span,
                                _mitem: @meta_item,
                                in_items: ~[@item])
                             -> ~[@item] {
    cx.span_err(span, ~"`#[deriving_clone]` is obsolete; use `#[deriving(Clone)]` instead");
    in_items
}

fn cs_clone(cx: @ext_ctxt, span: span,
            substr: &Substructure) -> @expr {
    let clone_ident = substr.method_ident;
    let ctor_ident;
    let all_fields;
    let subcall = |field|
        build::mk_method_call(cx, span, field, clone_ident, ~[]);

    match *substr.fields {
        Struct(af) => {
            ctor_ident = ~[ substr.type_ident ];
            all_fields = af;
        }
        EnumMatching(_, variant, af) => {
            ctor_ident = ~[ variant.node.name ];
            all_fields = af;
        },
        EnumNonMatching(*) => cx.bug("Non-matching enum variants in `deriving(Clone)`")
    }

    match all_fields {
        [(None, _, _), .. _] => {
            // enum-like
            let subcalls = all_fields.map(|&(_, self_f, _)| subcall(self_f));
            build::mk_call(cx, span, ctor_ident, subcalls)
        },
        _ => {
            // struct-like
            let fields = do all_fields.map |&(o_id, self_f, _)| {
                let ident = match o_id {
                    Some(i) => i,
                    None => cx.span_bug(span,
                                        ~"unnamed field in normal struct \
                                          in `deriving(Clone)`")
                };
                build::Field { ident: ident, ex: subcall(self_f) }
            };

            if fields.is_empty() {
                // no fields, so construct like `None`
                build::mk_path(cx, span, ctor_ident)
            } else {
                build::mk_struct_e(cx, span,
                                   ctor_ident,
                                   fields)
            }
        }
    }
}
