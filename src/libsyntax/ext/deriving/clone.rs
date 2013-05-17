// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast::{meta_item, item, expr};
use codemap::span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;

pub fn expand_deriving_clone(cx: @ExtCtxt,
                             span: span,
                             mitem: @meta_item,
                             in_items: ~[@item])
                          -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~["core", "clone", "Clone"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "clone",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[],
                ret_ty: Self,
                const_nonmatching: false,
                combine_substructure: |c, s, sub| cs_clone("Clone", c, s, sub)
            }
        ]
    };

    expand_deriving_generic(cx, span,
                            mitem, in_items,
                            &trait_def)
}

pub fn expand_deriving_deep_clone(cx: @ExtCtxt,
                                 span: span,
                                 mitem: @meta_item,
                                 in_items: ~[@item])
    -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~["core", "clone", "DeepClone"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "deep_clone",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[],
                ret_ty: Self,
                const_nonmatching: false,
                // cs_clone uses the ident passed to it, i.e. it will
                // call deep_clone (not clone) here.
                combine_substructure: |c, s, sub| cs_clone("DeepClone", c, s, sub)
            }
        ]
    };

    expand_deriving_generic(cx, span,
                            mitem, in_items,
                            &trait_def)
}

fn cs_clone(
    name: &str,
    cx: @ExtCtxt, span: span,
    substr: &Substructure) -> @expr {
    let clone_ident = substr.method_ident;
    let ctor_ident;
    let all_fields;
    let subcall = |field|
        cx.expr_method_call(span, field, clone_ident, ~[]);

    match *substr.fields {
        Struct(ref af) => {
            ctor_ident = substr.type_ident;
            all_fields = af;
        }
        EnumMatching(_, variant, ref af) => {
            ctor_ident = variant.node.name;
            all_fields = af;
        },
        EnumNonMatching(*) => cx.span_bug(span,
                                          fmt!("Non-matching enum variants in `deriving(%s)`",
                                               name)),
        StaticEnum(*) | StaticStruct(*) => cx.span_bug(span,
                                                       fmt!("Static method in `deriving(%s)`",
                                                            name))
    }

    match *all_fields {
        [(None, _, _), .. _] => {
            // enum-like
            let subcalls = all_fields.map(|&(_, self_f, _)| subcall(self_f));
            cx.expr_call_ident(span, ctor_ident, subcalls)
        },
        _ => {
            // struct-like
            let fields = do all_fields.map |&(o_id, self_f, _)| {
                let ident = match o_id {
                    Some(i) => i,
                    None => cx.span_bug(span,
                                        fmt!("unnamed field in normal struct in `deriving(%s)`",
                                             name))
                };
                cx.field_imm(span, ident, subcall(self_f))
            };

            if fields.is_empty() {
                // no fields, so construct like `None`
                cx.expr_ident(span, ctor_ident)
            } else {
                cx.expr_struct_ident(span, ctor_ident, fields)
            }
        }
    }
}
