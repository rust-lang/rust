// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{MetaItem, Item, Expr};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;

pub fn expand_deriving_clone(cx: &mut ExtCtxt,
                             span: Span,
                             mitem: @MetaItem,
                             item: @Item,
                             push: |@Item|) {
    let trait_def = TraitDef {
        span: span,
        path: Path::new(~["std", "clone", "Clone"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "clone",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[],
                ret_ty: Self,
                inline: true,
                const_nonmatching: false,
                combine_substructure: |c, s, sub| cs_clone("Clone", c, s, sub)
            }
        ]
    };

    trait_def.expand(cx, mitem, item, push)
}

pub fn expand_deriving_deep_clone(cx: &mut ExtCtxt,
                                  span: Span,
                                  mitem: @MetaItem,
                                  item: @Item,
                                  push: |@Item|) {
    let trait_def = TraitDef {
        span: span,
        path: Path::new(~["std", "clone", "DeepClone"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "deep_clone",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[],
                ret_ty: Self,
                inline: true,
                const_nonmatching: false,
                // cs_clone uses the ident passed to it, i.e. it will
                // call deep_clone (not clone) here.
                combine_substructure: |c, s, sub| cs_clone("DeepClone", c, s, sub)
            }
        ]
    };

    trait_def.expand(cx, mitem, item, push)
}

fn cs_clone(
    name: &str,
    cx: &mut ExtCtxt, trait_span: Span,
    substr: &Substructure) -> @Expr {
    let clone_ident = substr.method_ident;
    let ctor_ident;
    let all_fields;
    let subcall = |field: &FieldInfo|
        cx.expr_method_call(field.span, field.self_, clone_ident, ~[]);

    match *substr.fields {
        Struct(ref af) => {
            ctor_ident = substr.type_ident;
            all_fields = af;
        }
        EnumMatching(_, variant, ref af) => {
            ctor_ident = variant.node.name;
            all_fields = af;
        },
        EnumNonMatching(..) => cx.span_bug(trait_span,
                                           format!("non-matching enum variants in `deriving({})`",
                                                  name)),
        StaticEnum(..) | StaticStruct(..) => cx.span_bug(trait_span,
                                                         format!("static method in `deriving({})`",
                                                                 name))
    }

    match *all_fields {
        [FieldInfo { name: None, .. }, ..] => {
            // enum-like
            let subcalls = all_fields.map(subcall);
            cx.expr_call_ident(trait_span, ctor_ident, subcalls)
        },
        _ => {
            // struct-like
            let fields = all_fields.map(|field| {
                let ident = match field.name {
                    Some(i) => i,
                    None => cx.span_bug(trait_span,
                                        format!("unnamed field in normal struct in `deriving({})`",
                                                name))
                };
                cx.field_imm(field.span, ident, subcall(field))
            });

            if fields.is_empty() {
                // no fields, so construct like `None`
                cx.expr_ident(trait_span, ctor_ident)
            } else {
                cx.expr_struct_ident(trait_span, ctor_ident, fields)
            }
        }
    }
}
