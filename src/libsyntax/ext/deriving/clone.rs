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
        attributes: Vec::new(),
        path: Path::new(vec!("std", "clone", "Clone")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "clone",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: Vec::new(),
                ret_ty: Self,
                inline: true,
                const_nonmatching: false,
                combine_substructure: combine_substructure(|c, s, sub| {
                    cs_clone("Clone", c, s, sub)
                }),
            }
        )
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
        cx.expr_method_call(field.span, field.self_, clone_ident, Vec::new());

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

    if all_fields.len() >= 1 && all_fields.get(0).name.is_none() {
        // enum-like
        let subcalls = all_fields.iter().map(subcall).collect();
        cx.expr_call_ident(trait_span, ctor_ident, subcalls)
    } else {
        // struct-like
        let fields = all_fields.iter().map(|field| {
            let ident = match field.name {
                Some(i) => i,
                None => cx.span_bug(trait_span,
                                    format!("unnamed field in normal struct in `deriving({})`",
                                            name))
            };
            cx.field_imm(field.span, ident, subcall(field))
        }).collect::<Vec<_>>();

        if fields.is_empty() {
            // no fields, so construct like `None`
            cx.expr_ident(trait_span, ctor_ident)
        } else {
            cx.expr_struct_ident(trait_span, ctor_ident, fields)
        }
    }
}
