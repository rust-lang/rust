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
use ext::deriving::generic::ty::*;
use parse::token::InternedString;

pub fn expand_deriving_zero(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: @MetaItem,
                            item: @Item,
                            push: |@Item|) {
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "num", "Zero")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "zero",
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: Vec::new(),
                ret_ty: Self,
                attributes: attrs.clone(),
                const_nonmatching: false,
                combine_substructure: combine_substructure(|a, b, c| {
                    zero_substructure(a, b, c)
                })
            },
            MethodDef {
                name: "is_zero",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: Vec::new(),
                ret_ty: Literal(Path::new(vec!("bool"))),
                attributes: attrs,
                const_nonmatching: false,
                combine_substructure: combine_substructure(|cx, span, substr| {
                    cs_and(|cx, span, _, _| cx.span_bug(span,
                                                        "Non-matching enum \
                                                         variant in \
                                                         deriving(Zero)"),
                           cx, span, substr)
                })
            }
        )
    };
    trait_def.expand(cx, mitem, item, push)
}

fn zero_substructure(cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> @Expr {
    let zero_ident = vec!(
        cx.ident_of("std"),
        cx.ident_of("num"),
        cx.ident_of("Zero"),
        cx.ident_of("zero")
    );
    let zero_call = |span| cx.expr_call_global(span, zero_ident.clone(), Vec::new());

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            match *summary {
                Unnamed(ref fields) => {
                    if fields.is_empty() {
                        cx.expr_ident(trait_span, substr.type_ident)
                    } else {
                        let exprs = fields.iter().map(|sp| zero_call(*sp)).collect();
                        cx.expr_call_ident(trait_span, substr.type_ident, exprs)
                    }
                }
                Named(ref fields) => {
                    let zero_fields = fields.iter().map(|&(ident, span)| {
                        cx.field_imm(span, ident, zero_call(span))
                    }).collect();
                    cx.expr_struct_ident(trait_span, substr.type_ident, zero_fields)
                }
            }
        }
        StaticEnum(..) => {
            cx.span_err(trait_span, "`Zero` cannot be derived for enums, only structs");
            // let compilation continue
            cx.expr_uint(trait_span, 0)
        }
        _ => cx.bug("Non-static method in `deriving(Zero)`")
    };
}
