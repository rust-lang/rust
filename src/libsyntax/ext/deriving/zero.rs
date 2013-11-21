// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{MetaItem, item, Expr};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;

pub fn expand_deriving_zero(cx: @ExtCtxt,
                            span: Span,
                            mitem: @MetaItem,
                            in_items: ~[@item])
    -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~["std", "num", "Zero"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "zero",
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: ~[],
                ret_ty: Self,
                inline: true,
                const_nonmatching: false,
                combine_substructure: zero_substructure
            },
            MethodDef {
                name: "is_zero",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[],
                ret_ty: Literal(Path::new(~["bool"])),
                inline: true,
                const_nonmatching: false,
                combine_substructure: |cx, span, substr| {
                    cs_and(|cx, span, _, _| cx.span_bug(span,
                                                        "Non-matching enum \
                                                         variant in \
                                                         deriving(Zero)"),
                           cx, span, substr)
                }
            }
        ]
    };
    trait_def.expand(cx, span, mitem, in_items)
}

fn zero_substructure(cx: @ExtCtxt, span: Span, substr: &Substructure) -> @Expr {
    let zero_ident = ~[
        cx.ident_of("std"),
        cx.ident_of("num"),
        cx.ident_of("Zero"),
        cx.ident_of("zero")
    ];
    let zero_call = |span| cx.expr_call_global(span, zero_ident.clone(), ~[]);

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            match *summary {
                Unnamed(ref fields) => {
                    if fields.is_empty() {
                        cx.expr_ident(span, substr.type_ident)
                    } else {
                        let exprs = fields.map(|sp| zero_call(*sp));
                        cx.expr_call_ident(span, substr.type_ident, exprs)
                    }
                }
                Named(ref fields) => {
                    let zero_fields = fields.map(|&(ident, span)| {
                        cx.field_imm(span, ident, zero_call(span))
                    });
                    cx.expr_struct_ident(span, substr.type_ident, zero_fields)
                }
            }
        }
        StaticEnum(*) => {
            cx.span_fatal(span, "`Zero` cannot be derived for enums, \
                                 only structs")
        }
        _ => cx.bug("Non-static method in `deriving(Zero)`")
    };
}
