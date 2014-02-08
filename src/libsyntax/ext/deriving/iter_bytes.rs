// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{MetaItem, Item, Expr, BiAnd};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;


pub fn expand_deriving_iter_bytes(cx: &mut ExtCtxt,
                                  span: Span,
                                  mitem: @MetaItem,
                                  in_items: ~[@Item]) -> ~[@Item] {
    let trait_def = TraitDef {
        cx: cx, span: span,

        path: Path::new(~["std", "to_bytes", "IterBytes"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "iter_bytes",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: ~[
                    Literal(Path::new(~["bool"])),
                    Literal(Path::new(~["std", "to_bytes", "Cb"]))
                ],
                ret_ty: Literal(Path::new(~["bool"])),
                inline: true,
                const_nonmatching: false,
                combine_substructure: iter_bytes_substructure
            }
        ]
    };

    trait_def.expand(mitem, in_items)
}

fn iter_bytes_substructure(cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> @Expr {
    let (lsb0, f)= match substr.nonself_args {
        [l, f] => (l, f),
        _ => cx.span_bug(trait_span, "Incorrect number of arguments in `deriving(IterBytes)`")
    };
    // Build the "explicitly borrowed" stack closure, "|_buf| f(_buf)".
    let blk_arg = cx.ident_of("_buf");
    let borrowed_f =
        cx.lambda_expr_1(trait_span,
                         cx.expr_call(trait_span, f, ~[cx.expr_ident(trait_span, blk_arg)]),
                         blk_arg);

    let iter_bytes_ident = substr.method_ident;
    let call_iterbytes = |span, thing_expr| {
        cx.expr_method_call(span,
                            thing_expr,
                            iter_bytes_ident,
                            ~[lsb0, borrowed_f])
    };
    let mut exprs = ~[];
    let fields;
    match *substr.fields {
        Struct(ref fs) => {
            fields = fs
        }
        EnumMatching(index, ref variant, ref fs) => {
            // Determine the discriminant. We will feed this value to the byte
            // iteration function.
            let discriminant = match variant.node.disr_expr {
                Some(d)=> d,
                None => cx.expr_uint(trait_span, index)
            };

            exprs.push(call_iterbytes(trait_span, discriminant));

            fields = fs;
        }
        _ => cx.span_bug(trait_span, "Impossible substructure in `deriving(IterBytes)`")
    }

    for &FieldInfo { self_, span, .. } in fields.iter() {
        exprs.push(call_iterbytes(span, self_));
    }

    if exprs.len() == 0 {
        cx.span_bug(trait_span, "#[deriving(IterBytes)] needs at least one field");
    }

    exprs.slice(1, exprs.len()).iter().fold(exprs[0], |prev, me| {
        cx.expr_binary(trait_span, BiAnd, prev, *me)
    })
}
