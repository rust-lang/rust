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

pub fn expand_deriving_iter_bytes(cx: @ext_ctxt,
                                  span: span,
                                  mitem: @meta_item,
                                  in_items: ~[@item]) -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~[~"core", ~"to_bytes", ~"IterBytes"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: ~"iter_bytes",
                generics: LifetimeBounds::empty(),
                self_ty: borrowed_explicit_self(),
                args: ~[
                    Literal(Path::new(~[~"bool"])),
                    Literal(Path::new(~[~"core", ~"to_bytes", ~"Cb"]))
                ],
                ret_ty: nil_ty(),
                const_nonmatching: false,
                combine_substructure: iter_bytes_substructure
            }
        ]
    };

    expand_deriving_generic(cx, span, mitem, in_items, &trait_def)
}

pub fn expand_deriving_obsolete(cx: @ext_ctxt,
                                span: span,
                                _mitem: @meta_item,
                                in_items: ~[@item])
                             -> ~[@item] {
    cx.span_err(span, ~"`#[deriving_iter_bytes]` is obsolete; use `#[deriving(IterBytes)]` \
                        instead");
    in_items
}

fn iter_bytes_substructure(cx: @ext_ctxt, span: span, substr: &Substructure) -> @expr {
    let lsb0_f = match substr.nonself_args {
        [l, f] => ~[l, f],
        _ => cx.span_bug(span, "Incorrect number of arguments in `deriving(IterBytes)`")
    };
    let iter_bytes_ident = substr.method_ident;
    let call_iterbytes = |thing_expr| {
        build::mk_stmt(
            cx, span,
            build::mk_method_call(cx, span,
                                  thing_expr, iter_bytes_ident,
                                  copy lsb0_f))
    };
    let mut stmts = ~[];
    let fields;
    match *substr.fields {
        Struct(ref fs) => {
            fields = fs
        }
        EnumMatching(copy index, ref variant, ref fs) => {
            // Determine the discriminant. We will feed this value to the byte
            // iteration function.
            let discriminant = match variant.node.disr_expr {
                Some(copy d)=> d,
                None => build::mk_uint(cx, span, index)
            };

            stmts.push(call_iterbytes(discriminant));

            fields = fs;
        }
        _ => cx.span_bug(span, "Impossible substructure in `deriving(IterBytes)`")
    }

    for fields.each |&(_, field, _)| {
        stmts.push(call_iterbytes(field));
    }

    build::mk_block(cx, span, ~[], stmts, None)
}
