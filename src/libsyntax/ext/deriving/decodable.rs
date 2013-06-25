// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
The compiler code necessary for #[deriving(Decodable)]. See
encodable.rs for more.
*/

use std::vec;
use std::uint;

use ast::{meta_item, item, expr, m_mutbl};
use codemap::span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;

pub fn expand_deriving_decodable(cx: @ExtCtxt,
                                 span: span,
                                 mitem: @meta_item,
                                 in_items: ~[@item]) -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new_(~["extra", "serialize", "Decodable"], None,
                         ~[~Literal(Path::new_local("__D"))], true),
        additional_bounds: ~[],
        generics: LifetimeBounds {
            lifetimes: ~[],
            bounds: ~[("__D", ~[Path::new(~["extra", "serialize", "Decoder"])])],
        },
        methods: ~[
            MethodDef {
                name: "decode",
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: ~[Ptr(~Literal(Path::new_local("__D")),
                            Borrowed(None, m_mutbl))],
                ret_ty: Self,
                const_nonmatching: true,
                combine_substructure: decodable_substructure,
            },
        ]
    };

    trait_def.expand(cx, span, mitem, in_items)
}

fn decodable_substructure(cx: @ExtCtxt, span: span,
                          substr: &Substructure) -> @expr {
    let decoder = substr.nonself_args[0];
    let recurse = ~[cx.ident_of("extra"),
                    cx.ident_of("serialize"),
                    cx.ident_of("Decodable"),
                    cx.ident_of("decode")];
    // throw an underscore in front to suppress unused variable warnings
    let blkarg = cx.ident_of("_d");
    let blkdecoder = cx.expr_ident(span, blkarg);
    let calldecode = cx.expr_call_global(span, recurse, ~[blkdecoder]);
    let lambdadecode = cx.lambda_expr_1(span, calldecode, blkarg);

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            let nfields = match *summary {
                Left(n) => n, Right(ref fields) => fields.len()
            };
            let read_struct_field = cx.ident_of("read_struct_field");

            let getarg = |name: @str, field: uint| {
                cx.expr_method_call(span, blkdecoder, read_struct_field,
                                    ~[cx.expr_str(span, name),
                                      cx.expr_uint(span, field),
                                      lambdadecode])
            };

            let result = match *summary {
                Left(n) => {
                    if n == 0 {
                        cx.expr_ident(span, substr.type_ident)
                    } else {
                        let mut fields = vec::with_capacity(n);
                        for uint::range(0, n) |i| {
                            fields.push(getarg(fmt!("_field%u", i).to_managed(), i));
                        }
                        cx.expr_call_ident(span, substr.type_ident, fields)
                    }
                }
                Right(ref fields) => {
                    let fields = do fields.mapi |i, f| {
                        cx.field_imm(span, *f, getarg(cx.str_of(*f), i))
                    };
                    cx.expr_struct_ident(span, substr.type_ident, fields)
                }
            };

            cx.expr_method_call(span, decoder, cx.ident_of("read_struct"),
                                ~[cx.expr_str(span, cx.str_of(substr.type_ident)),
                                  cx.expr_uint(span, nfields),
                                  cx.lambda_expr_1(span, result, blkarg)])
        }
        StaticEnum(_, ref fields) => {
            let variant = cx.ident_of("i");

            let mut arms = ~[];
            let mut variants = ~[];
            let rvariant_arg = cx.ident_of("read_enum_variant_arg");

            for fields.iter().enumerate().advance |(i, f)| {
                let (name, parts) = match *f { (i, ref p) => (i, p) };
                variants.push(cx.expr_str(span, cx.str_of(name)));

                let getarg = |field: uint| {
                    cx.expr_method_call(span, blkdecoder, rvariant_arg,
                                        ~[cx.expr_uint(span, field),
                                          lambdadecode])
                };

                let decoded = match *parts {
                    Left(n) => {
                        if n == 0 {
                            cx.expr_ident(span, name)
                        } else {
                            let mut fields = vec::with_capacity(n);
                            for uint::range(0, n) |i| {
                                fields.push(getarg(i));
                            }
                            cx.expr_call_ident(span, name, fields)
                        }
                    }
                    Right(ref fields) => {
                        let fields = do fields.mapi |i, f| {
                            cx.field_imm(span, *f, getarg(i))
                        };
                        cx.expr_struct_ident(span, name, fields)
                    }
                };
                arms.push(cx.arm(span,
                                 ~[cx.pat_lit(span, cx.expr_uint(span, i))],
                                 decoded));
            }

            arms.push(cx.arm_unreachable(span));

            let result = cx.expr_match(span, cx.expr_ident(span, variant), arms);
            let lambda = cx.lambda_expr(span, ~[blkarg, variant], result);
            let variant_vec = cx.expr_vec(span, variants);
            let result = cx.expr_method_call(span, blkdecoder,
                                             cx.ident_of("read_enum_variant"),
                                             ~[variant_vec, lambda]);
            cx.expr_method_call(span, decoder, cx.ident_of("read_enum"),
                                ~[cx.expr_str(span, cx.str_of(substr.type_ident)),
                                  cx.lambda_expr_1(span, result, blkarg)])
        }
        _ => cx.bug("expected StaticEnum or StaticStruct in deriving(Decodable)")
    };
}
