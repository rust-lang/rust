// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::{meta_item, item, expr, ident};
use codemap::span;
use ext::base::ExtCtxt;
use ext::build::{AstBuilder, Duplicate, Field};
use ext::deriving::generic::*;

pub fn expand_deriving_rand(cx: @ExtCtxt,
                            span: span,
                            mitem: @meta_item,
                            in_items: ~[@item])
    -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~["core", "rand", "Rand"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "rand",
                generics: LifetimeBounds {
                    lifetimes: ~[],
                    bounds: ~[("R",
                               ~[ Path::new(~["core", "rand", "Rng"]) ])]
                },
                explicit_self: None,
                args: ~[
                    Ptr(~Literal(Path::new_local("R")),
                        Borrowed(None, ast::m_mutbl))
                ],
                ret_ty: Self,
                const_nonmatching: false,
                combine_substructure: rand_substructure
            }
        ]
    };

    expand_deriving_generic(cx, span, mitem, in_items, &trait_def)
}

fn rand_substructure(cx: @ExtCtxt, span: span, substr: &Substructure) -> @expr {
    let rng = match substr.nonself_args {
        [rng] => ~[ rng ],
        _ => cx.bug("Incorrect number of arguments to `rand` in `deriving(Rand)`")
    };
    let rand_ident = ~[
        cx.ident_of("core"),
        cx.ident_of("rand"),
        cx.ident_of("Rand"),
        cx.ident_of("rand")
    ];
    let rand_call = || {
        cx.mk_call_global(
                              span,
                              copy rand_ident,
                              ~[ rng[0].duplicate(cx) ])
    };

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            rand_thing(cx, span, substr.type_ident, summary, rand_call)
        }
        StaticEnum(_, ref variants) => {
            if variants.is_empty() {
                cx.span_fatal(span, "`Rand` cannot be derived for enums with no variants");
            }

            let variant_count = cx.mk_uint(span, variants.len());

            // need to specify the uint-ness of the random number
            let u32_ty = cx.mk_ty_path(span, ~[cx.ident_of("uint")]);
            let r_ty = cx.mk_ty_path(span, ~[cx.ident_of("R")]);
            let rand_name = cx.mk_raw_path_(span, copy rand_ident, None, ~[ u32_ty, r_ty ]);
            let rand_name = cx.mk_path_raw(span, rand_name);

            let rv_call = cx.mk_call_(
                                          span,
                                          rand_name,
                                          ~[ rng[0].duplicate(cx) ]);

            // rand() % variants.len()
            let rand_variant = cx.mk_binary(span, ast::rem,
                                                rv_call, variant_count);

            let mut arms = do variants.mapi |i, id_sum| {
                let i_expr = cx.mk_uint(span, i);
                let pat = cx.mk_pat_lit(span, i_expr);

                match *id_sum {
                    (ident, ref summary) => {
                        cx.mk_arm(span,
                                      ~[ pat ],
                                      rand_thing(cx, span, ident, summary, rand_call))
                    }
                }
            };

            // _ => {} at the end. Should never occur
            arms.push(cx.mk_unreachable_arm(span));

            cx.mk_expr(span,
                           ast::expr_match(rand_variant, arms))
        }
        _ => cx.bug("Non-static method in `deriving(Rand)`")
    };

    fn rand_thing(cx: @ExtCtxt, span: span,
                  ctor_ident: ident,
                  summary: &Either<uint, ~[ident]>,
                  rand_call: &fn() -> @expr) -> @expr {
        let ctor_ident = ~[ ctor_ident ];
        match *summary {
            Left(copy count) => {
                if count == 0 {
                    cx.mk_path(span, ctor_ident)
                } else {
                    let exprs = vec::from_fn(count, |_| rand_call());
                    cx.mk_call(span, ctor_ident, exprs)
                }
            }
            Right(ref fields) => {
                let rand_fields = do fields.map |ident| {
                    Field {
                        ident: *ident,
                        ex: rand_call()
                    }
                };
                cx.mk_struct_e(span, ctor_ident, rand_fields)
            }
        }
    }
}
