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
use ext::base::ext_ctxt;
use ext::build;
use ext::deriving::generic::*;

pub fn expand_deriving_rand(cx: @ext_ctxt,
                            span: span,
                            mitem: @meta_item,
                            in_items: ~[@item])
    -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~[~"core", ~"rand", ~"Rand"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: ~"rand",
                generics: LifetimeBounds {
                    lifetimes: ~[],
                    bounds: ~[(~"R",
                               ~[ Path::new(~[~"core", ~"rand", ~"Rng"]) ])]
                },
                self_ty: None,
                args: ~[
                    Ptr(~Literal(Path::new_local(~"R")),
                        Borrowed(None, ast::m_imm))
                ],
                ret_ty: Self,
                const_nonmatching: false,
                combine_substructure: rand_substructure
            }
        ]
    };

    expand_deriving_generic(cx, span, mitem, in_items, &trait_def)
}

fn rand_substructure(cx: @ext_ctxt, span: span, substr: &Substructure) -> @expr {
    let rng = match substr.nonself_args {
        [rng] => ~[ rng ],
        _ => cx.bug("Incorrect number of arguments to `rand` in `deriving(Rand)`")
    };
    let rand_ident = ~[
        cx.ident_of(~"core"),
        cx.ident_of(~"rand"),
        cx.ident_of(~"Rand"),
        cx.ident_of(~"rand")
    ];
    let rand_call = || {
        build::mk_call_global(cx, span,
                              copy rand_ident, copy rng)
    };

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            rand_thing(cx, span, substr.type_ident, summary, rand_call)
        }
        StaticEnum(_, ref variants) => {
            if variants.is_empty() {
                cx.span_fatal(span, "`Rand` cannot be derived for enums with no variants");
            }

            let variant_count = build::mk_uint(cx, span, variants.len());

            // need to specify the uint-ness of the random number
            let u32_ty = build::mk_ty_path(cx, span, ~[cx.ident_of(~"uint")]);
            let r_ty = build::mk_ty_path(cx, span, ~[cx.ident_of(~"R")]);
            let rand_name = build::mk_raw_path_(span, copy rand_ident, None, ~[ u32_ty, r_ty ]);
            let rand_name = build::mk_path_raw(cx, span, rand_name);

            let rv_call = build::mk_call_(cx, span, rand_name, copy rng);

            // rand() % variants.len()
            let rand_variant = build::mk_binary(cx, span, ast::rem,
                                                rv_call, variant_count);

            let mut arms = do variants.mapi |i, id_sum| {
                let i_expr = build::mk_uint(cx, span, i);
                let pat = build::mk_pat_lit(cx, span, i_expr);

                match *id_sum {
                    (ident, ref summary) => {
                        build::mk_arm(cx, span,
                                      ~[ pat ],
                                      rand_thing(cx, span, ident, summary, rand_call))
                    }
                }
            };

            // _ => {} at the end. Should never occur
            arms.push(build::mk_unreachable_arm(cx, span));

            build::mk_expr(cx, span,
                           ast::expr_match(rand_variant, arms))
        }
        _ => cx.bug("Non-static method in `deriving(Rand)`")
    };

    fn rand_thing(cx: @ext_ctxt, span: span,
                  ctor_ident: ident,
                  summary: &Either<uint, ~[ident]>,
                  rand_call: &fn() -> @expr) -> @expr {
        let ctor_ident = ~[ ctor_ident ];
        match *summary {
            Left(copy count) => {
                if count == 0 {
                    build::mk_path(cx, span, ctor_ident)
                } else {
                    let exprs = vec::from_fn(count, |_| rand_call());
                    build::mk_call(cx, span, ctor_ident, exprs)
                }
            }
            Right(ref fields) => {
                let rand_fields = do fields.map |ident| {
                    build::Field {
                        ident: *ident,
                        ex: rand_call()
                    }
                };
                build::mk_struct_e(cx, span, ctor_ident, rand_fields)
            }
        }
    }
}