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
use ext::build::{AstBuilder, Duplicate};
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
        cx.expr_call_global(span,
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

            let variant_count = cx.expr_uint(span, variants.len());

            // need to specify the uint-ness of the random number
            let u32_ty = cx.ty_ident(span, cx.ident_of("uint"));
            let r_ty = cx.ty_ident(span, cx.ident_of("R"));
            let rand_name = cx.path_all(span, false, copy rand_ident, None, ~[ u32_ty, r_ty ]);
            let rand_name = cx.expr_path(rand_name);

            let rv_call = cx.expr_call(span,
                                       rand_name,
                                       ~[ rng[0].duplicate(cx) ]);

            // rand() % variants.len()
            let rand_variant = cx.expr_binary(span, ast::rem,
                                                rv_call, variant_count);

            let mut arms = do variants.mapi |i, id_sum| {
                let i_expr = cx.expr_uint(span, i);
                let pat = cx.pat_lit(span, i_expr);

                match *id_sum {
                    (ident, ref summary) => {
                        cx.arm(span,
                               ~[ pat ],
                               rand_thing(cx, span, ident, summary, rand_call))
                    }
                }
            };

            // _ => {} at the end. Should never occur
            arms.push(cx.arm_unreachable(span));

            cx.expr_match(span, rand_variant, arms)
        }
        _ => cx.bug("Non-static method in `deriving(Rand)`")
    };

    fn rand_thing(cx: @ExtCtxt, span: span,
                  ctor_ident: ident,
                  summary: &Either<uint, ~[ident]>,
                  rand_call: &fn() -> @expr) -> @expr {
        match *summary {
            Left(copy count) => {
                if count == 0 {
                    cx.expr_ident(span, ctor_ident)
                } else {
                    let exprs = vec::from_fn(count, |_| rand_call());
                    cx.expr_call_ident(span, ctor_ident, exprs)
                }
            }
            Right(ref fields) => {
                let rand_fields = do fields.map |ident| {
                    cx.field_imm(span, *ident, rand_call())
                };
                cx.expr_struct_ident(span, ctor_ident, rand_fields)
            }
        }
    }
}
