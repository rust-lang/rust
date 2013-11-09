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
use ast::{MetaItem, item, Expr, Ident};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::{AstBuilder};
use ext::deriving::generic::*;
use opt_vec;

pub fn expand_deriving_rand(cx: @ExtCtxt,
                            span: Span,
                            mitem: @MetaItem,
                            in_items: ~[@item])
    -> ~[@item] {
    let trait_def = TraitDef {
        path: Path::new(~["std", "rand", "Rand"]),
        additional_bounds: ~[],
        generics: LifetimeBounds::empty(),
        methods: ~[
            MethodDef {
                name: "rand",
                generics: LifetimeBounds {
                    lifetimes: ~[],
                    bounds: ~[("R",
                               ~[ Path::new(~["std", "rand", "Rng"]) ])]
                },
                explicit_self: None,
                args: ~[
                    Ptr(~Literal(Path::new_local("R")),
                        Borrowed(None, ast::MutMutable))
                ],
                ret_ty: Self,
                const_nonmatching: false,
                combine_substructure: rand_substructure
            }
        ]
    };
    trait_def.expand(cx, span, mitem, in_items)
}

fn rand_substructure(cx: @ExtCtxt, span: Span, substr: &Substructure) -> @Expr {
    let rng = match substr.nonself_args {
        [rng] => ~[ rng ],
        _ => cx.bug("Incorrect number of arguments to `rand` in `deriving(Rand)`")
    };
    let rand_ident = ~[
        cx.ident_of("std"),
        cx.ident_of("rand"),
        cx.ident_of("Rand"),
        cx.ident_of("rand")
    ];
    let rand_call = |span| {
        cx.expr_call_global(span,
                            rand_ident.clone(),
                            ~[ rng[0] ])
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

            let rand_name = cx.path_all(span,
                                        true,
                                        rand_ident.clone(),
                                        opt_vec::Empty,
                                        ~[]);
            let rand_name = cx.expr_path(rand_name);

            // ::std::rand::Rand::rand(rng)
            let rv_call = cx.expr_call(span,
                                       rand_name,
                                       ~[ rng[0] ]);

            // need to specify the uint-ness of the random number
            let uint_ty = cx.ty_ident(span, cx.ident_of("uint"));
            let value_ident = cx.ident_of("__value");
            let let_statement = cx.stmt_let_typed(span,
                                                  false,
                                                  value_ident,
                                                  uint_ty,
                                                  rv_call);

            // rand() % variants.len()
            let value_ref = cx.expr_ident(span, value_ident);
            let rand_variant = cx.expr_binary(span,
                                              ast::BiRem,
                                              value_ref,
                                              variant_count);

            let mut arms = do variants.iter().enumerate().map |(i, id_sum)| {
                let i_expr = cx.expr_uint(span, i);
                let pat = cx.pat_lit(span, i_expr);

                match *id_sum {
                    (ident, ref summary) => {
                        cx.arm(span,
                               ~[ pat ],
                               rand_thing(cx, span, ident, summary, |sp| rand_call(sp)))
                    }
                }
            }.collect::<~[ast::Arm]>();

            // _ => {} at the end. Should never occur
            arms.push(cx.arm_unreachable(span));

            let match_expr = cx.expr_match(span, rand_variant, arms);

            let block = cx.block(span, ~[ let_statement ], Some(match_expr));
            cx.expr_block(block)
        }
        _ => cx.bug("Non-static method in `deriving(Rand)`")
    };

    fn rand_thing(cx: @ExtCtxt, span: Span,
                  ctor_ident: Ident,
                  summary: &StaticFields,
                  rand_call: &fn(Span) -> @Expr) -> @Expr {
        match *summary {
            Unnamed(ref fields) => {
                if fields.is_empty() {
                    cx.expr_ident(span, ctor_ident)
                } else {
                    let exprs = fields.map(|span| rand_call(*span));
                    cx.expr_call_ident(span, ctor_ident, exprs)
                }
            }
            Named(ref fields) => {
                let rand_fields = do fields.map |&(ident, span)| {
                    cx.field_imm(span, ident, rand_call(span))
                };
                cx.expr_struct_ident(span, ctor_ident, rand_fields)
            }
        }
    }
}
