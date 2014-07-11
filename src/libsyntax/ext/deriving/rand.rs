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
use ast::{MetaItem, Item, Expr, Ident};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::{AstBuilder};
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;

use std::gc::Gc;

pub fn expand_deriving_rand(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: Gc<MetaItem>,
                            item: Gc<Item>,
                            push: |Gc<Item>|) {
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "rand", "Rand")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "rand",
                generics: LifetimeBounds {
                    lifetimes: Vec::new(),
                    bounds: vec!(("R",
                                  None,
                                  vec!( Path::new(vec!("std", "rand", "Rng")) )))
                },
                explicit_self: None,
                args: vec!(
                    Ptr(box Literal(Path::new_local("R")),
                        Borrowed(None, ast::MutMutable))
                ),
                ret_ty: Self,
                attributes: Vec::new(),
                combine_substructure: combine_substructure(|a, b, c| {
                    rand_substructure(a, b, c)
                })
            }
        )
    };
    trait_def.expand(cx, mitem, item, push)
}

fn rand_substructure(cx: &mut ExtCtxt, trait_span: Span,
                     substr: &Substructure) -> Gc<Expr> {
    let rng = match substr.nonself_args {
        [rng] => vec!( rng ),
        _ => cx.bug("Incorrect number of arguments to `rand` in `deriving(Rand)`")
    };
    let rand_ident = vec!(
        cx.ident_of("std"),
        cx.ident_of("rand"),
        cx.ident_of("Rand"),
        cx.ident_of("rand")
    );
    let rand_call = |cx: &mut ExtCtxt, span| {
        cx.expr_call_global(span,
                            rand_ident.clone(),
                            vec!( *rng.get(0) ))
    };

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            rand_thing(cx, trait_span, substr.type_ident, summary, rand_call)
        }
        StaticEnum(_, ref variants) => {
            if variants.is_empty() {
                cx.span_err(trait_span, "`Rand` cannot be derived for enums with no variants");
                // let compilation continue
                return cx.expr_uint(trait_span, 0);
            }

            let variant_count = cx.expr_uint(trait_span, variants.len());

            let rand_name = cx.path_all(trait_span,
                                        true,
                                        rand_ident.clone(),
                                        Vec::new(),
                                        Vec::new());
            let rand_name = cx.expr_path(rand_name);

            // ::rand::Rand::rand(rng)
            let rv_call = cx.expr_call(trait_span,
                                       rand_name,
                                       vec!( *rng.get(0) ));

            // need to specify the uint-ness of the random number
            let uint_ty = cx.ty_ident(trait_span, cx.ident_of("uint"));
            let value_ident = cx.ident_of("__value");
            let let_statement = cx.stmt_let_typed(trait_span,
                                                  false,
                                                  value_ident,
                                                  uint_ty,
                                                  rv_call);

            // rand() % variants.len()
            let value_ref = cx.expr_ident(trait_span, value_ident);
            let rand_variant = cx.expr_binary(trait_span,
                                              ast::BiRem,
                                              value_ref,
                                              variant_count);

            let mut arms = variants.iter().enumerate().map(|(i, &(ident, v_span, ref summary))| {
                let i_expr = cx.expr_uint(v_span, i);
                let pat = cx.pat_lit(v_span, i_expr);

                let thing = rand_thing(cx, v_span, ident, summary, |cx, sp| rand_call(cx, sp));
                cx.arm(v_span, vec!( pat ), thing)
            }).collect::<Vec<ast::Arm> >();

            // _ => {} at the end. Should never occur
            arms.push(cx.arm_unreachable(trait_span));

            let match_expr = cx.expr_match(trait_span, rand_variant, arms);

            let block = cx.block(trait_span, vec!( let_statement ), Some(match_expr));
            cx.expr_block(block)
        }
        _ => cx.bug("Non-static method in `deriving(Rand)`")
    };

    fn rand_thing(cx: &mut ExtCtxt,
                  trait_span: Span,
                  ctor_ident: Ident,
                  summary: &StaticFields,
                  rand_call: |&mut ExtCtxt, Span| -> Gc<Expr>)
                  -> Gc<Expr> {
        match *summary {
            Unnamed(ref fields) => {
                if fields.is_empty() {
                    cx.expr_ident(trait_span, ctor_ident)
                } else {
                    let exprs = fields.iter().map(|span| rand_call(cx, *span)).collect();
                    cx.expr_call_ident(trait_span, ctor_ident, exprs)
                }
            }
            Named(ref fields) => {
                let rand_fields = fields.iter().map(|&(ident, span)| {
                    let e = rand_call(cx, span);
                    cx.field_imm(span, ident, e)
                }).collect();
                cx.expr_struct_ident(trait_span, ctor_ident, rand_fields)
            }
        }
    }
}
