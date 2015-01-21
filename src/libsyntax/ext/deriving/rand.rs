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
use ast::{MetaItem, Item, Expr};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::{AstBuilder};
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use ptr::P;

pub fn expand_deriving_rand<F>(cx: &mut ExtCtxt,
                               span: Span,
                               mitem: &MetaItem,
                               item: &Item,
                               push: F) where
    F: FnOnce(P<Item>),
{
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
                                  vec!( Path::new(vec!("std", "rand", "Rng")) )))
                },
                explicit_self: None,
                args: vec!(
                    Ptr(box Literal(Path::new_local("R")),
                        Borrowed(None, ast::MutMutable))
                ),
                ret_ty: Self,
                attributes: Vec::new(),
                combine_substructure: combine_substructure(box |a, b, c| {
                    rand_substructure(a, b, c)
                })
            }
        )
    };
    trait_def.expand(cx, mitem, item, push)
}

fn rand_substructure(cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> P<Expr> {
    let rng = match substr.nonself_args {
        [ref rng] => rng,
        _ => cx.bug("Incorrect number of arguments to `rand` in `derive(Rand)`")
    };
    let rand_ident = vec!(
        cx.ident_of("std"),
        cx.ident_of("rand"),
        cx.ident_of("Rand"),
        cx.ident_of("rand")
    );
    let mut rand_call = |&mut: cx: &mut ExtCtxt, span| {
        cx.expr_call_global(span,
                            rand_ident.clone(),
                            vec!(rng.clone()))
    };

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            let path = cx.path_ident(trait_span, substr.type_ident);
            rand_thing(cx, trait_span, path, summary, rand_call)
        }
        StaticEnum(_, ref variants) => {
            if variants.is_empty() {
                cx.span_err(trait_span, "`Rand` cannot be derived for enums with no variants");
                // let compilation continue
                return cx.expr_usize(trait_span, 0);
            }

            let variant_count = cx.expr_usize(trait_span, variants.len());

            let rand_name = cx.path_all(trait_span,
                                        true,
                                        rand_ident.clone(),
                                        Vec::new(),
                                        Vec::new(),
                                        Vec::new());
            let rand_name = cx.expr_path(rand_name);

            // ::rand::Rand::rand(rng)
            let rv_call = cx.expr_call(trait_span,
                                       rand_name,
                                       vec!(rng.clone()));

            // need to specify the usize-ness of the random number
            let usize_ty = cx.ty_ident(trait_span, cx.ident_of("usize"));
            let value_ident = cx.ident_of("__value");
            let let_statement = cx.stmt_let_typed(trait_span,
                                                  false,
                                                  value_ident,
                                                  usize_ty,
                                                  rv_call);

            // rand() % variants.len()
            let value_ref = cx.expr_ident(trait_span, value_ident);
            let rand_variant = cx.expr_binary(trait_span,
                                              ast::BiRem,
                                              value_ref,
                                              variant_count);

            let mut arms = variants.iter().enumerate().map(|(i, &(ident, v_span, ref summary))| {
                let i_expr = cx.expr_usize(v_span, i);
                let pat = cx.pat_lit(v_span, i_expr);

                let path = cx.path(v_span, vec![substr.type_ident, ident]);
                let thing = rand_thing(cx, v_span, path, summary, |cx, sp| rand_call(cx, sp));
                cx.arm(v_span, vec!( pat ), thing)
            }).collect::<Vec<ast::Arm> >();

            // _ => {} at the end. Should never occur
            arms.push(cx.arm_unreachable(trait_span));

            let match_expr = cx.expr_match(trait_span, rand_variant, arms);

            let block = cx.block(trait_span, vec!( let_statement ), Some(match_expr));
            cx.expr_block(block)
        }
        _ => cx.bug("Non-static method in `derive(Rand)`")
    };

    fn rand_thing<F>(cx: &mut ExtCtxt,
                     trait_span: Span,
                     ctor_path: ast::Path,
                     summary: &StaticFields,
                     mut rand_call: F)
                     -> P<Expr> where
        F: FnMut(&mut ExtCtxt, Span) -> P<Expr>,
    {
        let path = cx.expr_path(ctor_path.clone());
        match *summary {
            Unnamed(ref fields) => {
                if fields.is_empty() {
                    path
                } else {
                    let exprs = fields.iter().map(|span| rand_call(cx, *span)).collect();
                    cx.expr_call(trait_span, path, exprs)
                }
            }
            Named(ref fields) => {
                let rand_fields = fields.iter().map(|&(ident, span)| {
                    let e = rand_call(cx, span);
                    cx.field_imm(span, ident, e)
                }).collect();
                cx.expr_struct(trait_span, ctor_path, rand_fields)
            }
        }
    }
}
