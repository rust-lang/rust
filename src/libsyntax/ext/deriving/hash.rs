// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{MetaItem, Item, Expr, MutMutable};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use ptr::P;

pub fn expand_deriving_hash<F>(cx: &mut ExtCtxt,
                               span: Span,
                               mitem: &MetaItem,
                               item: &Item,
                               push: F) where
    F: FnOnce(P<Item>),
{

    let path = Path::new_(pathvec_std!(cx, core::hash::Hash), None,
                          vec!(), true);
    let arg = Path::new_local("__H");
    let hash_trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path,
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "hash",
                generics: LifetimeBounds {
                    lifetimes: Vec::new(),
                    bounds: vec![("__H",
                                  vec![path_std!(cx, core::hash::Hasher)])],
                },
                explicit_self: borrowed_explicit_self(),
                args: vec!(Ptr(box Literal(arg), Borrowed(None, MutMutable))),
                ret_ty: nil_ty(),
                attributes: vec![],
                combine_substructure: combine_substructure(box |a, b, c| {
                    hash_substructure(a, b, c)
                })
            }
        ),
        associated_types: Vec::new(),
    };

    hash_trait_def.expand(cx, mitem, item, push);
}

fn hash_substructure(cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> P<Expr> {
    let state_expr = match substr.nonself_args {
        [ref state_expr] => state_expr,
        _ => cx.span_bug(trait_span, "incorrect number of arguments in `derive(Hash)`")
    };
    let call_hash = |span, thing_expr| {
        let hash_path = {
            let strs = vec![
                cx.ident_of_std("core"),
                cx.ident_of("hash"),
                cx.ident_of("Hash"),
                cx.ident_of("hash"),
            ];

            cx.expr_path(cx.path_global(span, strs))
        };
        let ref_thing = cx.expr_addr_of(span, thing_expr);
        let expr = cx.expr_call(span, hash_path, vec!(ref_thing, state_expr.clone()));
        cx.stmt_expr(expr)
    };
    let mut stmts = Vec::new();

    let fields = match *substr.fields {
        Struct(ref fs) => fs,
        EnumMatching(index, variant, ref fs) => {
            // Determine the discriminant. We will feed this value to the byte
            // iteration function.
            let discriminant = match variant.node.disr_expr {
                Some(ref d) => d.clone(),
                None => cx.expr_usize(trait_span, index)
            };

            stmts.push(call_hash(trait_span, discriminant));

            fs
        }
        _ => cx.span_bug(trait_span, "impossible substructure in `derive(Hash)`")
    };

    for &FieldInfo { ref self_, span, .. } in fields {
        stmts.push(call_hash(span, self_.clone()));
    }

    cx.expr_block(cx.block(trait_span, stmts, None))
}
