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

use std::vec_ng::Vec;

pub fn expand_deriving_hash(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: @MetaItem,
                            item: @Item,
                            push: |@Item|) {

    let hash_trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "hash", "Hash")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "hash",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(Ptr(~Literal(Path::new(vec!("std", "hash", "sip", "SipState"))),
                            Borrowed(None, MutMutable))),
                ret_ty: nil_ty(),
                inline: true,
                const_nonmatching: false,
                combine_substructure: hash_substructure
            }
        )
    };

    hash_trait_def.expand(cx, mitem, item, push);
}

fn hash_substructure(cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> @Expr {
    let state_expr = match substr.nonself_args {
        [state_expr] => state_expr,
        _ => cx.span_bug(trait_span, "incorrect number of arguments in `deriving(Hash)`")
    };
    let hash_ident = substr.method_ident;
    let call_hash = |span, thing_expr| {
        let expr = cx.expr_method_call(span, thing_expr, hash_ident, vec!(state_expr));
        cx.stmt_expr(expr)
    };
    let mut stmts = Vec::new();

    let fields = match *substr.fields {
        Struct(ref fs) => fs,
        EnumMatching(index, variant, ref fs) => {
            // Determine the discriminant. We will feed this value to the byte
            // iteration function.
            let discriminant = match variant.node.disr_expr {
                Some(d) => d,
                None => cx.expr_uint(trait_span, index)
            };

            stmts.push(call_hash(trait_span, discriminant));

            fs
        }
        _ => cx.span_bug(trait_span, "impossible substructure in `deriving(Hash)`")
    };

    for &FieldInfo { self_, span, .. } in fields.iter() {
        stmts.push(call_hash(span, self_));
    }

    if stmts.len() == 0 {
        cx.span_bug(trait_span, "#[deriving(Hash)] needs at least one field");
    }

    cx.expr_block(cx.block(trait_span, stmts, None))
}
