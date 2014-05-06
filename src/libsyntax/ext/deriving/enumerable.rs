// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{MetaItem, Item, Expr};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use parse::token::InternedString;

pub fn expand_deriving_enumerable(cx: &mut ExtCtxt,
                                  span: Span,
                                  mitem: @MetaItem,
                                  item: @Item,
                                  push: |@Item|) {
    let self_vec_path = Path::new_(vec!("std", "vec", "Vec"), None, vec!(~Self), true);
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "enums", "Enumerable")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "values",
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: Vec::new(),
                ret_ty: Literal(self_vec_path),
                attributes: attrs,
                const_nonmatching: false,
                combine_substructure: combine_substructure(|a, b, c| {
                    values_substructure(a, b, c)
                })
            })
    };
    trait_def.expand(cx, mitem, item, push)
}

fn values_substructure(cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> @Expr {
    let vec_new_ident = vec!(
        cx.ident_of("std"),
        cx.ident_of("vec"),
        cx.ident_of("Vec"),
        cx.ident_of("new")
    );

    let mk_vec_new_expr  = |span| cx.expr_call_global(span, vec_new_ident.clone(), vec!());
    let mk_vec_push_expr = |span, id, e| cx.expr_method_call(span, cx.expr_ident(span, id), cx.ident_of("push"), vec!(e));

    return match *substr.fields {
        StaticEnum(_, ref variants) => {
            let ret_ident = cx.ident_of("ret");
            let mut stmts = vec!(cx.stmt_let(trait_span, true, ret_ident, mk_vec_new_expr(trait_span)));
            let mut pushes = vec!();
            let mut clike = true;

            for &(ref ident, ref span, ref flds) in variants.iter() {
                match *flds {
                    Unnamed(ref flds) => {
                        if flds.len() > 0 {
                            clike = false;
                        } else {
                            pushes.push(cx.stmt_expr(mk_vec_push_expr(*span, ret_ident, cx.expr_ident(*span, *ident))));
                        }
                    }
                    Named(_) => {
                        clike = false;
                    }
                }
            }

            if !clike {
                cx.span_err(trait_span, "`Enumerable` can only be derived for C-like enums");
            } else {
                stmts.extend(pushes.move_iter());
            }

            let block = cx.block(trait_span, stmts, Some(cx.expr_ident(trait_span, ret_ident)));

            cx.expr_block(block)
        }
        StaticStruct(..) => {
            cx.span_err(trait_span, "`Enumerable` can only be derived for enums, not structs");
            // let compilation continue
            cx.expr_uint(trait_span, 0)
        }
        _ => cx.span_bug(trait_span, "Non-static method in `deriving(Enumerable)`")
    };
}
