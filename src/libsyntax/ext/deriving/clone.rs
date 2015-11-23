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
use ast::{MetaItem, Expr};
use codemap::Span;
use ext::base::{ExtCtxt, Annotatable};
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use parse::token::InternedString;
use ptr::P;

pub fn expand_deriving_clone(cx: &mut ExtCtxt,
                             span: Span,
                             mitem: &MetaItem,
                             item: &Annotatable,
                             push: &mut FnMut(Annotatable))
{
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::clone::Clone),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        methods: vec!(
            MethodDef {
                name: "clone",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: Vec::new(),
                ret_ty: Self_,
                attributes: attrs.clone(),
                is_unsafe: false,
                combine_substructure: combine_substructure(Box::new(|c, s, sub| {
                    cs_clone("Clone", c, s, sub)
                })),
            },
            MethodDef {
                name: "clone_from",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_mut_explicit_self(),
                args: vec![borrowed_self()],
                ret_ty: nil_ty(),
                attributes: attrs,
                is_unsafe: false,
                combine_substructure: combine_substructure(Box::new(|c, s, sub| {
                    cs_clone_from("Clone", c, s, sub)
                })),
            }
        ),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push)
}

fn cs_clone(
    name: &str,
    cx: &mut ExtCtxt, trait_span: Span,
    substr: &Substructure) -> P<Expr> {
    let ctor_path;
    let all_fields;
    let fn_path = cx.std_path(&["clone", "Clone", "clone"]);
    let subcall = |field: &FieldInfo| {
        let args = vec![cx.expr_addr_of(field.span, field.self_.clone())];

        cx.expr_call_global(field.span, fn_path.clone(), args)
    };

    match *substr.fields {
        Struct(ref af) => {
            ctor_path = cx.path(trait_span, vec![substr.type_ident]);
            all_fields = af;
        }
        EnumMatching(_, variant, ref af) => {
            ctor_path = cx.path(trait_span, vec![substr.type_ident, variant.node.name]);
            all_fields = af;
        },
        EnumNonMatchingCollapsed (..) => {
            cx.span_bug(trait_span,
                        &format!("non-matching enum variants in \
                                 `derive({})`", name))
        }
        StaticEnum(..) | StaticStruct(..) => {
            cx.span_bug(trait_span,
                        &format!("static method in `derive({})`", name))
        }
    }

    if !all_fields.is_empty() && all_fields[0].name.is_none() {
        // enum-like
        let subcalls = all_fields.iter().map(subcall).collect();
        let path = cx.expr_path(ctor_path);
        cx.expr_call(trait_span, path, subcalls)
    } else {
        // struct-like
        let fields = all_fields.iter().map(|field| {
            let ident = match field.name {
                Some(i) => i,
                None => {
                    cx.span_bug(trait_span,
                                &format!("unnamed field in normal struct in \
                                         `derive({})`", name))
                }
            };
            cx.field_imm(field.span, ident, subcall(field))
        }).collect::<Vec<_>>();

        if fields.is_empty() {
            // no fields, so construct like `None`
            cx.expr_path(ctor_path)
        } else {
            cx.expr_struct(trait_span, ctor_path, fields)
        }
    }
}

fn cs_clone_from(
    name: &str,
    cx: &mut ExtCtxt, span: Span,
    substr: &Substructure) -> P<Expr> {

    let fn_path = cx.std_path(&["clone", "Clone", "clone_from"]);
    cs_call_global(
        |cx, span, exprs| {
            cx.expr_block(cx.block(span, exprs.into_iter().map(|expr| {
                cx.stmt_expr(expr)
            }).collect(), None))
        },
        Box::new(|cx, span, (self_args, _), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, &format!("not exactly 2 arguments in `derive({})`", name))
            } else {
                let clone_path = cx.std_path(&["clone", "Clone", "clone"]);
                let args = vec![cx.expr_addr_of(span, self_args[1].clone())];

                let rhs = cx.expr_call_global(span, clone_path, args);
                let lhs = self_args[0].clone();
                cx.expr(span, ast::ExprAssign(lhs, rhs))
            }
        }),
        cx, span, substr, fn_path, ast::MutMutable)
}
