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
use attr::{AttrMetaMethods, AttributeMethods};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::quote::rt::ExtParseUtils;
use fold::Folder;
use parse::token::InternedString;

pub fn expand_deriving_default(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: @MetaItem,
                            item: @Item,
                            push: |@Item|) {
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "default", "Default")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "default",
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: Vec::new(),
                ret_ty: Self,
                attributes: attrs,
                const_nonmatching: false,
                combine_substructure: combine_substructure(|a, b, c| {
                    default_substructure(a, b, c)
                })
            })
    };
    trait_def.expand(cx, mitem, item, push)
}

fn default_substructure(cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> @Expr {
    let default_ident = vec!(
        cx.ident_of("std"),
        cx.ident_of("default"),
        cx.ident_of("Default"),
        cx.ident_of("default")
    );
    let default_call = |cx: &mut ExtCtxt, span| cx.expr_call_global(span, default_ident.clone(),
                                                                    Vec::new());

    return match *substr.fields {
        StaticStruct(struct_def, ref summary) => {
            match *summary {
                Unnamed(ref fields) => {
                    if fields.is_empty() {
                        cx.expr_ident(trait_span, substr.type_ident)
                    } else {
                        let exprs = fields.iter().zip(struct_def.fields.iter()).map(|(sp, field)| {
                            match get_default_attr_expr(cx, field) {
                                None => default_call(cx, *sp),
                                Some(e) => e
                            }
                        }).collect();
                        cx.expr_call_ident(trait_span, substr.type_ident, exprs)
                    }
                }
                Named(ref fields) => {
                    let exprs = fields.iter().zip(struct_def.fields.iter())
                                      .map(|(&(ident, span), field)| {
                        let expr = match get_default_attr_expr(cx, field) {
                            None => default_call(cx, span),
                            Some(e) => e
                        };
                        cx.field_imm(span, ident, expr)
                    }).collect();
                    cx.expr_struct_ident(trait_span, substr.type_ident, exprs)
                }
            }
        }
        StaticEnum(..) => {
            cx.span_err(trait_span, "`Default` cannot be derived for enums, only structs");
            // let compilation continue
            cx.expr_uint(trait_span, 0)
        }
        _ => cx.span_bug(trait_span, "Non-static method in `deriving(Default)`")
    };
}

fn get_default_attr_expr(cx: &mut ExtCtxt, field: &ast::StructField) -> Option<@Expr> {
    let attrs = field.node.attrs.as_slice();
    attrs.iter().find(|at| at.name().get() == "default").and_then(|at| {
        match at.meta().node {
            ast::MetaNameValue(_, ref v) => {
                match v.node {
                    ast::LitStr(ref s, _) => {
                        let s = s.get().to_strbuf();
                        let expr = cx.parse_expr(s);
                        let mut folder = SpanFolder { sp: v.span };
                        Some(folder.fold_expr(expr))

                    }
                    _ => {
                        cx.span_err(v.span,
                            "non-string literals are not allowed in `#[default]` attribute");
                        None
                    }
                }
            }
            _ => {
                cx.span_err(at.span, "`#[default]` attribute must have a value");
                None
            }
        }
    })
}

struct SpanFolder {
    sp: Span
}

impl Folder for SpanFolder {
    fn new_span(&mut self, _sp: Span) -> Span {
        self.sp
    }
}
