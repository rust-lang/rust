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
use ast;
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use parse::token::InternedString;

pub fn expand_deriving_from_primitive(cx: &mut ExtCtxt,
                                      span: Span,
                                      mitem: @MetaItem,
                                      item: @Item,
                                      push: |@Item|) {
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new(vec!("std", "num", "FromPrimitive")),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "from_i64",
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: vec!(
                    Literal(Path::new(vec!("i64")))),
                ret_ty: Literal(Path::new_(vec!("std", "option", "Option"),
                                           None,
                                           vec!(box Self),
                                           true)),
                // #[inline] liable to cause code-bloat
                attributes: attrs.clone(),
                const_nonmatching: false,
                combine_substructure: combine_substructure(|c, s, sub| {
                    cs_from("i64", c, s, sub)
                }),
            },
            MethodDef {
                name: "from_u64",
                generics: LifetimeBounds::empty(),
                explicit_self: None,
                args: vec!(
                    Literal(Path::new(vec!("u64")))),
                ret_ty: Literal(Path::new_(vec!("std", "option", "Option"),
                                           None,
                                           vec!(box Self),
                                           true)),
                // #[inline] liable to cause code-bloat
                attributes: attrs,
                const_nonmatching: false,
                combine_substructure: combine_substructure(|c, s, sub| {
                    cs_from("u64", c, s, sub)
                }),
            })
    };

    trait_def.expand(cx, mitem, item, push)
}

fn cs_from(name: &str, cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> @Expr {
    let n = match substr.nonself_args {
        [n] => n,
        _ => cx.span_bug(trait_span, "incorrect number of arguments in `deriving(FromPrimitive)`")
    };

    match *substr.fields {
        StaticStruct(..) => {
            cx.span_err(trait_span, "`FromPrimitive` cannot be derived for structs");
            return cx.expr_fail(trait_span, InternedString::new(""));
        }
        StaticEnum(enum_def, _) => {
            if enum_def.variants.is_empty() {
                cx.span_err(trait_span,
                            "`FromPrimitive` cannot be derived for enums with no variants");
                return cx.expr_fail(trait_span, InternedString::new(""));
            }

            let mut arms = Vec::new();

            for variant in enum_def.variants.iter() {
                match variant.node.kind {
                    ast::TupleVariantKind(ref args) => {
                        if !args.is_empty() {
                            cx.span_err(trait_span,
                                        "`FromPrimitive` cannot be derived for \
                                        enum variants with arguments");
                            return cx.expr_fail(trait_span,
                                                InternedString::new(""));
                        }
                        let span = variant.span;

                        // expr for `$n == $variant as $name`
                        let variant = cx.expr_ident(span, variant.node.name);
                        let ty = cx.ty_ident(span, cx.ident_of(name));
                        let cast = cx.expr_cast(span, variant, ty);
                        let guard = cx.expr_binary(span, ast::BiEq, n, cast);

                        // expr for `Some($variant)`
                        let body = cx.expr_some(span, variant);

                        // arm for `_ if $guard => $body`
                        let arm = ast::Arm {
                            attrs: vec!(),
                            pats: vec!(cx.pat_wild(span)),
                            guard: Some(guard),
                            body: body,
                        };

                        arms.push(arm);
                    }
                    ast::StructVariantKind(_) => {
                        cx.span_err(trait_span,
                                    "`FromPrimitive` cannot be derived for enums \
                                    with struct variants");
                        return cx.expr_fail(trait_span,
                                            InternedString::new(""));
                    }
                }
            }

            // arm for `_ => None`
            let arm = ast::Arm {
                attrs: vec!(),
                pats: vec!(cx.pat_wild(trait_span)),
                guard: None,
                body: cx.expr_none(trait_span),
            };
            arms.push(arm);

            cx.expr_match(trait_span, n, arms)
        }
        _ => cx.span_bug(trait_span, "expected StaticEnum in deriving(FromPrimitive)")
    }
}
