// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The compiler code necessary for `#[derive(Decodable)]`. See encodable.rs for more.

use deriving::generic::*;
use deriving::generic::ty::*;

use syntax::ast;
use syntax::ast::{MetaItem, Expr, MutMutable};
use syntax::codemap::Span;
use syntax::ext::base::{ExtCtxt, Annotatable};
use syntax::ext::build::AstBuilder;
use syntax::parse::token::InternedString;
use syntax::parse::token;
use syntax::ptr::P;

pub fn expand_deriving_rustc_decodable(cx: &mut ExtCtxt,
                                       span: Span,
                                       mitem: &MetaItem,
                                       item: &Annotatable,
                                       push: &mut FnMut(Annotatable))
{
    expand_deriving_decodable_imp(cx, span, mitem, item, push, "rustc_serialize")
}

pub fn expand_deriving_decodable(cx: &mut ExtCtxt,
                                 span: Span,
                                 mitem: &MetaItem,
                                 item: &Annotatable,
                                 push: &mut FnMut(Annotatable))
{
    expand_deriving_decodable_imp(cx, span, mitem, item, push, "serialize")
}

fn expand_deriving_decodable_imp(cx: &mut ExtCtxt,
                                 span: Span,
                                 mitem: &MetaItem,
                                 item: &Annotatable,
                                 push: &mut FnMut(Annotatable),
                                 krate: &'static str)
{
    if cx.crate_root != Some("std") {
        // FIXME(#21880): lift this requirement.
        cx.span_err(span, "this trait cannot be derived with #![no_std] \
                           or #![no_core]");
        return
    }

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new_(vec!(krate, "Decodable"), None, vec!(), true),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        methods: vec!(
            MethodDef {
                name: "decode",
                generics: LifetimeBounds {
                    lifetimes: Vec::new(),
                    bounds: vec!(("__D", vec!(Path::new_(
                                    vec!(krate, "Decoder"), None,
                                    vec!(), true))))
                },
                explicit_self: None,
                args: vec!(Ptr(Box::new(Literal(Path::new_local("__D"))),
                            Borrowed(None, MutMutable))),
                ret_ty: Literal(Path::new_(
                    pathvec_std!(cx, core::result::Result),
                    None,
                    vec!(Box::new(Self_), Box::new(Literal(Path::new_(
                        vec!["__D", "Error"], None, vec![], false
                    )))),
                    true
                )),
                attributes: Vec::new(),
                is_unsafe: false,
                combine_substructure: combine_substructure(Box::new(|a, b, c| {
                    decodable_substructure(a, b, c, krate)
                })),
            }
        ),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push)
}

fn decodable_substructure(cx: &mut ExtCtxt, trait_span: Span,
                          substr: &Substructure,
                          krate: &str) -> P<Expr> {
    let decoder = substr.nonself_args[0].clone();
    let recurse = vec!(cx.ident_of(krate),
                    cx.ident_of("Decodable"),
                    cx.ident_of("decode"));
    let exprdecode = cx.expr_path(cx.path_global(trait_span, recurse));
    // throw an underscore in front to suppress unused variable warnings
    let blkarg = cx.ident_of("_d");
    let blkdecoder = cx.expr_ident(trait_span, blkarg);

    return match *substr.fields {
        StaticStruct(_, ref summary) => {
            let nfields = match *summary {
                Unnamed(ref fields) => fields.len(),
                Named(ref fields) => fields.len()
            };
            let read_struct_field = cx.ident_of("read_struct_field");

            let path = cx.path_ident(trait_span, substr.type_ident);
            let result = decode_static_fields(cx,
                                              trait_span,
                                              path,
                                              summary,
                                              |cx, span, name, field| {
                cx.expr_try(span,
                    cx.expr_method_call(span, blkdecoder.clone(), read_struct_field,
                                        vec!(cx.expr_str(span, name),
                                          cx.expr_usize(span, field),
                                          exprdecode.clone())))
            });
            let result = cx.expr_ok(trait_span, result);
            cx.expr_method_call(trait_span,
                                decoder,
                                cx.ident_of("read_struct"),
                                vec!(
                cx.expr_str(trait_span, substr.type_ident.name.as_str()),
                cx.expr_usize(trait_span, nfields),
                cx.lambda_expr_1(trait_span, result, blkarg)
            ))
        }
        StaticEnum(_, ref fields) => {
            let variant = cx.ident_of("i");

            let mut arms = Vec::new();
            let mut variants = Vec::new();
            let rvariant_arg = cx.ident_of("read_enum_variant_arg");

            for (i, &(ident, v_span, ref parts)) in fields.iter().enumerate() {
                variants.push(cx.expr_str(v_span, ident.name.as_str()));

                let path = cx.path(trait_span, vec![substr.type_ident, ident]);
                let decoded = decode_static_fields(cx,
                                                   v_span,
                                                   path,
                                                   parts,
                                                   |cx, span, _, field| {
                    let idx = cx.expr_usize(span, field);
                    cx.expr_try(span,
                        cx.expr_method_call(span, blkdecoder.clone(), rvariant_arg,
                                            vec!(idx, exprdecode.clone())))
                });

                arms.push(cx.arm(v_span,
                                 vec!(cx.pat_lit(v_span, cx.expr_usize(v_span, i))),
                                 decoded));
            }

            arms.push(cx.arm_unreachable(trait_span));

            let result = cx.expr_ok(trait_span,
                                    cx.expr_match(trait_span,
                                                  cx.expr_ident(trait_span, variant), arms));
            let lambda = cx.lambda_expr(trait_span, vec!(blkarg, variant), result);
            let variant_vec = cx.expr_vec(trait_span, variants);
            let variant_vec = cx.expr_addr_of(trait_span, variant_vec);
            let result = cx.expr_method_call(trait_span, blkdecoder,
                                             cx.ident_of("read_enum_variant"),
                                             vec!(variant_vec, lambda));
            cx.expr_method_call(trait_span,
                                decoder,
                                cx.ident_of("read_enum"),
                                vec!(
                cx.expr_str(trait_span, substr.type_ident.name.as_str()),
                cx.lambda_expr_1(trait_span, result, blkarg)
            ))
        }
        _ => cx.bug("expected StaticEnum or StaticStruct in derive(Decodable)")
    };
}

/// Create a decoder for a single enum variant/struct:
/// - `outer_pat_path` is the path to this enum variant/struct
/// - `getarg` should retrieve the `usize`-th field with name `@str`.
fn decode_static_fields<F>(cx: &mut ExtCtxt,
                           trait_span: Span,
                           outer_pat_path: ast::Path,
                           fields: &StaticFields,
                           mut getarg: F)
                           -> P<Expr> where
    F: FnMut(&mut ExtCtxt, Span, InternedString, usize) -> P<Expr>,
{
    match *fields {
        Unnamed(ref fields) => {
            let path_expr = cx.expr_path(outer_pat_path);
            if fields.is_empty() {
                path_expr
            } else {
                let fields = fields.iter().enumerate().map(|(i, &span)| {
                    getarg(cx, span,
                           token::intern_and_get_ident(&format!("_field{}", i)),
                           i)
                }).collect();

                cx.expr_call(trait_span, path_expr, fields)
            }
        }
        Named(ref fields) => {
            // use the field's span to get nicer error messages.
            let fields = fields.iter().enumerate().map(|(i, &(ident, span))| {
                let arg = getarg(cx, span, ident.name.as_str(), i);
                cx.field_imm(span, ident, arg)
            }).collect();
            cx.expr_struct(trait_span, outer_pat_path, fields)
        }
    }
}
