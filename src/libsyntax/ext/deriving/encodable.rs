// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The compiler code necessary to implement the `#[derive(Encodable)]`
//! (and `Decodable`, in decodable.rs) extension.  The idea here is that
//! type-defining items may be tagged with `#[derive(Encodable, Decodable)]`.
//!
//! For example, a type like:
//!
//! ```ignore
//! #[derive(Encodable, Decodable)]
//! struct Node { id: uint }
//! ```
//!
//! would generate two implementations like:
//!
//! ```ignore
//! impl<S: Encoder<E>, E> Encodable<S, E> for Node {
//!     fn encode(&self, s: &mut S) -> Result<(), E> {
//!         s.emit_struct("Node", 1, |this| {
//!             this.emit_struct_field("id", 0, |this| {
//!                 Encodable::encode(&self.id, this)
//!                 /* this.emit_uint(self.id) can also be used */
//!             })
//!         })
//!     }
//! }
//!
//! impl<D: Decoder<E>, E> Decodable<D, E> for Node {
//!     fn decode(d: &mut D) -> Result<Node, E> {
//!         d.read_struct("Node", 1, |this| {
//!             match this.read_struct_field("id", 0, |this| Decodable::decode(this)) {
//!                 Ok(id) => Ok(Node { id: id }),
//!                 Err(e) => Err(e),
//!             }
//!         })
//!     }
//! }
//! ```
//!
//! Other interesting scenarios are when the item has type parameters or
//! references other non-built-in types.  A type definition like:
//!
//! ```ignore
//! #[derive(Encodable, Decodable)]
//! struct Spanned<T> { node: T, span: Span }
//! ```
//!
//! would yield functions like:
//!
//! ```ignore
//! impl<
//!     S: Encoder<E>,
//!     E,
//!     T: Encodable<S, E>
//! > Encodable<S, E> for Spanned<T> {
//!     fn encode(&self, s: &mut S) -> Result<(), E> {
//!         s.emit_struct("Spanned", 2, |this| {
//!             this.emit_struct_field("node", 0, |this| self.node.encode(this))
//!                 .ok().unwrap();
//!             this.emit_struct_field("span", 1, |this| self.span.encode(this))
//!         })
//!     }
//! }
//!
//! impl<
//!     D: Decoder<E>,
//!     E,
//!     T: Decodable<D, E>
//! > Decodable<D, E> for Spanned<T> {
//!     fn decode(d: &mut D) -> Result<Spanned<T>, E> {
//!         d.read_struct("Spanned", 2, |this| {
//!             Ok(Spanned {
//!                 node: this.read_struct_field("node", 0, |this| Decodable::decode(this))
//!                     .ok().unwrap(),
//!                 span: this.read_struct_field("span", 1, |this| Decodable::decode(this))
//!                     .ok().unwrap(),
//!             })
//!         })
//!     }
//! }
//! ```

use ast::{MetaItem, Item, Expr, ExprRet, MutMutable};
use codemap::Span;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use parse::token;
use ptr::P;

pub fn expand_deriving_rustc_encodable<F>(cx: &mut ExtCtxt,
                                          span: Span,
                                          mitem: &MetaItem,
                                          item: &Item,
                                          push: F) where
    F: FnOnce(P<Item>),
{
    expand_deriving_encodable_imp(cx, span, mitem, item, push, "rustc_serialize")
}

pub fn expand_deriving_encodable<F>(cx: &mut ExtCtxt,
                                    span: Span,
                                    mitem: &MetaItem,
                                    item: &Item,
                                    push: F) where
    F: FnOnce(P<Item>),
{
    expand_deriving_encodable_imp(cx, span, mitem, item, push, "serialize")
}

fn expand_deriving_encodable_imp<F>(cx: &mut ExtCtxt,
                                    span: Span,
                                    mitem: &MetaItem,
                                    item: &Item,
                                    push: F,
                                    krate: &'static str) where
    F: FnOnce(P<Item>),
{
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: Path::new_(vec!(krate, "Encodable"), None, vec!(), true),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "encode",
                generics: LifetimeBounds {
                    lifetimes: Vec::new(),
                    bounds: vec!(("__S", vec!(Path::new_(
                                    vec!(krate, "Encoder"), None,
                                    vec!(), true))))
                },
                explicit_self: borrowed_explicit_self(),
                args: vec!(Ptr(box Literal(Path::new_local("__S")),
                            Borrowed(None, MutMutable))),
                ret_ty: Literal(Path::new_(
                    vec!("std", "result", "Result"),
                    None,
                    vec!(box Tuple(Vec::new()), box Literal(Path::new_(
                        vec!["__S", "Error"], None, vec![], false
                    ))),
                    true
                )),
                attributes: Vec::new(),
                combine_substructure: combine_substructure(box |a, b, c| {
                    encodable_substructure(a, b, c)
                }),
            })
    };

    trait_def.expand(cx, mitem, item, push)
}

fn encodable_substructure(cx: &mut ExtCtxt, trait_span: Span,
                          substr: &Substructure) -> P<Expr> {
    let encoder = substr.nonself_args[0].clone();
    // throw an underscore in front to suppress unused variable warnings
    let blkarg = cx.ident_of("_e");
    let blkencoder = cx.expr_ident(trait_span, blkarg);
    let encode = cx.ident_of("encode");

    return match *substr.fields {
        Struct(ref fields) => {
            let emit_struct_field = cx.ident_of("emit_struct_field");
            let mut stmts = Vec::new();
            let last = fields.len() - 1;
            for (i, &FieldInfo {
                    name,
                    ref self_,
                    span,
                    ..
                }) in fields.iter().enumerate() {
                let name = match name {
                    Some(id) => token::get_ident(id),
                    None => {
                        token::intern_and_get_ident(&format!("_field{}", i)[])
                    }
                };
                let enc = cx.expr_method_call(span, self_.clone(),
                                              encode, vec!(blkencoder.clone()));
                let lambda = cx.lambda_expr_1(span, enc, blkarg);
                let call = cx.expr_method_call(span, blkencoder.clone(),
                                               emit_struct_field,
                                               vec!(cx.expr_str(span, name),
                                                 cx.expr_uint(span, i),
                                                 lambda));

                // last call doesn't need a try!
                let call = if i != last {
                    cx.expr_try(span, call)
                } else {
                    cx.expr(span, ExprRet(Some(call)))
                };
                stmts.push(cx.stmt_expr(call));
            }

            // unit structs have no fields and need to return Ok()
            if stmts.is_empty() {
                let ret_ok = cx.expr(trait_span,
                                     ExprRet(Some(cx.expr_ok(trait_span,
                                                             cx.expr_tuple(trait_span, vec![])))));
                stmts.push(cx.stmt_expr(ret_ok));
            }

            let blk = cx.lambda_stmts_1(trait_span, stmts, blkarg);
            cx.expr_method_call(trait_span,
                                encoder,
                                cx.ident_of("emit_struct"),
                                vec!(
                cx.expr_str(trait_span, token::get_ident(substr.type_ident)),
                cx.expr_uint(trait_span, fields.len()),
                blk
            ))
        }

        EnumMatching(idx, variant, ref fields) => {
            // We're not generating an AST that the borrow checker is expecting,
            // so we need to generate a unique local variable to take the
            // mutable loan out on, otherwise we get conflicts which don't
            // actually exist.
            let me = cx.stmt_let(trait_span, false, blkarg, encoder);
            let encoder = cx.expr_ident(trait_span, blkarg);
            let emit_variant_arg = cx.ident_of("emit_enum_variant_arg");
            let mut stmts = Vec::new();
            let last = fields.len() - 1;
            for (i, &FieldInfo { ref self_, span, .. }) in fields.iter().enumerate() {
                let enc = cx.expr_method_call(span, self_.clone(),
                                              encode, vec!(blkencoder.clone()));
                let lambda = cx.lambda_expr_1(span, enc, blkarg);
                let call = cx.expr_method_call(span, blkencoder.clone(),
                                               emit_variant_arg,
                                               vec!(cx.expr_uint(span, i),
                                                 lambda));
                let call = if i != last {
                    cx.expr_try(span, call)
                } else {
                    cx.expr(span, ExprRet(Some(call)))
                };
                stmts.push(cx.stmt_expr(call));
            }

            // enums with no fields need to return Ok()
            if stmts.len() == 0 {
                let ret_ok = cx.expr(trait_span,
                                     ExprRet(Some(cx.expr_ok(trait_span,
                                                             cx.expr_tuple(trait_span, vec![])))));
                stmts.push(cx.stmt_expr(ret_ok));
            }

            let blk = cx.lambda_stmts_1(trait_span, stmts, blkarg);
            let name = cx.expr_str(trait_span, token::get_ident(variant.node.name));
            let call = cx.expr_method_call(trait_span, blkencoder,
                                           cx.ident_of("emit_enum_variant"),
                                           vec!(name,
                                             cx.expr_uint(trait_span, idx),
                                             cx.expr_uint(trait_span, fields.len()),
                                             blk));
            let blk = cx.lambda_expr_1(trait_span, call, blkarg);
            let ret = cx.expr_method_call(trait_span,
                                          encoder,
                                          cx.ident_of("emit_enum"),
                                          vec!(
                cx.expr_str(trait_span, token::get_ident(substr.type_ident)),
                blk
            ));
            cx.expr_block(cx.block(trait_span, vec!(me), Some(ret)))
        }

        _ => cx.bug("expected Struct or EnumMatching in deriving(Encodable)")
    };
}
