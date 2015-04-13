// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi;
use ast::{self, MetaItem, Item};
use codemap::{Span, respan};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use owned_slice::OwnedSlice;
use parse::token::{InternedString, special_idents};
use ptr::P;

use super::clone;

pub fn expand_deriving_unsafe_bound(cx: &mut ExtCtxt,
                                    span: Span,
                                    _: &MetaItem,
                                    _: &Item,
                                    _: &mut FnMut(P<Item>))
{
    cx.span_err(span, "this unsafe trait should be implemented explicitly");
}

pub fn expand_deriving_copy(cx: &mut ExtCtxt,
                            span: Span,
                            mitem: &MetaItem,
                            item: &Item,
                            push: &mut FnMut(P<Item>))
{
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::marker::Copy),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: Vec::new(),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push);

    expand_deriving_clone_when_copy(cx, span, mitem, item, push)
}

fn expand_deriving_clone_when_copy(cx: &mut ExtCtxt,
                                   span: Span,
                                   mitem: &MetaItem,
                                   item: &Item,
                                   push: &mut FnMut(P<Item>))
{
    // For generic types we need to destructure our value in order to recursively call clone.
    // However, as an optimization for non-generic types, we can just generate:
    //
    // impl<...> Clone for $ty {
    //     fn clone(&self) -> Self { *self }
    // }
    //
    // But the generic deriving helpers do not support generating such a simple method. So we'll
    // build this method by hand. However, we want to take advantage of generic deriving generating
    // the `Generics` for us. So we'll generate an empty impl, then later on add our method. It's
    // not pretty, but it works until we get a more general purpose ast builder.
    match item.node {
        ast::ItemStruct(_, ref generics) | ast::ItemEnum(_, ref generics) => {
            if generics.is_type_parameterized() {
                clone::expand_deriving_clone(cx, span, mitem, item, push);
                return;
            }
        }
        _ => {
            cx.span_err(mitem.span, "`derive` may only be applied to structs and enums");
            return;
        }
    }

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::clone::Clone),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: Vec::new(),
        associated_types: Vec::new(),
    };

    // We want to use the `cx` to build our ast, but it's passed by `&mut` to the expand method. So
    // we'll extract out the generated item by way of an option.
    let mut expanded_item = None;

    trait_def.expand(cx, mitem, item, &mut |item: P<ast::Item>| {
        expanded_item = Some(item);
    });

    let expanded_item = expanded_item.unwrap().map(|mut item| {
        match item.node {
            ast::ItemImpl(_, _, _, _, ref ty, ref mut impl_items) => {
                let self_arg = ast::Arg::new_self(span, ast::MutImmutable, special_idents::self_);
                let decl = cx.fn_decl(vec![self_arg], ty.clone());

                let sig = ast::MethodSig {
                    unsafety: ast::Unsafety::Normal,
                    abi: abi::Rust,
                    decl: decl.clone(),
                    generics: ast::Generics {
                        lifetimes: Vec::new(),
                        ty_params: OwnedSlice::empty(),
                        where_clause: ast::WhereClause {
                            id: ast::DUMMY_NODE_ID,
                            predicates: Vec::new(),
                        }
                    },
                    explicit_self: respan(
                        span,
                        ast::SelfRegion(None, ast::MutImmutable, cx.ident_of("self")),
                    ),
                };

                let block = cx.block_expr(cx.expr_deref(span, cx.expr_self(span)));

                let inline = cx.meta_word(span, InternedString::new("inline"));
                let attrs = vec!(cx.attribute(span, inline));

                impl_items.push(P(ast::ImplItem {
                    id: ast::DUMMY_NODE_ID,
                    ident: cx.ident_of("clone"),
                    vis: ast::Visibility::Inherited,
                    attrs: attrs,
                    node: ast::ImplItem_::MethodImplItem(sig, block),
                    span: span,
                }));
            }
            _ => {
                cx.span_bug(span, "we should have gotten an impl")
            }
        };

        item
    });

    push(expanded_item)
}
