// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi;
use ast;
use ast::{MetaItem, Item, Expr};
use codemap::{Span, respan};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::generic::*;
use ext::deriving::generic::ty::*;
use owned_slice::OwnedSlice;
use parse::token::{InternedString, special_idents};
use ptr::P;

use super::partial_ord;

pub fn expand_deriving_ord(cx: &mut ExtCtxt,
                           span: Span,
                           mitem: &MetaItem,
                           item: &Item,
                           push: &mut FnMut(P<Item>))
{
    let inline = cx.meta_word(span, InternedString::new("inline"));
    let attrs = vec!(cx.attribute(span, inline));
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::cmp::Ord),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        methods: vec!(
            MethodDef {
                name: "cmp",
                generics: LifetimeBounds::empty(),
                explicit_self: borrowed_explicit_self(),
                args: vec!(borrowed_self()),
                ret_ty: Literal(path_std!(cx, core::cmp::Ordering)),
                attributes: attrs,
                combine_substructure: combine_substructure(Box::new(|a, b, c| {
                    cs_cmp(a, b, c)
                })),
            }
        ),
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push);

    expand_deriving_partial_ord_when_ord(cx, span, mitem, item, push)
}


pub fn ordering_collapsed(cx: &mut ExtCtxt,
                          span: Span,
                          self_arg_tags: &[ast::Ident]) -> P<ast::Expr> {
    let lft = cx.expr_ident(span, self_arg_tags[0]);
    let rgt = cx.expr_addr_of(span, cx.expr_ident(span, self_arg_tags[1]));
    cx.expr_method_call(span, lft, cx.ident_of("cmp"), vec![rgt])
}

pub fn cs_cmp(cx: &mut ExtCtxt, span: Span,
              substr: &Substructure) -> P<Expr> {
    let test_id = cx.ident_of("__test");
    let equals_path = cx.path_global(span,
                                     vec!(cx.ident_of_std("core"),
                                          cx.ident_of("cmp"),
                                          cx.ident_of("Ordering"),
                                          cx.ident_of("Equal")));

    let cmp_path = vec![
        cx.ident_of_std("core"),
        cx.ident_of("cmp"),
        cx.ident_of("Ord"),
        cx.ident_of("cmp"),
    ];

    /*
    Builds:

    let __test = ::std::cmp::Ord::cmp(&self_field1, &other_field1);
    if other == ::std::cmp::Ordering::Equal {
        let __test = ::std::cmp::Ord::cmp(&self_field2, &other_field2);
        if __test == ::std::cmp::Ordering::Equal {
            ...
        } else {
            __test
        }
    } else {
        __test
    }

    FIXME #6449: These `if`s could/should be `match`es.
    */
    cs_fold(
        // foldr nests the if-elses correctly, leaving the first field
        // as the outermost one, and the last as the innermost.
        false,
        |cx, span, old, self_f, other_fs| {
            // let __test = new;
            // if __test == ::std::cmp::Ordering::Equal {
            //    old
            // } else {
            //    __test
            // }

            let new = {
                let other_f = match other_fs {
                    [ref o_f] => o_f,
                    _ => cx.span_bug(span, "not exactly 2 arguments in `derive(PartialOrd)`"),
                };

                let args = vec![
                    cx.expr_addr_of(span, self_f),
                    cx.expr_addr_of(span, other_f.clone()),
                ];

                cx.expr_call_global(span, cmp_path.clone(), args)
            };

            let assign = cx.stmt_let(span, false, test_id, new);

            let cond = cx.expr_binary(span, ast::BiEq,
                                      cx.expr_ident(span, test_id),
                                      cx.expr_path(equals_path.clone()));
            let if_ = cx.expr_if(span,
                                 cond,
                                 old, Some(cx.expr_ident(span, test_id)));
            cx.expr_block(cx.block(span, vec!(assign), Some(if_)))
        },
        cx.expr_path(equals_path.clone()),
        Box::new(|cx, span, (self_args, tag_tuple), _non_self_args| {
            if self_args.len() != 2 {
                cx.span_bug(span, "not exactly 2 arguments in `derives(Ord)`")
            } else {
                ordering_collapsed(cx, span, tag_tuple)
            }
        }),
        cx, span, substr)
}

fn expand_deriving_partial_ord_when_ord(cx: &mut ExtCtxt,
                                        span: Span,
                                        mitem: &MetaItem,
                                        item: &Item,
                                        push: &mut FnMut(P<Item>)) {
    // For generic types we need to destructure our value in order to recursively call clone.
    // However, as an optimization for non-generic types, we can just generate:
    //
    // impl<...> PartialOrd for $ty {
    //     fn partial_cmp(&self, other: &$ty) -> Option<Self> { Some(self.cmp(other)) }
    // }
    //
    // But the generic deriving helpers do not support generating such a simple method. So we'll
    // build this method by hand. However, we want to take advantage of generic deriving generating
    // the `Generics` for us. So we'll generate an empty impl, then later on add our method. It's
    // not pretty, but it works until we get a more general purpose ast builder.
    match item.node {
        ast::ItemStruct(_, ref generics) | ast::ItemEnum(_, ref generics) => {
            if generics.is_type_parameterized() {
                partial_ord::expand_deriving_partial_ord(cx, span, mitem, item, push);
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
        path: path_std!(cx, core::cmp::PartialOrd),
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
                let other_arg = cx.arg(
                    span,
                    cx.ident_of("other"),
                    cx.ty_rptr(
                        span,
                        ty.clone(),
                        None,
                        ast::Mutability::MutImmutable,
                    ),
                );
                let decl = cx.fn_decl(
                    vec![self_arg, other_arg],
                    cx.ty_option(
                        cx.ty_path(
                            cx.path_global(
                                span,
                                vec![
                                    cx.ident_of_std("core"),
                                    cx.ident_of("cmp"),
                                    cx.ident_of("Ordering"),
                                ],
                            )
                        )
                    )
                );

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

                let block = cx.block_expr(
                    cx.expr_some(
                        span,
                        cx.expr_call(
                            span,
                            cx.expr_path(
                                cx.path_global(
                                    span,
                                    vec![
                                        cx.ident_of_std("core"),
                                        cx.ident_of("cmp"),
                                        cx.ident_of("Ord"),
                                        cx.ident_of("cmp"),
                                    ],
                                ),
                            ),
                            vec![
                                cx.expr_self(span),
                                cx.expr_addr_of(
                                    span,
                                    cx.expr_ident(span, cx.ident_of("other"))
                                ),
                            ]
                        )
                    )
                );

                let inline = cx.meta_word(span, InternedString::new("inline"));
                let attrs = vec!(cx.attribute(span, inline));

                impl_items.push(P(ast::ImplItem {
                    id: ast::DUMMY_NODE_ID,
                    ident: cx.ident_of("partial_cmp"),
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
