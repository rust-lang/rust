// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The compiler code necessary to implement the #[deriving(Eq)] and
/// #[deriving(IterBytes)] extensions.

use core::prelude::*;

use ast;
use ast::{TraitTyParamBound, Ty, and, bind_by_ref, binop, deref, enum_def};
use ast::{expr, expr_match, ident, impure_fn, item, item_};
use ast::{item_enum, item_impl, item_struct, Generics};
use ast::{m_imm, meta_item, method};
use ast::{named_field, or, pat, pat_ident, pat_wild, public, pure_fn};
use ast::{stmt, struct_def, struct_variant_kind};
use ast::{sty_region, tuple_variant_kind, ty_nil, TyParam};
use ast::{TyParamBound, ty_path, ty_rptr, unnamed_field, variant};
use ext::base::ext_ctxt;
use ext::build;
use codemap::{span, spanned};
use parse::token::special_idents::clownshoes_extensions;
use opt_vec;

use core::uint;

pub mod clone;
pub mod eq;
pub mod iter_bytes;

type ExpandDerivingStructDefFn<'self> = &'self fn(@ext_ctxt,
                                                  span,
                                                  x: &struct_def,
                                                  ident,
                                                  y: &Generics)
                                               -> @item;
type ExpandDerivingEnumDefFn<'self> = &'self fn(@ext_ctxt,
                                                span,
                                                x: &enum_def,
                                                ident,
                                                y: &Generics)
                                             -> @item;

pub fn expand_meta_deriving(cx: @ext_ctxt,
                            _span: span,
                            mitem: @meta_item,
                            in_items: ~[@item])
                         -> ~[@item] {
    use ast::{meta_list, meta_name_value, meta_word};

    match mitem.node {
        meta_name_value(_, l) => {
            cx.span_err(l.span, ~"unexpected value in `deriving`");
            in_items
        }
        meta_word(_) | meta_list(_, []) => {
            cx.span_warn(mitem.span, ~"empty trait list in `deriving`");
            in_items
        }
        meta_list(_, titems) => {
            do titems.foldr(in_items) |&titem, in_items| {
                match titem.node {
                    meta_name_value(tname, _) |
                    meta_list(tname, _) |
                    meta_word(tname) => {
                        match *tname {
                            ~"Clone" => clone::expand_deriving_clone(cx,
                                titem.span, titem, in_items),
                            ~"Eq" => eq::expand_deriving_eq(cx, titem.span,
                                titem, in_items),
                            ~"IterBytes" => iter_bytes::expand_deriving_iter_bytes(cx,
                                titem.span, titem, in_items),
                            tname => {
                                cx.span_err(titem.span, fmt!("unknown \
                                    `deriving` trait: `%s`", tname));
                                in_items
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn expand_deriving(cx: @ext_ctxt,
                   span: span,
                   in_items: ~[@item],
                   expand_deriving_struct_def: ExpandDerivingStructDefFn,
                   expand_deriving_enum_def: ExpandDerivingEnumDefFn)
                -> ~[@item] {
    let mut result = ~[];
    for in_items.each |item| {
        result.push(copy *item);
        match item.node {
            item_struct(struct_def, ref generics) => {
                result.push(expand_deriving_struct_def(cx,
                                                       span,
                                                       struct_def,
                                                       item.ident,
                                                       generics));
            }
            item_enum(ref enum_definition, ref generics) => {
                result.push(expand_deriving_enum_def(cx,
                                                     span,
                                                     enum_definition,
                                                     item.ident,
                                                     generics));
            }
            _ => ()
        }
    }
    result
}

fn create_impl_item(cx: @ext_ctxt, span: span, +item: item_) -> @item {
    @ast::item {
        ident: clownshoes_extensions,
        attrs: ~[],
        id: cx.next_id(),
        node: item,
        vis: public,
        span: span,
    }
}

pub fn create_self_type_with_params(cx: @ext_ctxt,
                                span: span,
                                type_ident: ident,
                                generics: &Generics)
                             -> @Ty {
    // Create the type parameters on the `self` path.
    let mut self_ty_params = ~[];
    for generics.ty_params.each |ty_param| {
        let self_ty_param = build::mk_simple_ty_path(cx,
                                                     span,
                                                     ty_param.ident);
        self_ty_params.push(self_ty_param);
    }

    // Create the type of `self`.
    let self_type = build::mk_raw_path_(span,
                                        ~[ type_ident ],
                                        self_ty_params);
    let self_type = ty_path(self_type, cx.next_id());
    @ast::Ty { id: cx.next_id(), node: self_type, span: span }
}

pub fn create_derived_impl(cx: @ext_ctxt,
                       span: span,
                       type_ident: ident,
                       generics: &Generics,
                       methods: &[@method],
                       trait_path: &[ident])
                    -> @item {
    /*!
     *
     * Given that we are deriving a trait `Tr` for a type `T<'a, ...,
     * 'z, A, ..., Z>`, creates an impl like:
     *
     *      impl<'a, ..., 'z, A:Tr, ..., Z: Tr> Tr for T<A, ..., Z> { ... }
     *
     * FIXME(#5090): Remove code duplication between this and the
     * code in auto_encode.rs
     */

    // Copy the lifetimes
    let impl_lifetimes = generics.lifetimes.map(|l| {
        build::mk_lifetime(cx, l.span, l.ident)
    });

    // Create the type parameters.
    let impl_ty_params = generics.ty_params.map(|ty_param| {
        let bound = build::mk_ty_path_global(cx,
                                             span,
                                             trait_path.map(|x| *x));
        let bounds = @opt_vec::with(TraitTyParamBound(bound));
        build::mk_ty_param(cx, ty_param.ident, bounds)
    });

    // Create the reference to the trait.
    let trait_path = ast::path {
        span: span,
        global: true,
        idents: trait_path.map(|x| *x),
        rp: None,
        types: ~[]
    };
    let trait_path = @trait_path;
    let trait_ref = ast::trait_ref {
        path: trait_path,
        ref_id: cx.next_id()
    };
    let trait_ref = @trait_ref;

    // Create the type of `self`.
    let self_type = create_self_type_with_params(cx,
                                                 span,
                                                 type_ident,
                                                 generics);

    // Create the impl item.
    let impl_item = item_impl(Generics {lifetimes: impl_lifetimes,
                                        ty_params: impl_ty_params},
                              Some(trait_ref),
                              self_type,
                              methods.map(|x| *x));
    return create_impl_item(cx, span, impl_item);
}

pub fn create_subpatterns(cx: @ext_ctxt,
                      span: span,
                      prefix: ~str,
                      n: uint)
                   -> ~[@pat] {
    let mut subpats = ~[];
    for uint::range(0, n) |_i| {
        // Create the subidentifier.
        let index = subpats.len().to_str();
        let ident = cx.ident_of(prefix + index);

        // Create the subpattern.
        let subpath = build::mk_raw_path(span, ~[ ident ]);
        let subpat = pat_ident(bind_by_ref(m_imm), subpath, None);
        let subpat = build::mk_pat(cx, span, subpat);
        subpats.push(subpat);
    }
    return subpats;
}

pub fn is_struct_tuple(struct_def: &struct_def) -> bool {
    struct_def.fields.len() > 0 && struct_def.fields.all(|f| {
        match f.node.kind {
            named_field(*) => false,
            unnamed_field => true
        }
    })
}

pub fn create_enum_variant_pattern(cx: @ext_ctxt,
                               span: span,
                               variant: &variant,
                               prefix: ~str)
                            -> @pat {
    let variant_ident = variant.node.name;
    match variant.node.kind {
        tuple_variant_kind(ref variant_args) => {
            if variant_args.len() == 0 {
                return build::mk_pat_ident_with_binding_mode(
                    cx, span, variant_ident, ast::bind_infer);
            }

            let matching_path = build::mk_raw_path(span, ~[ variant_ident ]);
            let subpats = create_subpatterns(cx,
                                             span,
                                             prefix,
                                             variant_args.len());

            return build::mk_pat_enum(cx, span, matching_path, subpats);
        }
        struct_variant_kind(struct_def) => {
            let matching_path = build::mk_raw_path(span, ~[ variant_ident ]);
            let subpats = create_subpatterns(cx,
                                             span,
                                             prefix,
                                             struct_def.fields.len());

            let field_pats = do struct_def.fields.mapi |i, struct_field| {
                let ident = match struct_field.node.kind {
                    named_field(ident, _, _) => ident,
                    unnamed_field => {
                        cx.span_bug(span, ~"unexpected unnamed field");
                    }
                };
                ast::field_pat { ident: ident, pat: subpats[i] }
            };

            build::mk_pat_struct(cx, span, matching_path, field_pats)
        }
    }
}

pub fn variant_arg_count(cx: @ext_ctxt, span: span, variant: &variant) -> uint {
    match variant.node.kind {
        tuple_variant_kind(ref args) => args.len(),
        struct_variant_kind(ref struct_def) => struct_def.fields.len(),
    }
}

pub fn expand_enum_or_struct_match(cx: @ext_ctxt,
                               span: span,
                               arms: ~[ ast::arm ])
                            -> @expr {
    let self_ident = cx.ident_of(~"self");
    let self_expr = build::mk_path(cx, span, ~[ self_ident ]);
    let self_expr = build::mk_unary(cx, span, deref, self_expr);
    let self_match_expr = expr_match(self_expr, arms);
    build::mk_expr(cx, span, self_match_expr)
}
