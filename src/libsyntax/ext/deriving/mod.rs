// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
The compiler code necessary to implement the #[deriving] extensions.


FIXME (#2810)--Hygiene. Search for "__" strings (in other files too).
We also assume "extra" is the standard library, and "std" is the core
library.

*/

use core::prelude::*;

use ast;
use ast::{Ty, enum_def, expr, ident, item, Generics, meta_item, struct_def};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use codemap::span;
use parse::token::special_idents::clownshoes_extensions;
use opt_vec;

use core::uint;
use core::vec;

pub mod clone;
pub mod iter_bytes;
pub mod encodable;
pub mod decodable;
pub mod rand;
pub mod to_str;

#[path="cmp/eq.rs"]
pub mod eq;
#[path="cmp/totaleq.rs"]
pub mod totaleq;
#[path="cmp/ord.rs"]
pub mod ord;
#[path="cmp/totalord.rs"]
pub mod totalord;


pub mod generic;

pub type ExpandDerivingStructDefFn<'self> = &'self fn(@ExtCtxt,
                                                       span,
                                                       x: &struct_def,
                                                       ident,
                                                       y: &Generics)
                                                 -> @item;
pub type ExpandDerivingEnumDefFn<'self> = &'self fn(@ExtCtxt,
                                                    span,
                                                    x: &enum_def,
                                                    ident,
                                                    y: &Generics)
                                                 -> @item;

pub fn expand_meta_deriving(cx: @ExtCtxt,
                            _span: span,
                            mitem: @meta_item,
                            in_items: ~[@item])
                         -> ~[@item] {
    use ast::{meta_list, meta_name_value, meta_word};

    match mitem.node {
        meta_name_value(_, ref l) => {
            cx.span_err(l.span, "unexpected value in `deriving`");
            in_items
        }
        meta_word(_) | meta_list(_, []) => {
            cx.span_warn(mitem.span, "empty trait list in `deriving`");
            in_items
        }
        meta_list(_, ref titems) => {
            do titems.foldr(in_items) |&titem, in_items| {
                match titem.node {
                    meta_name_value(tname, _) |
                    meta_list(tname, _) |
                    meta_word(tname) => {
                        macro_rules! expand(($func:path) => ($func(cx, titem.span,
                                                                   titem, in_items)));
                        match *tname {
                            ~"Clone" => expand!(clone::expand_deriving_clone),
                            ~"DeepClone" => expand!(clone::expand_deriving_deep_clone),

                            ~"IterBytes" => expand!(iter_bytes::expand_deriving_iter_bytes),

                            ~"Encodable" => expand!(encodable::expand_deriving_encodable),
                            ~"Decodable" => expand!(decodable::expand_deriving_decodable),

                            ~"Eq" => expand!(eq::expand_deriving_eq),
                            ~"TotalEq" => expand!(totaleq::expand_deriving_totaleq),
                            ~"Ord" => expand!(ord::expand_deriving_ord),
                            ~"TotalOrd" => expand!(totalord::expand_deriving_totalord),

                            ~"Rand" => expand!(rand::expand_deriving_rand),

                            ~"ToStr" => expand!(to_str::expand_deriving_to_str),

                            ref tname => {
                                cx.span_err(titem.span, fmt!("unknown \
                                    `deriving` trait: `%s`", *tname));
                                in_items
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn expand_deriving(cx: @ExtCtxt,
                   span: span,
                   in_items: ~[@item],
                   expand_deriving_struct_def: ExpandDerivingStructDefFn,
                   expand_deriving_enum_def: ExpandDerivingEnumDefFn)
                -> ~[@item] {
    let mut result = ~[];
    for in_items.each |item| {
        result.push(copy *item);
        match item.node {
            ast::item_struct(struct_def, ref generics) => {
                result.push(expand_deriving_struct_def(cx,
                                                       span,
                                                       struct_def,
                                                       item.ident,
                                                       generics));
            }
            ast::item_enum(ref enum_definition, ref generics) => {
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

pub fn create_self_type_with_params(cx: @ExtCtxt,
                                    span: span,
                                    type_ident: ident,
                                    generics: &Generics)
    -> @Ty {
    // Create the type parameters on the `self` path.
    let mut self_ty_params = ~[];
    for generics.ty_params.each |ty_param| {
        let self_ty_param = cx.ty_ident(span, ty_param.ident);
        self_ty_params.push(self_ty_param);
    }

    let lifetime = if generics.lifetimes.is_empty() {
        None
    } else {
        Some(@*generics.lifetimes.get(0))
    };


    // Create the type of `self`.
    cx.ty_path(cx.path_all(span, false, ~[ type_ident ], lifetime, self_ty_params))
}

pub fn create_derived_impl(cx: @ExtCtxt,
                           span: span,
                           type_ident: ident,
                           generics: &Generics,
                           methods: &[@ast::method],
                           trait_path: @ast::Path,
                           mut impl_generics:  Generics,
                           bounds_paths: opt_vec::OptVec<@ast::Path>)
                        -> @item {
    /*!
     *
     * Given that we are deriving a trait `Tr` for a type `T<'a, ...,
     * 'z, A, ..., Z>`, creates an impl like:
     *
     *      impl<'a, ..., 'z, A:Tr B1 B2, ..., Z: Tr B1 B2> Tr for T<A, ..., Z> { ... }
     *
     * where B1, B2, ... are the bounds given by `bounds_paths`.
     *
     */

    // Copy the lifetimes
    for generics.lifetimes.each |l| {
        impl_generics.lifetimes.push(copy *l)
    };

    // Create the type parameters.
    for generics.ty_params.each |ty_param| {
        // extra restrictions on the generics parameters to the type being derived upon
        let mut bounds = do bounds_paths.map |&bound_path| {
            cx.typarambound(bound_path)
        };

        let this_trait_bound = cx.typarambound(trait_path);
        bounds.push(this_trait_bound);

        impl_generics.ty_params.push(cx.typaram(ty_param.ident, @bounds));
    }

    // Create the reference to the trait.
    let trait_ref = cx.trait_ref(trait_path);

    // Create the type of `self`.
    let self_type = create_self_type_with_params(cx,
                                                 span,
                                                 type_ident,
                                                 generics);

    let doc_attr = cx.attribute(
        span,
        cx.meta_name_value(span,
                           ~"doc", ast::lit_str(@~"Automatically derived.")));
    cx.item(
        span,
        clownshoes_extensions,
        ~[doc_attr],
        ast::item_impl(impl_generics,
                       Some(trait_ref),
                       self_type,
                       methods.map(|x| *x)))
}

pub fn create_subpatterns(cx: @ExtCtxt,
                          span: span,
                          field_paths: ~[@ast::Path],
                          mutbl: ast::mutability)
                   -> ~[@ast::pat] {
    do field_paths.map |&path| {
        cx.pat(span,
                      ast::pat_ident(ast::bind_by_ref(mutbl), path, None))
    }
}

#[deriving(Eq)] // dogfooding!
enum StructType {
    Unknown, Record, Tuple
}

pub fn create_struct_pattern(cx: @ExtCtxt,
                             span: span,
                             struct_ident: ident,
                             struct_def: &struct_def,
                             prefix: &str,
                             mutbl: ast::mutability)
    -> (@ast::pat, ~[(Option<ident>, @expr)]) {
    if struct_def.fields.is_empty() {
        return (
            cx.pat_ident_binding_mode(
                span, struct_ident, ast::bind_infer),
            ~[]);
    }

    let matching_path = cx.path(span, ~[ struct_ident ]);

    let mut paths = ~[], ident_expr = ~[];

    let mut struct_type = Unknown;

    for struct_def.fields.eachi |i, struct_field| {
        let opt_id = match struct_field.node.kind {
            ast::named_field(ident, _) if (struct_type == Unknown ||
                                           struct_type == Record) => {
                struct_type = Record;
                Some(ident)
            }
            ast::unnamed_field if (struct_type == Unknown ||
                                   struct_type == Tuple) => {
                struct_type = Tuple;
                None
            }
            _ => {
                cx.span_bug(span, "A struct with named and unnamed fields in `deriving`");
            }
        };
        let path = cx.path_ident(span,
                                 cx.ident_of(fmt!("%s_%u", prefix, i)));
        paths.push(path);
        ident_expr.push((opt_id, cx.expr_path(path)));
    }

    let subpats = create_subpatterns(cx, span, paths, mutbl);

    // struct_type is definitely not Unknown, since struct_def.fields
    // must be nonempty to reach here
    let pattern = if struct_type == Record {
        let field_pats = do vec::build |push| {
            for vec::each2(subpats, ident_expr) |&pat, &(id, _)| {
                // id is guaranteed to be Some
                push(ast::field_pat { ident: id.get(), pat: pat })
            }
        };
        cx.pat_struct(span, matching_path, field_pats)
    } else {
        cx.pat_enum(span, matching_path, subpats)
    };

    (pattern, ident_expr)
}

pub fn create_enum_variant_pattern(cx: @ExtCtxt,
                                   span: span,
                                   variant: &ast::variant,
                                   prefix: &str,
                                   mutbl: ast::mutability)
    -> (@ast::pat, ~[(Option<ident>, @expr)]) {

    let variant_ident = variant.node.name;
    match variant.node.kind {
        ast::tuple_variant_kind(ref variant_args) => {
            if variant_args.is_empty() {
                return (cx.pat_ident_binding_mode(
                    span, variant_ident, ast::bind_infer), ~[]);
            }

            let matching_path = cx.path_ident(span, variant_ident);

            let mut paths = ~[], ident_expr = ~[];
            for uint::range(0, variant_args.len()) |i| {
                let path = cx.path_ident(span,
                                         cx.ident_of(fmt!("%s_%u", prefix, i)));

                paths.push(path);
                ident_expr.push((None, cx.expr_path(path)));
            }

            let subpats = create_subpatterns(cx, span, paths, mutbl);

            (cx.pat_enum(span, matching_path, subpats),
             ident_expr)
        }
        ast::struct_variant_kind(struct_def) => {
            create_struct_pattern(cx, span,
                                  variant_ident, struct_def,
                                  prefix,
                                  mutbl)
        }
    }
}

pub fn variant_arg_count(_cx: @ExtCtxt, _span: span, variant: &ast::variant) -> uint {
    match variant.node.kind {
        ast::tuple_variant_kind(ref args) => args.len(),
        ast::struct_variant_kind(ref struct_def) => struct_def.fields.len(),
    }
}

pub fn expand_enum_or_struct_match(cx: @ExtCtxt,
                               span: span,
                               arms: ~[ ast::arm ])
                            -> @expr {
    let self_expr = cx.expr_deref(span, cx.expr_self(span));
    cx.expr_match(span, self_expr, arms)
}
