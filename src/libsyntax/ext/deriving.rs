// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The compiler code necessary to implement the #[deriving_eq] and
/// #[deriving_iter_bytes] extensions.

use core::prelude::*;

use ast;
use ast::{TraitTyParamBound, Ty, and, bind_by_ref, binop, deref, enum_def};
use ast::{enum_variant_kind, expr, expr_match, ident, impure_fn, item, item_};
use ast::{item_enum, item_impl, item_struct, Generics};
use ast::{m_imm, meta_item, method};
use ast::{named_field, or, pat, pat_ident, pat_wild, public, pure_fn};
use ast::{stmt, struct_def, struct_variant_kind};
use ast::{sty_by_ref, sty_region, tuple_variant_kind, ty_nil, TyParam};
use ast::{TyParamBound, ty_path, ty_rptr, unnamed_field, variant};
use ext::base::ext_ctxt;
use ext::build;
use codemap::{span, spanned};
use parse::token::special_idents::clownshoes_extensions;
use ast_util;
use opt_vec;

use core::uint;

enum Junction {
    Conjunction,
    Disjunction,
}

pub impl Junction {
    fn to_binop(self) -> binop {
        match self {
            Conjunction => and,
            Disjunction => or,
        }
    }
}

type ExpandDerivingStructDefFn = &self/fn(ext_ctxt,
                                          span,
                                          x: &struct_def,
                                          ident,
                                          y: &Generics) -> @item;
type ExpandDerivingEnumDefFn = &self/fn(ext_ctxt,
                                        span,
                                        x: &enum_def,
                                        ident,
                                        y: &Generics) -> @item;

pub fn expand_deriving_eq(cx: ext_ctxt,
                          span: span,
                          _mitem: @meta_item,
                          in_items: ~[@item])
                       -> ~[@item] {
    expand_deriving(cx,
                    span,
                    in_items,
                    expand_deriving_eq_struct_def,
                    expand_deriving_eq_enum_def)
}

pub fn expand_deriving_iter_bytes(cx: ext_ctxt,
                                  span: span,
                                  _mitem: @meta_item,
                                  in_items: ~[@item])
                               -> ~[@item] {
    expand_deriving(cx,
                    span,
                    in_items,
                    expand_deriving_iter_bytes_struct_def,
                    expand_deriving_iter_bytes_enum_def)
}

pub fn expand_deriving_clone(cx: ext_ctxt,
                             span: span,
                             _: @meta_item,
                             in_items: ~[@item])
                          -> ~[@item] {
    expand_deriving(cx,
                    span,
                    in_items,
                    expand_deriving_clone_struct_def,
                    expand_deriving_clone_enum_def)
}

fn expand_deriving(cx: ext_ctxt,
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

fn create_impl_item(cx: ext_ctxt, span: span, +item: item_) -> @item {
    @ast::item {
        ident: clownshoes_extensions,
        attrs: ~[],
        id: cx.next_id(),
        node: item,
        vis: public,
        span: span,
    }
}

/// Creates a method from the given expression, the signature of which
/// conforms to the `eq` or `ne` method.
fn create_eq_method(cx: ext_ctxt,
                    span: span,
                    method_ident: ident,
                    type_ident: ident,
                    generics: &Generics,
                    body: @expr)
                 -> @method {
    // Create the type of the `other` parameter.
    let arg_path_type = create_self_type_with_params(cx,
                                                     span,
                                                     type_ident,
                                                     generics);
    let arg_type = ty_rptr(
        None,
        ast::mt { ty: arg_path_type, mutbl: m_imm }
    );
    let arg_type = @ast::Ty {
        id: cx.next_id(),
        node: arg_type,
        span: span,
    };

    // Create the `other` parameter.
    let other_ident = cx.ident_of(~"__other");
    let arg = build::mk_arg(cx, span, other_ident, arg_type);

    // Create the type of the return value.
    let bool_ident = cx.ident_of(~"bool");
    let output_type = build::mk_raw_path(span, ~[ bool_ident ]);
    let output_type = ty_path(output_type, cx.next_id());
    let output_type = @ast::Ty {
        id: cx.next_id(),
        node: output_type,
        span: span,
    };

    // Create the function declaration.
    let fn_decl = build::mk_fn_decl(~[ arg ], output_type);

    // Create the body block.
    let body_block = build::mk_simple_block(cx, span, body);

    // Create the method.
    let self_ty = spanned { node: sty_region(m_imm), span: span };
    @ast::method {
        ident: method_ident,
        attrs: ~[],
        generics: ast_util::empty_generics(),
        self_ty: self_ty,
        purity: pure_fn,
        decl: fn_decl,
        body: body_block,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: public
    }
}

fn create_self_type_with_params(cx: ext_ctxt,
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

fn create_derived_impl(cx: ext_ctxt,
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

fn create_derived_eq_impl(cx: ext_ctxt,
                          span: span,
                          type_ident: ident,
                          generics: &Generics,
                          eq_method: @method,
                          ne_method: @method)
                       -> @item {
    let methods = [ eq_method, ne_method ];
    let trait_path = [
        cx.ident_of(~"core"),
        cx.ident_of(~"cmp"),
        cx.ident_of(~"Eq")
    ];
    create_derived_impl(cx, span, type_ident, generics, methods, trait_path)
}

fn create_derived_iter_bytes_impl(cx: ext_ctxt,
                                  span: span,
                                  type_ident: ident,
                                  generics: &Generics,
                                  method: @method)
                               -> @item {
    let methods = [ method ];
    let trait_path = [
        cx.ident_of(~"core"),
        cx.ident_of(~"to_bytes"),
        cx.ident_of(~"IterBytes")
    ];
    create_derived_impl(cx, span, type_ident, generics, methods, trait_path)
}

fn create_derived_clone_impl(cx: ext_ctxt,
                             span: span,
                             type_ident: ident,
                             generics: &Generics,
                             method: @method)
                          -> @item {
    let methods = [ method ];
    let trait_path = [
        cx.ident_of(~"core"),
        cx.ident_of(~"clone"),
        cx.ident_of(~"Clone"),
    ];
    create_derived_impl(cx, span, type_ident, generics, methods, trait_path)
}

// Creates a method from the given set of statements conforming to the
// signature of the `iter_bytes` method.
fn create_iter_bytes_method(cx: ext_ctxt,
                            span: span,
                            +statements: ~[@stmt])
                         -> @method {
    // Create the `lsb0` parameter.
    let bool_ident = cx.ident_of(~"bool");
    let lsb0_arg_type = build::mk_simple_ty_path(cx, span, bool_ident);
    let lsb0_ident = cx.ident_of(~"__lsb0");
    let lsb0_arg = build::mk_arg(cx, span, lsb0_ident, lsb0_arg_type);

    // Create the `f` parameter.
    let core_ident = cx.ident_of(~"core");
    let to_bytes_ident = cx.ident_of(~"to_bytes");
    let cb_ident = cx.ident_of(~"Cb");
    let core_to_bytes_cb_ident = ~[ core_ident, to_bytes_ident, cb_ident ];
    let f_arg_type = build::mk_ty_path(cx, span, core_to_bytes_cb_ident);
    let f_ident = cx.ident_of(~"__f");
    let f_arg = build::mk_arg(cx, span, f_ident, f_arg_type);

    // Create the type of the return value.
    let output_type = @ast::Ty { id: cx.next_id(), node: ty_nil, span: span };

    // Create the function declaration.
    let inputs = ~[ lsb0_arg, f_arg ];
    let fn_decl = build::mk_fn_decl(inputs, output_type);

    // Create the body block.
    let body_block = build::mk_block_(cx, span, statements);

    // Create the method.
    let self_ty = spanned { node: sty_region(m_imm), span: span };
    let method_ident = cx.ident_of(~"iter_bytes");
    @ast::method {
        ident: method_ident,
        attrs: ~[],
        generics: ast_util::empty_generics(),
        self_ty: self_ty,
        purity: pure_fn,
        decl: fn_decl,
        body: body_block,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: public
    }
}

// Creates a method from the given expression conforming to the signature of
// the `clone` method.
fn create_clone_method(cx: ext_ctxt,
                       span: span,
                       +type_ident: ast::ident,
                       generics: &Generics,
                       expr: @ast::expr)
                    -> @method {
    // Create the type parameters of the return value.
    let mut output_ty_params = ~[];
    for generics.ty_params.each |ty_param| {
        let path = build::mk_ty_path(cx, span, ~[ ty_param.ident ]);
        output_ty_params.push(path);
    }

    // Create the type of the return value.
    let output_type_path = build::mk_raw_path_(span,
                                               ~[ type_ident ],
                                               output_ty_params);
    let output_type = ast::ty_path(output_type_path, cx.next_id());
    let output_type = @ast::Ty {
        id: cx.next_id(),
        node: output_type,
        span: span
    };

    // Create the function declaration.
    let fn_decl = build::mk_fn_decl(~[], output_type);

    // Create the body block.
    let body_block = build::mk_simple_block(cx, span, expr);

    // Create the self type and method identifier.
    let self_ty = spanned { node: sty_region(m_imm), span: span };
    let method_ident = cx.ident_of(~"clone");

    // Create the method.
    @ast::method {
        ident: method_ident,
        attrs: ~[],
        generics: ast_util::empty_generics(),
        self_ty: self_ty,
        purity: impure_fn,
        decl: fn_decl,
        body: body_block,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: public,
    }
}

fn create_subpatterns(cx: ext_ctxt,
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

fn is_struct_tuple(struct_def: &struct_def) -> bool {
    struct_def.fields.len() > 0 && struct_def.fields.all(|f| {
        match f.node.kind {
            named_field(*) => false,
            unnamed_field => true
        }
    })
}

fn create_enum_variant_pattern(cx: ext_ctxt,
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
        enum_variant_kind(*) => {
            cx.span_unimpl(span, ~"enum variants for `deriving`");
        }
    }
}

fn call_substructure_eq_method(cx: ext_ctxt,
                               span: span,
                               self_field: @expr,
                               other_field_ref: @expr,
                               method_ident: ident,
                               junction: Junction,
                               chain_expr: &mut Option<@expr>) {
    // Call the substructure method.
    let self_method = build::mk_access_(cx, span, self_field, method_ident);
    let self_call = build::mk_call_(cx,
                                    span,
                                    self_method,
                                    ~[ other_field_ref ]);

    // Connect to the outer expression if necessary.
    *chain_expr = match *chain_expr {
        None => Some(self_call),
        Some(copy old_outer_expr) => {
            let binop = junction.to_binop();
            let chain_expr = build::mk_binary(cx,
                                              span,
                                              binop,
                                              old_outer_expr,
                                              self_call);
            Some(chain_expr)
        }
    };
}

fn finish_eq_chain_expr(cx: ext_ctxt,
                        span: span,
                        chain_expr: Option<@expr>,
                        junction: Junction)
                     -> @expr {
    match chain_expr {
        None => {
            match junction {
                Conjunction => build::mk_bool(cx, span, true),
                Disjunction => build::mk_bool(cx, span, false),
            }
        }
        Some(ref outer_expr) => *outer_expr,
    }
}

fn call_substructure_iter_bytes_method(cx: ext_ctxt,
                                       span: span,
                                       self_field: @expr)
                                    -> @stmt {
    // Gather up the parameters we want to chain along.
    let lsb0_ident = cx.ident_of(~"__lsb0");
    let f_ident = cx.ident_of(~"__f");
    let lsb0_expr = build::mk_path(cx, span, ~[ lsb0_ident ]);
    let f_expr = build::mk_path(cx, span, ~[ f_ident ]);

    // Call the substructure method.
    let iter_bytes_ident = cx.ident_of(~"iter_bytes");
    let self_method = build::mk_access_(cx,
                                        span,
                                        self_field,
                                        iter_bytes_ident);
    let self_call = build::mk_call_(cx,
                                    span,
                                    self_method,
                                    ~[ lsb0_expr, f_expr ]);

    // Create a statement out of this expression.
    build::mk_stmt(cx, span, self_call)
}

fn call_substructure_clone_method(cx: ext_ctxt,
                                  span: span,
                                  self_field: @expr)
                               -> @expr {
    // Call the substructure method.
    let clone_ident = cx.ident_of(~"clone");
    let self_method = build::mk_access_(cx, span, self_field, clone_ident);
    build::mk_call_(cx, span, self_method, ~[])
}

fn variant_arg_count(cx: ext_ctxt, span: span, variant: &variant) -> uint {
    match variant.node.kind {
        tuple_variant_kind(ref args) => args.len(),
        struct_variant_kind(ref struct_def) => struct_def.fields.len(),
        enum_variant_kind(*) => {
            cx.span_bug(span, ~"variant_arg_count: enum variants deprecated")
        }
    }
}

fn expand_deriving_eq_struct_def(cx: ext_ctxt,
                                 span: span,
                                 struct_def: &struct_def,
                                 type_ident: ident,
                                 generics: &Generics)
                              -> @item {
    // Create the methods.
    let eq_ident = cx.ident_of(~"eq");
    let ne_ident = cx.ident_of(~"ne");

    let derive_struct_fn = if is_struct_tuple(struct_def) {
        expand_deriving_eq_struct_tuple_method
    } else {
        expand_deriving_eq_struct_method
    };

    let eq_method = derive_struct_fn(cx,
                                     span,
                                     struct_def,
                                     eq_ident,
                                     type_ident,
                                     generics,
                                     Conjunction);
    let ne_method = derive_struct_fn(cx,
                                     span,
                                     struct_def,
                                     ne_ident,
                                     type_ident,
                                     generics,
                                     Disjunction);

    // Create the implementation.
    return create_derived_eq_impl(cx,
                                  span,
                                  type_ident,
                                  generics,
                                  eq_method,
                                  ne_method);
}

fn expand_deriving_eq_enum_def(cx: ext_ctxt,
                               span: span,
                               enum_definition: &enum_def,
                               type_ident: ident,
                               generics: &Generics)
                            -> @item {
    // Create the methods.
    let eq_ident = cx.ident_of(~"eq");
    let ne_ident = cx.ident_of(~"ne");
    let eq_method = expand_deriving_eq_enum_method(cx,
                                                   span,
                                                   enum_definition,
                                                   eq_ident,
                                                   type_ident,
                                                   generics,
                                                   Conjunction);
    let ne_method = expand_deriving_eq_enum_method(cx,
                                                   span,
                                                   enum_definition,
                                                   ne_ident,
                                                   type_ident,
                                                   generics,
                                                   Disjunction);

    // Create the implementation.
    return create_derived_eq_impl(cx,
                                  span,
                                  type_ident,
                                  generics,
                                  eq_method,
                                  ne_method);
}

fn expand_deriving_iter_bytes_struct_def(cx: ext_ctxt,
                                         span: span,
                                         struct_def: &struct_def,
                                         type_ident: ident,
                                         generics: &Generics)
                                      -> @item {
    // Create the method.
    let method = expand_deriving_iter_bytes_struct_method(cx,
                                                          span,
                                                          struct_def);

    // Create the implementation.
    return create_derived_iter_bytes_impl(cx,
                                          span,
                                          type_ident,
                                          generics,
                                          method);
}

fn expand_deriving_iter_bytes_enum_def(cx: ext_ctxt,
                                       span: span,
                                       enum_definition: &enum_def,
                                       type_ident: ident,
                                       generics: &Generics)
                                    -> @item {
    // Create the method.
    let method = expand_deriving_iter_bytes_enum_method(cx,
                                                        span,
                                                        enum_definition);

    // Create the implementation.
    return create_derived_iter_bytes_impl(cx,
                                          span,
                                          type_ident,
                                          generics,
                                          method);
}

fn expand_deriving_clone_struct_def(cx: ext_ctxt,
                                    span: span,
                                    struct_def: &struct_def,
                                    type_ident: ident,
                                    generics: &Generics)
                                 -> @item {
    // Create the method.
    let method = if !is_struct_tuple(struct_def) {
        expand_deriving_clone_struct_method(cx,
                                            span,
                                            struct_def,
                                            type_ident,
                                            generics)
    } else {
        expand_deriving_clone_tuple_struct_method(cx,
                                                  span,
                                                  struct_def,
                                                  type_ident,
                                                  generics)
    };

    // Create the implementation.
    create_derived_clone_impl(cx, span, type_ident, generics, method)
}

fn expand_deriving_clone_enum_def(cx: ext_ctxt,
                                  span: span,
                                  enum_definition: &enum_def,
                                  type_ident: ident,
                                  generics: &Generics)
                               -> @item {
    // Create the method.
    let method = expand_deriving_clone_enum_method(cx,
                                                   span,
                                                   enum_definition,
                                                   type_ident,
                                                   generics);

    // Create the implementation.
    create_derived_clone_impl(cx, span, type_ident, generics, method)
}

fn expand_deriving_eq_struct_method(cx: ext_ctxt,
                                    span: span,
                                    struct_def: &struct_def,
                                    method_ident: ident,
                                    type_ident: ident,
                                    generics: &Generics,
                                    junction: Junction)
                                 -> @method {
    let self_ident = cx.ident_of(~"self");
    let other_ident = cx.ident_of(~"__other");

    // Create the body of the method.
    let mut outer_expr = None;
    for struct_def.fields.each |struct_field| {
        match struct_field.node.kind {
            named_field(ident, _, _) => {
                // Create the accessor for the other field.
                let other_field = build::mk_access(cx,
                                                   span,
                                                   ~[ other_ident ],
                                                   ident);
                let other_field_ref = build::mk_addr_of(cx,
                                                        span,
                                                        other_field);

                // Create the accessor for this field.
                let self_field = build::mk_access(cx,
                                                  span,
                                                  ~[ self_ident ],
                                                  ident);

                // Call the substructure method.
                call_substructure_eq_method(cx,
                                            span,
                                            self_field,
                                            other_field_ref,
                                            method_ident,
                                            junction,
                                            &mut outer_expr);
            }
            unnamed_field => {
                cx.span_unimpl(span, ~"unnamed fields with `deriving_eq`");
            }
        }
    }

    // Create the method itself.
    let body = finish_eq_chain_expr(cx, span, outer_expr, junction);
    return create_eq_method(cx,
                            span,
                            method_ident,
                            type_ident,
                            generics,
                            body);
}

fn expand_deriving_iter_bytes_struct_method(cx: ext_ctxt,
                                            span: span,
                                            struct_def: &struct_def)
                                         -> @method {
    let self_ident = cx.ident_of(~"self");

    // Create the body of the method.
    let mut statements = ~[];
    for struct_def.fields.each |struct_field| {
        match struct_field.node.kind {
            named_field(ident, _, _) => {
                // Create the accessor for this field.
                let self_field = build::mk_access(cx,
                                                  span,
                                                  ~[ self_ident ],
                                                  ident);

                // Call the substructure method.
                let stmt = call_substructure_iter_bytes_method(cx,
                                                               span,
                                                               self_field);
                statements.push(stmt);
            }
            unnamed_field => {
                cx.span_unimpl(span,
                               ~"unnamed fields with `deriving_iter_bytes`");
            }
        }
    }

    // Create the method itself.
    return create_iter_bytes_method(cx, span, statements);
}

fn expand_deriving_clone_struct_method(cx: ext_ctxt,
                                       span: span,
                                       struct_def: &struct_def,
                                       type_ident: ident,
                                       generics: &Generics)
                                    -> @method {
    let self_ident = cx.ident_of(~"self");

    // Create the new fields.
    let mut fields = ~[];
    for struct_def.fields.each |struct_field| {
        match struct_field.node.kind {
            named_field(ident, _, _) => {
                // Create the accessor for this field.
                let self_field = build::mk_access(cx,
                                                  span,
                                                  ~[ self_ident ],
                                                  ident);

                // Call the substructure method.
                let call = call_substructure_clone_method(cx,
                                                          span,
                                                          self_field);

                let field = build::Field { ident: ident, ex: call };
                fields.push(field);
            }
            unnamed_field => {
                cx.span_bug(span,
                            ~"unnamed fields in \
                              expand_deriving_clone_struct_method");
            }
        }
    }

    // Create the struct literal.
    let struct_literal = build::mk_struct_e(cx,
                                            span,
                                            ~[ type_ident ],
                                            fields);
    create_clone_method(cx, span, type_ident, generics, struct_literal)
}

fn expand_deriving_clone_tuple_struct_method(cx: ext_ctxt,
                                             span: span,
                                             struct_def: &struct_def,
                                             type_ident: ident,
                                             generics: &Generics)
                                          -> @method {
    // Create the pattern for the match.
    let matching_path = build::mk_raw_path(span, ~[ type_ident ]);
    let field_count = struct_def.fields.len();
    let subpats = create_subpatterns(cx, span, ~"__self", field_count);
    let pat = build::mk_pat_enum(cx, span, matching_path, subpats);

    // Create the new fields.
    let mut subcalls = ~[];
    for uint::range(0, struct_def.fields.len()) |i| {
        // Create the expression for this field.
        let field_ident = cx.ident_of(~"__self" + i.to_str());
        let field = build::mk_path(cx, span, ~[ field_ident ]);

        // Call the substructure method.
        let subcall = call_substructure_clone_method(cx, span, field);
        subcalls.push(subcall);
    }

    // Create the call to the struct constructor.
    let call = build::mk_call(cx, span, ~[ type_ident ], subcalls);

    // Create the pattern body.
    let match_body_block = build::mk_simple_block(cx, span, call);

    // Create the arm.
    let arm = ast::arm {
        pats: ~[ pat ],
        guard: None,
        body: match_body_block
    };

    // Create the method body.
    let self_match_expr = expand_enum_or_struct_match(cx, span, ~[ arm ]);

    // Create the method.
    create_clone_method(cx, span, type_ident, generics, self_match_expr)
}

fn expand_deriving_eq_enum_method(cx: ext_ctxt,
                                  span: span,
                                  enum_definition: &enum_def,
                                  method_ident: ident,
                                  type_ident: ident,
                                  generics: &Generics,
                                  junction: Junction)
                               -> @method {
    let self_ident = cx.ident_of(~"self");
    let other_ident = cx.ident_of(~"__other");

    let is_eq;
    match junction {
        Conjunction => is_eq = true,
        Disjunction => is_eq = false,
    }

    // Create the arms of the self match in the method body.
    let mut self_arms = ~[];
    for enum_definition.variants.each |self_variant| {
        let mut other_arms = ~[];

        // Create the matching pattern.
        let matching_pat = create_enum_variant_pattern(cx,
                                                       span,
                                                       self_variant,
                                                       ~"__other");

        // Create the matching pattern body.
        let mut matching_body_expr = None;
        for uint::range(0, variant_arg_count(cx, span, self_variant)) |i| {
            // Create the expression for the other field.
            let other_field_ident = cx.ident_of(~"__other" + i.to_str());
            let other_field = build::mk_path(cx,
                                             span,
                                             ~[ other_field_ident ]);

            // Create the expression for this field.
            let self_field_ident = cx.ident_of(~"__self" + i.to_str());
            let self_field = build::mk_path(cx, span, ~[ self_field_ident ]);

            // Call the substructure method.
            call_substructure_eq_method(cx,
                                        span,
                                        self_field,
                                        other_field,
                                        method_ident,
                                        junction,
                                        &mut matching_body_expr);
        }

        let matching_body_expr = finish_eq_chain_expr(cx,
                                                      span,
                                                      matching_body_expr,
                                                      junction);
        let matching_body_block = build::mk_simple_block(cx,
                                                         span,
                                                         matching_body_expr);

        // Create the matching arm.
        let matching_arm = ast::arm {
            pats: ~[ matching_pat ],
            guard: None,
            body: matching_body_block
        };
        other_arms.push(matching_arm);

        // Maybe generate a non-matching case. If there is only one
        // variant then there will always be a match.
        if enum_definition.variants.len() > 1 {
            // Create the nonmatching pattern.
            let nonmatching_pat = @ast::pat {
                id: cx.next_id(),
                node: pat_wild,
                span: span
            };

            // Create the nonmatching pattern body.
            let nonmatching_expr = build::mk_bool(cx, span, !is_eq);
            let nonmatching_body_block =
                build::mk_simple_block(cx,
                                       span,
                                       nonmatching_expr);

            // Create the nonmatching arm.
            let nonmatching_arm = ast::arm {
                pats: ~[ nonmatching_pat ],
                guard: None,
                body: nonmatching_body_block,
            };
            other_arms.push(nonmatching_arm);
        }

        // Create the self pattern.
        let self_pat = create_enum_variant_pattern(cx,
                                                   span,
                                                   self_variant,
                                                   ~"__self");

        // Create the self pattern body.
        let other_expr = build::mk_path(cx, span, ~[ other_ident ]);
        let other_expr = build::mk_unary(cx, span, deref, other_expr);
        let other_match_expr = expr_match(other_expr, other_arms);
        let other_match_expr = build::mk_expr(cx,
                                              span,
                                              other_match_expr);
        let other_match_body_block = build::mk_simple_block(cx,
                                                            span,
                                                            other_match_expr);

        // Create the self arm.
        let self_arm = ast::arm {
            pats: ~[ self_pat ],
            guard: None,
            body: other_match_body_block,
        };
        self_arms.push(self_arm);
    }

    // Create the method body.
    let self_expr = build::mk_path(cx, span, ~[ self_ident ]);
    let self_expr = build::mk_unary(cx, span, deref, self_expr);
    let self_match_expr = expr_match(self_expr, self_arms);
    let self_match_expr = build::mk_expr(cx, span, self_match_expr);

    // Create the method.
    return create_eq_method(cx,
                            span,
                            method_ident,
                            type_ident,
                            generics,
                            self_match_expr);
}

fn expand_deriving_eq_struct_tuple_method(cx: ext_ctxt,
                                          span: span,
                                          struct_def: &struct_def,
                                          method_ident: ident,
                                          type_ident: ident,
                                          generics: &Generics,
                                          junction: Junction)
                                        -> @method {
    let self_str = ~"self";
    let other_str = ~"__other";
    let type_path = build::mk_raw_path(span, ~[type_ident]);
    let fields = copy struct_def.fields;

    // Create comparison expression, comparing each of the fields
    let mut match_body = None;
    for fields.eachi |i, _| {
        let other_field_ident = cx.ident_of(other_str + i.to_str());
        let other_field = build::mk_path(cx, span, ~[ other_field_ident ]);

        let self_field_ident = cx.ident_of(self_str + i.to_str());
        let self_field = build::mk_path(cx, span, ~[ self_field_ident ]);

        call_substructure_eq_method(cx, span, self_field, other_field,
            method_ident, junction, &mut match_body);
    }
    let match_body = finish_eq_chain_expr(cx, span, match_body, junction);

    // Create arm for the '__other' match, containing the comparison expr
    let other_subpats = create_subpatterns(cx, span, other_str, fields.len());
    let other_arm = ast::arm {
        pats: ~[ build::mk_pat_enum(cx, span, type_path, other_subpats) ],
        guard: None,
        body: build::mk_simple_block(cx, span, match_body),
    };

    // Create the match on '__other'
    let other_expr = build::mk_path(cx, span, ~[ cx.ident_of(other_str) ]);
    let other_expr = build::mk_unary(cx, span, deref, other_expr);
    let other_match_expr = expr_match(other_expr, ~[other_arm]);
    let other_match_expr = build::mk_expr(cx, span, other_match_expr);

    // Create arm for the 'self' match, which contains the '__other' match
    let self_subpats = create_subpatterns(cx, span, self_str, fields.len());
    let self_arm = ast::arm {
        pats: ~[build::mk_pat_enum(cx, span, type_path, self_subpats)],
        guard: None,
        body: build::mk_simple_block(cx, span, other_match_expr),
    };

    // Create the match on 'self'
    let self_expr = build::mk_path(cx, span, ~[ cx.ident_of(self_str) ]);
    let self_expr = build::mk_unary(cx, span, deref, self_expr);
    let self_match_expr = expr_match(self_expr, ~[self_arm]);
    let self_match_expr = build::mk_expr(cx, span, self_match_expr);

    create_eq_method(cx, span, method_ident,
        type_ident, generics, self_match_expr)
}

fn expand_enum_or_struct_match(cx: ext_ctxt,
                               span: span,
                               arms: ~[ ast::arm ])
                            -> @expr {
    let self_ident = cx.ident_of(~"self");
    let self_expr = build::mk_path(cx, span, ~[ self_ident ]);
    let self_expr = build::mk_unary(cx, span, deref, self_expr);
    let self_match_expr = expr_match(self_expr, arms);
    build::mk_expr(cx, span, self_match_expr)
}

fn expand_deriving_iter_bytes_enum_method(cx: ext_ctxt,
                                          span: span,
                                          enum_definition: &enum_def)
                                       -> @method {
    // Create the arms of the match in the method body.
    let arms = do enum_definition.variants.mapi |i, variant| {
        // Create the matching pattern.
        let pat = create_enum_variant_pattern(cx, span, variant, ~"__self");

        // Determine the discriminant. We will feed this value to the byte
        // iteration function.
        let discriminant;
        match variant.node.disr_expr {
            Some(copy disr_expr) => discriminant = disr_expr,
            None => discriminant = build::mk_uint(cx, span, i),
        }

        // Feed the discriminant to the byte iteration function.
        let mut stmts = ~[];
        let discrim_stmt = call_substructure_iter_bytes_method(cx,
                                                               span,
                                                               discriminant);
        stmts.push(discrim_stmt);

        // Feed each argument in this variant to the byte iteration function
        // as well.
        for uint::range(0, variant_arg_count(cx, span, variant)) |j| {
            // Create the expression for this field.
            let field_ident = cx.ident_of(~"__self" + j.to_str());
            let field = build::mk_path(cx, span, ~[ field_ident ]);

            // Call the substructure method.
            let stmt = call_substructure_iter_bytes_method(cx, span, field);
            stmts.push(stmt);
        }

        // Create the pattern body.
        let match_body_block = build::mk_block_(cx, span, stmts);

        // Create the arm.
        ast::arm {
            pats: ~[ pat ],
            guard: None,
            body: match_body_block,
        }
    };

    // Create the method body.
    let self_match_expr = expand_enum_or_struct_match(cx, span, arms);
    let self_match_stmt = build::mk_stmt(cx, span, self_match_expr);

    // Create the method.
    create_iter_bytes_method(cx, span, ~[ self_match_stmt ])
}

fn expand_deriving_clone_enum_method(cx: ext_ctxt,
                                     span: span,
                                     enum_definition: &enum_def,
                                     type_ident: ident,
                                     generics: &Generics)
                                  -> @method {
    // Create the arms of the match in the method body.
    let arms = do enum_definition.variants.map |variant| {
        // Create the matching pattern.
        let pat = create_enum_variant_pattern(cx, span, variant, ~"__self");

        // Iterate over the variant arguments, creating the subcalls.
        let mut subcalls = ~[];
        for uint::range(0, variant_arg_count(cx, span, variant)) |j| {
            // Create the expression for this field.
            let field_ident = cx.ident_of(~"__self" + j.to_str());
            let field = build::mk_path(cx, span, ~[ field_ident ]);

            // Call the substructure method.
            let subcall = call_substructure_clone_method(cx, span, field);
            subcalls.push(subcall);
        }

        // Create the call to the enum variant (if necessary).
        let call = if subcalls.len() > 0 {
            build::mk_call(cx, span, ~[ variant.node.name ], subcalls)
        } else {
            build::mk_path(cx, span, ~[ variant.node.name ])
        };

        // Create the pattern body.
        let match_body_block = build::mk_simple_block(cx, span, call);

        // Create the arm.
        ast::arm { pats: ~[ pat ], guard: None, body: match_body_block }
    };

    // Create the method body.
    let self_match_expr = expand_enum_or_struct_match(cx, span, arms);

    // Create the method.
    create_clone_method(cx, span, type_ident, generics, self_match_expr)
}

