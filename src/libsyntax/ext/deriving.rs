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
use ast::{enum_variant_kind, expr, expr_match, ident, item, item_};
use ast::{item_enum, item_impl, item_struct, m_imm, meta_item, method};
use ast::{named_field, or, pat, pat_ident, pat_wild, public, pure_fn};
use ast::{re_anon, spanned, stmt, struct_def, struct_variant_kind};
use ast::{sty_by_ref, sty_region, tuple_variant_kind, ty_nil, ty_param};
use ast::{ty_param_bound, ty_path, ty_rptr, unnamed_field, variant};
use ext::base::ext_ctxt;
use ext::build;
use codemap::span;
use parse::token::special_idents::clownshoes_extensions;

use core::dvec;
use core::uint;

enum Junction {
    Conjunction,
    Disjunction,
}

impl Junction {
    fn to_binop(self) -> binop {
        match self {
            Conjunction => and,
            Disjunction => or,
        }
    }
}

type ExpandDerivingStructDefFn = &fn(ext_ctxt,
                                     span,
                                     x: &struct_def,
                                     ident,
                                     +y: ~[ty_param])
                                  -> @item;
type ExpandDerivingEnumDefFn = &fn(ext_ctxt,
                                   span,
                                   x: &enum_def,
                                   ident,
                                   +y: ~[ty_param])
                                -> @item;

pub fn expand_deriving_eq(cx: ext_ctxt,
                          span: span,
                          _mitem: meta_item,
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
                                  _mitem: meta_item,
                                  in_items: ~[@item])
                               -> ~[@item] {
    expand_deriving(cx,
                    span,
                    in_items,
                    expand_deriving_iter_bytes_struct_def,
                    expand_deriving_iter_bytes_enum_def)
}

fn expand_deriving(cx: ext_ctxt,
                   span: span,
                   in_items: ~[@item],
                   expand_deriving_struct_def: ExpandDerivingStructDefFn,
                   expand_deriving_enum_def: ExpandDerivingEnumDefFn)
                -> ~[@item] {
    let result = dvec::DVec();
    for in_items.each |item| {
        result.push(copy *item);
        match item.node {
            item_struct(struct_def, copy ty_params) => {
                result.push(expand_deriving_struct_def(cx,
                                                       span,
                                                       struct_def,
                                                       item.ident,
                                                       move ty_params));
            }
            item_enum(ref enum_definition, copy ty_params) => {
                result.push(expand_deriving_enum_def(cx,
                                                     span,
                                                     enum_definition,
                                                     item.ident,
                                                     move ty_params));
            }
            _ => ()
        }
    }
    dvec::unwrap(move result)
}

fn create_impl_item(cx: ext_ctxt, span: span, +item: item_) -> @item {
    @ast::item {
        ident: clownshoes_extensions,
        attrs: ~[],
        id: cx.next_id(),
        node: move item,
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
                    ty_params: &[ty_param],
                    body: @expr)
                 -> @method {
    // Create the type of the `other` parameter.
    let arg_path_type = create_self_type_with_params(cx,
                                                     span,
                                                     type_ident,
                                                     ty_params);
    let arg_region = @{ id: cx.next_id(), node: re_anon };
    let arg_type = ty_rptr(
        arg_region,
        ast::mt { ty: arg_path_type, mutbl: m_imm }
    );
    let arg_type = @{ id: cx.next_id(), node: move arg_type, span: span };

    // Create the `other` parameter.
    let other_ident = cx.ident_of(~"__other");
    let arg = build::mk_arg(cx, span, other_ident, arg_type);

    // Create the type of the return value.
    let bool_ident = cx.ident_of(~"bool");
    let output_type = build::mk_raw_path(span, ~[ bool_ident ]);
    let output_type = ty_path(output_type, cx.next_id());
    let output_type = @{
        id: cx.next_id(),
        node: move output_type,
        span: span
    };

    // Create the function declaration.
    let fn_decl = build::mk_fn_decl(~[ move arg ], output_type);

    // Create the body block.
    let body_block = build::mk_simple_block(cx, span, body);

    // Create the method.
    let self_ty = spanned { node: sty_region(m_imm), span: span };
    return @{
        ident: method_ident,
        attrs: ~[],
        tps: ~[],
        self_ty: self_ty,
        purity: pure_fn,
        decl: move fn_decl,
        body: move body_block,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: public
    };
}

fn create_self_type_with_params(cx: ext_ctxt,
                                span: span,
                                type_ident: ident,
                                ty_params: &[ty_param])
                             -> @Ty {
    // Create the type parameters on the `self` path.
    let self_ty_params = dvec::DVec();
    for ty_params.each |ty_param| {
        let self_ty_param = build::mk_simple_ty_path(cx,
                                                     span,
                                                     ty_param.ident);
        self_ty_params.push(move self_ty_param);
    }
    let self_ty_params = dvec::unwrap(move self_ty_params);

    // Create the type of `self`.
    let self_type = build::mk_raw_path_(span,
                                        ~[ type_ident ],
                                        move self_ty_params);
    let self_type = ty_path(self_type, cx.next_id());
    @{ id: cx.next_id(), node: move self_type, span: span }
}

fn create_derived_impl(cx: ext_ctxt,
                       span: span,
                       type_ident: ident,
                       +ty_params: ~[ty_param],
                       methods: &[@method],
                       trait_path: &[ident])
                    -> @item {
    // Create the type parameters.
    let impl_ty_params = dvec::DVec();
    for ty_params.each |ty_param| {
        let bound = build::mk_ty_path_global(cx,
                                             span,
                                             trait_path.map(|x| *x));
        let bounds = @~[ TraitTyParamBound(bound) ];
        let impl_ty_param = build::mk_ty_param(cx, ty_param.ident, bounds);
        impl_ty_params.push(move impl_ty_param);
    }
    let impl_ty_params = dvec::unwrap(move impl_ty_params);

    // Create the reference to the trait.
    let trait_path = ast::path {
        span: span,
        global: true,
        idents: trait_path.map(|x| *x),
        rp: None,
        types: ~[]
    };
    let trait_path = @move trait_path;
    let trait_ref = {
        path: trait_path,
        ref_id: cx.next_id()
    };
    let trait_ref = @move trait_ref;

    // Create the type of `self`.
    let self_type = create_self_type_with_params(cx,
                                                 span,
                                                 type_ident,
                                                 ty_params);

    // Create the impl item.
    let impl_item = item_impl(move impl_ty_params,
                              Some(trait_ref),
                              self_type,
                              methods.map(|x| *x));
    return create_impl_item(cx, span, move impl_item);
}

fn create_derived_eq_impl(cx: ext_ctxt,
                          span: span,
                          type_ident: ident,
                          +ty_params: ~[ty_param],
                          eq_method: @method,
                          ne_method: @method)
                       -> @item {
    let methods = [ eq_method, ne_method ];
    let trait_path = [
        cx.ident_of(~"core"),
        cx.ident_of(~"cmp"),
        cx.ident_of(~"Eq")
    ];
    create_derived_impl(cx, span, type_ident, ty_params, methods, trait_path)
}

fn create_derived_iter_bytes_impl(cx: ext_ctxt,
                                  span: span,
                                  type_ident: ident,
                                  +ty_params: ~[ty_param],
                                  method: @method)
                               -> @item {
    let methods = [ method ];
    let trait_path = [
        cx.ident_of(~"core"),
        cx.ident_of(~"to_bytes"),
        cx.ident_of(~"IterBytes")
    ];
    create_derived_impl(cx, span, type_ident, ty_params, methods, trait_path)
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
    let output_type = @{ id: cx.next_id(), node: ty_nil, span: span };

    // Create the function declaration.
    let inputs = ~[ move lsb0_arg, move f_arg ];
    let fn_decl = build::mk_fn_decl(move inputs, output_type);

    // Create the body block.
    let body_block = build::mk_block_(cx, span, move statements);

    // Create the method.
    let self_ty = spanned { node: sty_region(m_imm), span: span };
    let method_ident = cx.ident_of(~"iter_bytes");
    return @{
        ident: method_ident,
        attrs: ~[],
        tps: ~[],
        self_ty: self_ty,
        purity: pure_fn,
        decl: move fn_decl,
        body: move body_block,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: public
    }
}

fn create_subpatterns(cx: ext_ctxt,
                      span: span,
                      prefix: ~str,
                      n: uint)
                   -> ~[@pat] {
    let subpats = dvec::DVec();
    for uint::range(0, n) |_i| {
        // Create the subidentifier.
        let index = subpats.len().to_str();
        let ident = cx.ident_of(prefix + index);

        // Create the subpattern.
        let subpath = build::mk_raw_path(span, ~[ ident ]);
        let subpat = pat_ident(bind_by_ref(m_imm), subpath, None);
        let subpat = build::mk_pat(cx, span, move subpat);
        subpats.push(subpat);
    }
    return dvec::unwrap(move subpats);
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
                return build::mk_pat_ident(cx, span, variant_ident);
            }

            let matching_path = build::mk_raw_path(span, ~[ variant_ident ]);
            let subpats = create_subpatterns(cx,
                                             span,
                                             prefix,
                                             variant_args.len());

            return build::mk_pat_enum(cx, span, matching_path, move subpats);
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

fn variant_arg_count(cx: ext_ctxt, span: span, variant: &variant) -> uint {
    match variant.node.kind {
        tuple_variant_kind(args) => args.len(),
        struct_variant_kind(struct_def) => struct_def.fields.len(),
        enum_variant_kind(*) => {
            cx.span_bug(span, ~"variant_arg_count: enum variants deprecated")
        }
    }
}

fn expand_deriving_eq_struct_def(cx: ext_ctxt,
                                 span: span,
                                 struct_def: &struct_def,
                                 type_ident: ident,
                                 +ty_params: ~[ty_param])
                              -> @item {
    // Create the methods.
    let eq_ident = cx.ident_of(~"eq");
    let ne_ident = cx.ident_of(~"ne");
    let eq_method = expand_deriving_eq_struct_method(cx,
                                                     span,
                                                     struct_def,
                                                     eq_ident,
                                                     type_ident,
                                                     ty_params,
                                                     Conjunction);
    let ne_method = expand_deriving_eq_struct_method(cx,
                                                     span,
                                                     struct_def,
                                                     ne_ident,
                                                     type_ident,
                                                     ty_params,
                                                     Disjunction);

    // Create the implementation.
    return create_derived_eq_impl(cx,
                                  span,
                                  type_ident,
                                  move ty_params,
                                  eq_method,
                                  ne_method);
}

fn expand_deriving_eq_enum_def(cx: ext_ctxt,
                               span: span,
                               enum_definition: &enum_def,
                               type_ident: ident,
                               +ty_params: ~[ty_param])
                            -> @item {
    // Create the methods.
    let eq_ident = cx.ident_of(~"eq");
    let ne_ident = cx.ident_of(~"ne");
    let eq_method = expand_deriving_eq_enum_method(cx,
                                                   span,
                                                   enum_definition,
                                                   eq_ident,
                                                   type_ident,
                                                   ty_params,
                                                   Conjunction);
    let ne_method = expand_deriving_eq_enum_method(cx,
                                                   span,
                                                   enum_definition,
                                                   ne_ident,
                                                   type_ident,
                                                   ty_params,
                                                   Disjunction);

    // Create the implementation.
    return create_derived_eq_impl(cx,
                                  span,
                                  type_ident,
                                  move ty_params,
                                  eq_method,
                                  ne_method);
}

fn expand_deriving_iter_bytes_struct_def(cx: ext_ctxt,
                                         span: span,
                                         struct_def: &struct_def,
                                         type_ident: ident,
                                         +ty_params: ~[ty_param])
                                      -> @item {
    // Create the method.
    let method = expand_deriving_iter_bytes_struct_method(cx,
                                                          span,
                                                          struct_def);

    // Create the implementation.
    return create_derived_iter_bytes_impl(cx,
                                          span,
                                          type_ident,
                                          move ty_params,
                                          method);
}

fn expand_deriving_iter_bytes_enum_def(cx: ext_ctxt,
                                       span: span,
                                       enum_definition: &enum_def,
                                       type_ident: ident,
                                       +ty_params: ~[ty_param])
                                    -> @item {
    // Create the method.
    let method = expand_deriving_iter_bytes_enum_method(cx,
                                                        span,
                                                        enum_definition);

    // Create the implementation.
    return create_derived_iter_bytes_impl(cx,
                                          span,
                                          type_ident,
                                          move ty_params,
                                          method);
}

fn expand_deriving_eq_struct_method(cx: ext_ctxt,
                                    span: span,
                                    struct_def: &struct_def,
                                    method_ident: ident,
                                    type_ident: ident,
                                    ty_params: &[ty_param],
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
                            ty_params,
                            body);
}

fn expand_deriving_iter_bytes_struct_method(cx: ext_ctxt,
                                            span: span,
                                            struct_def: &struct_def)
                                         -> @method {
    let self_ident = cx.ident_of(~"self");

    // Create the body of the method.
    let statements = dvec::DVec();
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
    let statements = dvec::unwrap(move statements);
    return create_iter_bytes_method(cx, span, move statements);
}

fn expand_deriving_eq_enum_method(cx: ext_ctxt,
                                  span: span,
                                  enum_definition: &enum_def,
                                  method_ident: ident,
                                  type_ident: ident,
                                  ty_params: &[ty_param],
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
    let self_arms = dvec::DVec();
    for enum_definition.variants.each |self_variant| {
        let other_arms = dvec::DVec();

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
            body: move matching_body_block
        };
        other_arms.push(move matching_arm);

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
        let other_arms = dvec::unwrap(move other_arms);
        let other_match_expr = expr_match(other_expr, move other_arms);
        let other_match_expr = build::mk_expr(cx,
                                              span,
                                              move other_match_expr);
        let other_match_body_block = build::mk_simple_block(cx,
                                                            span,
                                                            other_match_expr);

        // Create the self arm.
        let self_arm = ast::arm {
            pats: ~[ self_pat ],
            guard: None,
            body: other_match_body_block,
        };
        self_arms.push(move self_arm);
    }

    // Create the method body.
    let self_expr = build::mk_path(cx, span, ~[ self_ident ]);
    let self_expr = build::mk_unary(cx, span, deref, self_expr);
    let self_arms = dvec::unwrap(move self_arms);
    let self_match_expr = expr_match(self_expr, move self_arms);
    let self_match_expr = build::mk_expr(cx, span, move self_match_expr);

    // Create the method.
    return create_eq_method(cx,
                            span,
                            method_ident,
                            type_ident,
                            ty_params,
                            self_match_expr);
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
        let stmts = dvec::DVec();
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
        let stmts = dvec::unwrap(move stmts);
        let match_body_block = build::mk_block_(cx, span, move stmts);

        // Create the arm.
        ast::arm {
            pats: ~[ pat ],
            guard: None,
            body: match_body_block,
        }
    };

    // Create the method body.
    let self_ident = cx.ident_of(~"self");
    let self_expr = build::mk_path(cx, span, ~[ self_ident ]);
    let self_expr = build::mk_unary(cx, span, deref, self_expr);
    let self_match_expr = expr_match(self_expr, arms);
    let self_match_expr = build::mk_expr(cx, span, self_match_expr);
    let self_match_stmt = build::mk_stmt(cx, span, self_match_expr);

    // Create the method.
    create_iter_bytes_method(cx, span, ~[ self_match_stmt ])
}

