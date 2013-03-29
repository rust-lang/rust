// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use ast;
use ast::*;
use ext::base::ext_ctxt;
use ext::build;
use ext::deriving::*;
use codemap::{span, spanned};
use ast_util;

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

pub fn expand_deriving_eq(cx: @ext_ctxt,
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

pub fn expand_deriving_obsolete(cx: @ext_ctxt,
                                span: span,
                                _mitem: @meta_item,
                                in_items: ~[@item])
                             -> ~[@item] {
    cx.span_err(span, ~"`#[deriving_eq]` is obsolete; use `#[deriving(Eq)]` instead");
    in_items
}

/// Creates a method from the given expression, the signature of which
/// conforms to the `eq` or `ne` method.
fn create_eq_method(cx: @ext_ctxt,
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
    let self_ty = spanned { node: sty_region(None, m_imm), span: span };
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
        vis: public
    }
}

fn create_derived_eq_impl(cx: @ext_ctxt,
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

fn call_substructure_eq_method(cx: @ext_ctxt,
                               span: span,
                               self_field: @expr,
                               other_field_ref: @expr,
                               method_ident: ident,
                               junction: Junction,
                               chain_expr: &mut Option<@expr>) {
    // Call the substructure method.
    let self_call = build::mk_method_call(cx, span,
                                          self_field, method_ident,
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

fn finish_eq_chain_expr(cx: @ext_ctxt,
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

fn expand_deriving_eq_struct_def(cx: @ext_ctxt,
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

fn expand_deriving_eq_enum_def(cx: @ext_ctxt,
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

fn expand_deriving_eq_struct_method(cx: @ext_ctxt,
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

fn expand_deriving_eq_enum_method(cx: @ext_ctxt,
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

fn expand_deriving_eq_struct_tuple_method(cx: @ext_ctxt,
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
