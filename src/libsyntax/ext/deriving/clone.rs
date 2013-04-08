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
use opt_vec;

use core::uint;

pub fn expand_deriving_clone(cx: @ext_ctxt,
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

pub fn expand_deriving_obsolete(cx: @ext_ctxt,
                                span: span,
                                _mitem: @meta_item,
                                in_items: ~[@item])
                             -> ~[@item] {
    cx.span_err(span, ~"`#[deriving_clone]` is obsolete; use `#[deriving(Clone)]` instead");
    in_items
}

fn create_derived_clone_impl(cx: @ext_ctxt,
                             span: span,
                             type_ident: ident,
                             generics: &Generics,
                             method: @method)
                          -> @item {
    let methods = [ method ];
    let trait_path = ~[
        cx.ident_of(~"core"),
        cx.ident_of(~"clone"),
        cx.ident_of(~"Clone"),
    ];
    let trait_path = build::mk_raw_path_global(span, trait_path);
    create_derived_impl(cx, span, type_ident, generics, methods, trait_path, opt_vec::Empty)
}
// Creates a method from the given expression conforming to the signature of
// the `clone` method.
fn create_clone_method(cx: @ext_ctxt,
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
    let self_ty = spanned { node: sty_region(None, m_imm), span: span };
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

fn call_substructure_clone_method(cx: @ext_ctxt,
                                  span: span,
                                  self_field: @expr)
                               -> @expr {
    // Call the substructure method.
    let clone_ident = cx.ident_of(~"clone");
    build::mk_method_call(cx, span,
                          self_field, clone_ident,
                          ~[])
}

fn expand_deriving_clone_struct_def(cx: @ext_ctxt,
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

fn expand_deriving_clone_enum_def(cx: @ext_ctxt,
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

fn expand_deriving_clone_struct_method(cx: @ext_ctxt,
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
                cx.span_bug(span, ~"unnamed fields in `deriving(Clone)`");
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

fn expand_deriving_clone_tuple_struct_method(cx: @ext_ctxt,
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

fn expand_deriving_clone_enum_method(cx: @ext_ctxt,
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
