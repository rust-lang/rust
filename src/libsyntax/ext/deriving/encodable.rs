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

pub fn expand_deriving_encodable(
    cx: @ext_ctxt,
    span: span,
    _mitem: @meta_item,
    in_items: ~[@item]
) -> ~[@item] {
    expand_deriving(
        cx,
        span,
        in_items,
        expand_deriving_encodable_struct_def,
        expand_deriving_encodable_enum_def
    )
}

fn create_derived_encodable_impl(
    cx: @ext_ctxt,
    span: span,
    type_ident: ident,
    generics: &Generics,
    method: @method
) -> @item {
    let encoder_ty_param = build::mk_ty_param(
        cx,
        cx.ident_of(~"__E"),
        @opt_vec::with(
            build::mk_trait_ty_param_bound_global(
                cx,
                span,
                ~[
                    cx.ident_of(~"std"),
                    cx.ident_of(~"serialize"),
                    cx.ident_of(~"Encoder"),
                ]
            )
        )
    );

    // All the type parameters need to bound to the trait.
    let generic_ty_params = opt_vec::with(encoder_ty_param);

    let methods = [method];
    let trait_path = build::mk_raw_path_global_(
        span,
        ~[
            cx.ident_of(~"std"),
            cx.ident_of(~"serialize"),
            cx.ident_of(~"Encodable")
        ],
        ~[
            build::mk_simple_ty_path(cx, span, cx.ident_of(~"__E"))
        ]
    );
    create_derived_impl(
        cx,
        span,
        type_ident,
        generics,
        methods,
        trait_path,
        generic_ty_params
    )
}

// Creates a method from the given set of statements conforming to the
// signature of the `encodable` method.
fn create_encode_method(
    cx: @ext_ctxt,
    span: span,
    +statements: ~[@stmt]
) -> @method {
    // Create the `e` parameter.
    let e_arg_type = build::mk_ty_rptr(
        cx,
        span,
        build::mk_simple_ty_path(cx, span, cx.ident_of(~"__E")),
        ast::m_imm
    );
    let e_ident = cx.ident_of(~"__e");
    let e_arg = build::mk_arg(cx, span, e_ident, e_arg_type);

    // Create the type of the return value.
    let output_type = @ast::Ty { id: cx.next_id(), node: ty_nil, span: span };

    // Create the function declaration.
    let inputs = ~[e_arg];
    let fn_decl = build::mk_fn_decl(inputs, output_type);

    // Create the body block.
    let body_block = build::mk_block_(cx, span, statements);

    // Create the method.
    let self_ty = spanned { node: sty_region(None, m_imm), span: span };
    let method_ident = cx.ident_of(~"encode");
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

fn call_substructure_encode_method(
    cx: @ext_ctxt,
    span: span,
    self_field: @expr
) -> @ast::expr {
    // Gather up the parameters we want to chain along.
    let e_ident = cx.ident_of(~"__e");
    let e_expr = build::mk_path(cx, span, ~[e_ident]);

    // Call the substructure method.
    let encode_ident = cx.ident_of(~"encode");
    build::mk_method_call(
        cx,
        span,
        self_field,
        encode_ident,
        ~[e_expr]
    )
}

fn expand_deriving_encodable_struct_def(
    cx: @ext_ctxt,
    span: span,
    struct_def: &struct_def,
    type_ident: ident,
    generics: &Generics
) -> @item {
    // Create the method.
    let method = expand_deriving_encodable_struct_method(
        cx,
        span,
        type_ident,
        struct_def
    );

    // Create the implementation.
    create_derived_encodable_impl(
        cx,
        span,
        type_ident,
        generics,
        method
    )
}

fn expand_deriving_encodable_enum_def(
    cx: @ext_ctxt,
    span: span,
    enum_definition: &enum_def,
    type_ident: ident,
    generics: &Generics
) -> @item {
    // Create the method.
    let method = expand_deriving_encodable_enum_method(
        cx,
        span,
        type_ident,
        enum_definition
    );

    // Create the implementation.
    create_derived_encodable_impl(
        cx,
        span,
        type_ident,
        generics,
        method
    )
}

fn expand_deriving_encodable_struct_method(
    cx: @ext_ctxt,
    span: span,
    type_ident: ident,
    struct_def: &struct_def
) -> @method {
    let self_ident = cx.ident_of(~"self");

    // Create the body of the method.
    let mut idx = 0;
    let mut statements = ~[];
    for struct_def.fields.each |struct_field| {
        match struct_field.node.kind {
            named_field(ident, _, _) => {
                // Create the accessor for this field.
                let self_field = build::mk_access(
                    cx,
                    span,
                    ~[self_ident],
                    ident
                );

                // Call the substructure method.
                let encode_expr = call_substructure_encode_method(
                    cx,
                    span,
                    self_field
                );

                let blk_expr = build::mk_lambda(
                    cx,
                    span,
                    build::mk_fn_decl(~[], build::mk_ty_infer(cx, span)),
                    encode_expr
                );

                let call_expr = build::mk_method_call(
                    cx,
                    span,
                    build::mk_path(cx, span, ~[cx.ident_of(~"__e")]),
                    cx.ident_of(~"emit_struct_field"),
                    ~[
                        build::mk_base_str(cx, span, cx.str_of(ident)),
                        build::mk_uint(cx, span, idx),
                        blk_expr
                    ]
                );

                statements.push(build::mk_stmt(cx, span, call_expr));
            }
            unnamed_field => {
                cx.span_unimpl(
                    span,
                    ~"unnamed fields with `deriving(Encodable)`"
                );
            }
        }
        idx += 1;
    }

    let emit_struct_stmt = build::mk_method_call(
        cx,
        span,
        build::mk_path(
            cx,
            span,
            ~[cx.ident_of(~"__e")]
        ),
        cx.ident_of(~"emit_struct"),
        ~[
            build::mk_base_str(cx, span, cx.str_of(type_ident)),
            build::mk_uint(cx, span, statements.len()),
            build::mk_lambda_stmts(
                cx,
                span,
                build::mk_fn_decl(~[], build::mk_ty_infer(cx, span)),
                statements
            ),
        ]
    );

    let statements = ~[build::mk_stmt(cx, span, emit_struct_stmt)];

    // Create the method itself.
    return create_encode_method(cx, span, statements);
}

fn expand_deriving_encodable_enum_method(
    cx: @ext_ctxt,
    span: span,
    type_ident: ast::ident,
    enum_definition: &enum_def
) -> @method {
    // Create the arms of the match in the method body.
    let arms = do enum_definition.variants.mapi |i, variant| {
        // Create the matching pattern.
        let pat = create_enum_variant_pattern(cx, span, variant, ~"__self");

        // Feed the discriminant to the encode function.
        let mut stmts = ~[];

        // Feed each argument in this variant to the encode function
        // as well.
        let variant_arg_len = variant_arg_count(cx, span, variant);
        for uint::range(0, variant_arg_len) |j| {
            // Create the expression for this field.
            let field_ident = cx.ident_of(~"__self" + j.to_str());
            let field = build::mk_path(cx, span, ~[ field_ident ]);

            // Call the substructure method.
            let expr = call_substructure_encode_method(cx, span, field);

            let blk_expr = build::mk_lambda(
                cx,
                span,
                build::mk_fn_decl(~[], build::mk_ty_infer(cx, span)),
                expr
            );

            let call_expr = build::mk_method_call(
                cx,
                span,
                build::mk_path(cx, span, ~[cx.ident_of(~"__e")]),
                cx.ident_of(~"emit_enum_variant_arg"),
                ~[
                    build::mk_uint(cx, span, j),
                    blk_expr,
                ]
            );

            stmts.push(build::mk_stmt(cx, span, call_expr));
        }

        // Create the pattern body.
        let call_expr = build::mk_method_call(
            cx,
            span,
            build::mk_path(cx, span, ~[cx.ident_of(~"__e")]),
            cx.ident_of(~"emit_enum_variant"),
            ~[
                build::mk_base_str(cx, span, cx.str_of(variant.node.name)),
                build::mk_uint(cx, span, i),
                build::mk_uint(cx, span, variant_arg_len),
                build::mk_lambda_stmts(
                    cx,
                    span,
                    build::mk_fn_decl(~[], build::mk_ty_infer(cx, span)),
                    stmts
                )
            ]
        );

        let match_body_block = build::mk_simple_block(cx, span, call_expr);

        // Create the arm.
        ast::arm {
            pats: ~[pat],
            guard: None,
            body: match_body_block,
        }
    };

    // Create the method body.
    let lambda_expr = build::mk_lambda(
        cx,
        span,
        build::mk_fn_decl(~[], build::mk_ty_infer(cx, span)),
        expand_enum_or_struct_match(cx, span, arms)
    );

    let call_expr = build::mk_method_call(
        cx,
        span,
        build::mk_path(cx, span, ~[cx.ident_of(~"__e")]),
        cx.ident_of(~"emit_enum"),
        ~[
            build::mk_base_str(cx, span, cx.str_of(type_ident)),
            lambda_expr,
        ]
    );

    let stmt = build::mk_stmt(cx, span, call_expr);

    // Create the method.
    create_encode_method(cx, span, ~[stmt])
}
