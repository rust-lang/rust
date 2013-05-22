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
The compiler code necessary for #[deriving(Decodable)]. See
encodable.rs for more.
*/

use ast;
use ast::*;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::deriving::*;
use codemap::{span, spanned};
use ast_util;
use opt_vec;

pub fn expand_deriving_decodable(
    cx: @ExtCtxt,
    span: span,
    _mitem: @meta_item,
    in_items: ~[@item]
) -> ~[@item] {
    expand_deriving(
        cx,
        span,
        in_items,
        expand_deriving_decodable_struct_def,
        expand_deriving_decodable_enum_def
    )
}

fn create_derived_decodable_impl(
    cx: @ExtCtxt,
    span: span,
    type_ident: ident,
    generics: &Generics,
    method: @method
) -> @item {
    let decoder_ty_param = cx.typaram(
        cx.ident_of("__D"),
        @opt_vec::with(
            cx.typarambound(
                cx.path_global(
                    span,
                    ~[
                        cx.ident_of("std"),
                        cx.ident_of("serialize"),
                        cx.ident_of("Decoder"),
                    ]))));

    // All the type parameters need to bound to the trait.
    let generic_ty_params = opt_vec::with(decoder_ty_param);

    let methods = [method];
    let trait_path = cx.path_all(
        span,
        true,
        ~[
            cx.ident_of("std"),
            cx.ident_of("serialize"),
            cx.ident_of("Decodable")
        ],
        None,
        ~[
            cx.ty_ident(span, cx.ident_of("__D"))
        ]
    );
    create_derived_impl(
        cx,
        span,
        type_ident,
        generics,
        methods,
        trait_path,
        Generics { ty_params: generic_ty_params, lifetimes: opt_vec::Empty },
        opt_vec::Empty
    )
}

// Creates a method from the given set of statements conforming to the
// signature of the `decodable` method.
fn create_decode_method(
    cx: @ExtCtxt,
    span: span,
    type_ident: ast::ident,
    generics: &Generics,
    expr: @ast::expr
) -> @method {
    // Create the `e` parameter.
    let d_arg_type = cx.ty_rptr(
        span,
        cx.ty_ident(span, cx.ident_of("__D")),
        None,
        ast::m_mutbl
    );
    let d_ident = cx.ident_of("__d");
    let d_arg = cx.arg(span, d_ident, d_arg_type);

    // Create the type of the return value.
    let output_type = create_self_type_with_params(
        cx,
        span,
        type_ident,
        generics
    );

    // Create the function declaration.
    let inputs = ~[d_arg];
    let fn_decl = cx.fn_decl(inputs, output_type);

    // Create the body block.
    let body_block = cx.blk_expr(expr);

    // Create the method.
    let explicit_self = spanned { node: sty_static, span: span };
    let method_ident = cx.ident_of("decode");
    @ast::method {
        ident: method_ident,
        attrs: ~[],
        generics: ast_util::empty_generics(),
        explicit_self: explicit_self,
        purity: impure_fn,
        decl: fn_decl,
        body: body_block,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: public
    }
}

fn call_substructure_decode_method(
    cx: @ExtCtxt,
    span: span
) -> @ast::expr {
    // Call the substructure method.
    cx.expr_call(
        span,
        cx.expr_path(
            cx.path_global(
                span,
                ~[
                    cx.ident_of("std"),
                    cx.ident_of("serialize"),
                    cx.ident_of("Decodable"),
                    cx.ident_of("decode"),
                ]
            )
        ),
        ~[
            cx.expr_ident(span, cx.ident_of("__d"))
        ]
    )
}

fn expand_deriving_decodable_struct_def(
    cx: @ExtCtxt,
    span: span,
    struct_def: &struct_def,
    type_ident: ident,
    generics: &Generics
) -> @item {
    // Create the method.
    let method = expand_deriving_decodable_struct_method(
        cx,
        span,
        struct_def,
        type_ident,
        generics
    );

    // Create the implementation.
    create_derived_decodable_impl(
        cx,
        span,
        type_ident,
        generics,
        method
    )
}

fn expand_deriving_decodable_enum_def(
    cx: @ExtCtxt,
    span: span,
    enum_definition: &enum_def,
    type_ident: ident,
    generics: &Generics
) -> @item {
    // Create the method.
    let method = expand_deriving_decodable_enum_method(
        cx,
        span,
        enum_definition,
        type_ident,
        generics
    );

    // Create the implementation.
    create_derived_decodable_impl(
        cx,
        span,
        type_ident,
        generics,
        method
    )
}

fn create_read_struct_field(
    cx: @ExtCtxt,
    span: span,
    idx: uint,
    ident: ident
) -> ast::field {
    // Call the substructure method.
    let decode_expr = call_substructure_decode_method(cx, span);

    let d_id = cx.ident_of("__d");

    let call_expr = cx.expr_method_call(
        span,
        cx.expr_ident(span, d_id),
        cx.ident_of("read_struct_field"),
        ~[
            cx.expr_str(span, cx.str_of(ident)),
            cx.expr_uint(span, idx),
            cx.lambda_expr_1(span, decode_expr, d_id)
        ]
    );

    cx.field_imm(span, ident, call_expr)
}

fn create_read_struct_arg(
    cx: @ExtCtxt,
    span: span,
    idx: uint,
    ident: ident
) -> ast::field {
    // Call the substructure method.
    let decode_expr = call_substructure_decode_method(cx, span);

    let call_expr = cx.expr_method_call(
        span,
        cx.expr_ident(span, cx.ident_of("__d")),
        cx.ident_of("read_struct_arg"),
        ~[
            cx.expr_uint(span, idx),
            cx.lambda_expr_0(span, decode_expr),
        ]
    );

    cx.field_imm(span, ident, call_expr)
}

fn expand_deriving_decodable_struct_method(
    cx: @ExtCtxt,
    span: span,
    struct_def: &struct_def,
    type_ident: ident,
    generics: &Generics
) -> @method {
    // Create the body of the method.
    let mut i = 0;
    let mut fields = ~[];
    for struct_def.fields.each |struct_field| {
        match struct_field.node.kind {
            named_field(ident, _) => {
                fields.push(create_read_struct_field(cx, span, i, ident));
            }
            unnamed_field => {
                cx.span_unimpl(
                    span,
                    "unnamed fields with `deriving(Decodable)`"
                );
            }
        }
        i += 1;
    }

    let d_id = cx.ident_of("__d");

    let read_struct_expr = cx.expr_method_call(
        span,
        cx.expr_ident(span, d_id),
        cx.ident_of("read_struct"),
        ~[
            cx.expr_str(span, cx.str_of(type_ident)),
            cx.expr_uint(span, fields.len()),
            cx.lambda_expr_1(
                span,
                cx.expr_struct_ident(span, type_ident, fields),
                d_id)
        ]
    );

    // Create the method itself.
    create_decode_method(cx, span, type_ident, generics, read_struct_expr)
}

fn create_read_variant_arg(
    cx: @ExtCtxt,
    span: span,
    idx: uint,
    variant: &ast::variant
) -> ast::arm {
    // Create the matching pattern.
    let pat = cx.pat_lit(span, cx.expr_uint(span, idx));

    // Feed each argument in this variant to the decode function
    // as well.
    let variant_arg_len = variant_arg_count(cx, span, variant);

    let expr = if variant_arg_len == 0 {
        cx.expr_ident(span, variant.node.name)
    } else {
        // Feed the discriminant to the decode function.
        let mut args = ~[];

        for uint::range(0, variant_arg_len) |j| {
            // Call the substructure method.
            let expr = call_substructure_decode_method(cx, span);

            let d_id = cx.ident_of("__d");

            let call_expr = cx.expr_method_call(
                span,
                cx.expr_ident(span, d_id),
                cx.ident_of("read_enum_variant_arg"),
                ~[
                    cx.expr_uint(span, j),
                    cx.lambda_expr_1(span, expr, d_id),
                ]
            );

            args.push(call_expr);
        }

        cx.expr_call_ident(span, variant.node.name, args)
    };

    // Create the arm.
    cx.arm(span, ~[pat], expr)
}

fn create_read_enum_variant(
    cx: @ExtCtxt,
    span: span,
    enum_definition: &enum_def
) -> @expr {
    // Create a vector that contains all the variant names.
    let expr_arm_names = cx.expr_vec(
        span,
        do enum_definition.variants.map |variant| {
            cx.expr_str(
                span,
                cx.str_of(variant.node.name)
            )
        }
    );

    // Create the arms of the match in the method body.
    let mut arms = do enum_definition.variants.mapi |i, variant| {
        create_read_variant_arg(cx, span, i, variant)
    };

    // Add the impossible case arm.
    arms.push(cx.arm_unreachable(span));

    // Create the read_enum_variant expression.
    cx.expr_method_call(
        span,
        cx.expr_ident(span, cx.ident_of("__d")),
        cx.ident_of("read_enum_variant"),
        ~[
            expr_arm_names,
            cx.lambda_expr(span,
                           ~[cx.ident_of("__d"), cx.ident_of("__i")],
                           cx.expr_match(span, cx.expr_ident(span, cx.ident_of("__i")), arms))
        ]
    )
}

fn expand_deriving_decodable_enum_method(
    cx: @ExtCtxt,
    span: span,
    enum_definition: &enum_def,
    type_ident: ast::ident,
    generics: &Generics
) -> @method {
    let read_enum_variant_expr = create_read_enum_variant(
        cx,
        span,
        enum_definition
    );

    let d_id = cx.ident_of("__d");

    // Create the read_enum expression
    let read_enum_expr = cx.expr_method_call(
        span,
        cx.expr_ident(span, d_id),
        cx.ident_of("read_enum"),
        ~[
            cx.expr_str(span, cx.str_of(type_ident)),
            cx.lambda_expr_1(span, read_enum_variant_expr, d_id)
        ]
    );

    // Create the method.
    create_decode_method(cx, span, type_ident, generics, read_enum_expr)
}
