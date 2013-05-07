// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use ast::*;
use ext::base::ext_ctxt;
use ext::build;
use ext::deriving::*;
use codemap::{span, spanned};
use ast_util;
use opt_vec;

pub fn expand_deriving_decodable(
    cx: @ext_ctxt,
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
    cx: @ext_ctxt,
    span: span,
    type_ident: ident,
    generics: &Generics,
    method: @method
) -> @item {
    let decoder_ty_param = build::mk_ty_param(
        cx,
        cx.ident_of(~"__D"),
        @opt_vec::with(
            build::mk_trait_ty_param_bound_global(
                cx,
                span,
                ~[
                    cx.ident_of(~"std"),
                    cx.ident_of(~"serialize"),
                    cx.ident_of(~"Decoder"),
                ]
            )
        )
    );

    // All the type parameters need to bound to the trait.
    let generic_ty_params = opt_vec::with(decoder_ty_param);

    let methods = [method];
    let trait_path = build::mk_raw_path_global_(
        span,
        ~[
            cx.ident_of(~"std"),
            cx.ident_of(~"serialize"),
            cx.ident_of(~"Decodable")
        ],
        ~[
            build::mk_simple_ty_path(cx, span, cx.ident_of(~"__D"))
        ]
    );
    create_derived_impl(
        cx,
        span,
        type_ident,
        generics,
        methods,
        trait_path,
        generic_ty_params,
        opt_vec::Empty
    )
}

// Creates a method from the given set of statements conforming to the
// signature of the `decodable` method.
fn create_decode_method(
    cx: @ext_ctxt,
    span: span,
    type_ident: ast::ident,
    generics: &Generics,
    expr: @ast::expr
) -> @method {
    // Create the `e` parameter.
    let d_arg_type = build::mk_ty_rptr(
        cx,
        span,
        build::mk_simple_ty_path(cx, span, cx.ident_of(~"__D")),
        ast::m_mutbl
    );
    let d_ident = cx.ident_of(~"__d");
    let d_arg = build::mk_arg(cx, span, d_ident, d_arg_type);

    // Create the type of the return value.
    let output_type = create_self_type_with_params(
        cx,
        span,
        type_ident,
        generics
    );

    // Create the function declaration.
    let inputs = ~[d_arg];
    let fn_decl = build::mk_fn_decl(inputs, output_type);

    // Create the body block.
    let body_block = build::mk_simple_block(cx, span, expr);

    // Create the method.
    let self_ty = spanned { node: sty_static, span: span };
    let method_ident = cx.ident_of(~"decode");
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

fn call_substructure_decode_method(
    cx: @ext_ctxt,
    span: span
) -> @ast::expr {
    // Call the substructure method.
    build::mk_call_(
        cx,
        span,
        build::mk_path_global(
            cx,
            span,
            ~[
                cx.ident_of(~"std"),
                cx.ident_of(~"serialize"),
                cx.ident_of(~"Decodable"),
                cx.ident_of(~"decode"),
            ]
        ),
        ~[
            build::mk_path(cx, span, ~[cx.ident_of(~"__d")])
        ]
    )
}

fn expand_deriving_decodable_struct_def(
    cx: @ext_ctxt,
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
    cx: @ext_ctxt,
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
    cx: @ext_ctxt,
    span: span,
    idx: uint,
    ident: ident
) -> build::Field {
    // Call the substructure method.
    let decode_expr = call_substructure_decode_method(cx, span);

    let d_arg = build::mk_arg(cx,
                              span,
                              cx.ident_of(~"__d"),
                              build::mk_ty_infer(cx, span));

    let call_expr = build::mk_method_call(
        cx,
        span,
        build::mk_path(cx, span, ~[cx.ident_of(~"__d")]),
        cx.ident_of(~"read_struct_field"),
        ~[
            build::mk_base_str(cx, span, cx.str_of(ident)),
            build::mk_uint(cx, span, idx),
            build::mk_lambda(cx,
                             span,
                             build::mk_fn_decl(~[d_arg],
                                               build::mk_ty_infer(cx, span)),
                             decode_expr),
        ]
    );

    build::Field { ident: ident, ex: call_expr }
}

fn create_read_struct_arg(
    cx: @ext_ctxt,
    span: span,
    idx: uint,
    ident: ident
) -> build::Field {
    // Call the substructure method.
    let decode_expr = call_substructure_decode_method(cx, span);

    let call_expr = build::mk_method_call(
        cx,
        span,
        build::mk_path(cx, span, ~[cx.ident_of(~"__d")]),
        cx.ident_of(~"read_struct_arg"),
        ~[
            build::mk_uint(cx, span, idx),
            build::mk_lambda_no_args(cx, span, decode_expr),
        ]
    );

    build::Field { ident: ident, ex: call_expr }
}

fn expand_deriving_decodable_struct_method(
    cx: @ext_ctxt,
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
            named_field(ident, _, _) => {
                fields.push(create_read_struct_field(cx, span, i, ident));
            }
            unnamed_field => {
                cx.span_unimpl(
                    span,
                    ~"unnamed fields with `deriving(Decodable)`"
                );
            }
        }
        i += 1;
    }

    let d_arg = build::mk_arg(cx,
                              span,
                              cx.ident_of(~"__d"),
                              build::mk_ty_infer(cx, span));

    let read_struct_expr = build::mk_method_call(
        cx,
        span,
        build::mk_path(
            cx,
            span,
            ~[cx.ident_of(~"__d")]
        ),
        cx.ident_of(~"read_struct"),
        ~[
            build::mk_base_str(cx, span, cx.str_of(type_ident)),
            build::mk_uint(cx, span, fields.len()),
            build::mk_lambda(
                cx,
                span,
                build::mk_fn_decl(~[d_arg], build::mk_ty_infer(cx, span)),
                build::mk_struct_e(
                    cx,
                    span,
                    ~[type_ident],
                    fields
                )
            ),
        ]
    );

    // Create the method itself.
    create_decode_method(cx, span, type_ident, generics, read_struct_expr)
}

fn create_read_variant_arg(
    cx: @ext_ctxt,
    span: span,
    idx: uint,
    variant: &ast::variant
) -> ast::arm {
    // Create the matching pattern.
    let pat = build::mk_pat_lit(cx, span, build::mk_uint(cx, span, idx));

    // Feed each argument in this variant to the decode function
    // as well.
    let variant_arg_len = variant_arg_count(cx, span, variant);

    let expr = if variant_arg_len == 0 {
        build::mk_path(cx, span, ~[variant.node.name])
    } else {
        // Feed the discriminant to the decode function.
        let mut args = ~[];

        for uint::range(0, variant_arg_len) |j| {
            // Call the substructure method.
            let expr = call_substructure_decode_method(cx, span);

            let d_arg = build::mk_arg(cx,
                                      span,
                                      cx.ident_of(~"__d"),
                                      build::mk_ty_infer(cx, span));
            let t_infer = build::mk_ty_infer(cx, span);

            let call_expr = build::mk_method_call(
                cx,
                span,
                build::mk_path(cx, span, ~[cx.ident_of(~"__d")]),
                cx.ident_of(~"read_enum_variant_arg"),
                ~[
                    build::mk_uint(cx, span, j),
                    build::mk_lambda(cx,
                                     span,
                                     build::mk_fn_decl(~[d_arg], t_infer),
                                     expr),
                ]
            );

            args.push(call_expr);
        }

        build::mk_call(
            cx,
            span,
            ~[variant.node.name],
            args
        )
    };

    // Create the arm.
    build::mk_arm(cx, span, ~[pat], expr)
}

fn create_read_enum_variant(
    cx: @ext_ctxt,
    span: span,
    enum_definition: &enum_def
) -> @expr {
    // Create a vector that contains all the variant names.
    let expr_arm_names = build::mk_base_vec_e(
        cx,
        span,
        do enum_definition.variants.map |variant| {
            build::mk_base_str(
                cx,
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
    arms.push(build::mk_unreachable_arm(cx, span));

    // Create the read_enum_variant expression.
    build::mk_method_call(
        cx,
        span,
        build::mk_path(cx, span, ~[cx.ident_of(~"__d")]),
        cx.ident_of(~"read_enum_variant"),
        ~[
            expr_arm_names,
            build::mk_lambda(
                cx,
                span,
                build::mk_fn_decl(
                    ~[
                        build::mk_arg(
                            cx,
                            span,
                            cx.ident_of(~"__d"),
                            build::mk_ty_infer(cx, span)
                        ),
                        build::mk_arg(
                            cx,
                            span,
                            cx.ident_of(~"__i"),
                            build::mk_ty_infer(cx, span)
                        )
                    ],
                    build::mk_ty_infer(cx, span)
                ),
                build::mk_expr(
                    cx,
                    span,
                    ast::expr_match(
                        build::mk_path(cx, span, ~[cx.ident_of(~"__i")]),
                        arms
                    )
                )
            )
        ]
    )
}

fn expand_deriving_decodable_enum_method(
    cx: @ext_ctxt,
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

    let d_arg = build::mk_arg(cx,
                              span,
                              cx.ident_of(~"__d"),
                              build::mk_ty_infer(cx, span));

    // Create the read_enum expression
    let read_enum_expr = build::mk_method_call(
        cx,
        span,
        build::mk_path(cx, span, ~[cx.ident_of(~"__d")]),
        cx.ident_of(~"read_enum"),
        ~[
            build::mk_base_str(cx, span, cx.str_of(type_ident)),
            build::mk_lambda(cx,
                             span,
                             build::mk_fn_decl(~[d_arg],
                                               build::mk_ty_infer(cx, span)),
                             read_enum_variant_expr),
        ]
    );

    // Create the method.
    create_decode_method(cx, span, type_ident, generics, read_enum_expr)
}
