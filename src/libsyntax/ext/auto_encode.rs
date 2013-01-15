// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

The compiler code necessary to implement the #[auto_encode] and
#[auto_decode] extension.  The idea here is that type-defining items may
be tagged with #[auto_encode] and #[auto_decode], which will cause
us to generate a little companion module with the same name as the item.

For example, a type like:

    #[auto_encode]
    #[auto_decode]
    struct Node {id: uint}

would generate two implementations like:

    impl<S: Encoder> node_id: Encodable<S> {
        fn encode(s: &S) {
            do s.emit_struct("Node", 1) {
                s.emit_field("id", 0, || s.emit_uint(self))
            }
        }
    }

    impl<D: Decoder> node_id: Decodable {
        static fn decode(d: &D) -> Node {
            do d.read_struct("Node", 1) {
                Node {
                    id: d.read_field(~"x", 0, || decode(d))
                }
            }
        }
    }

Other interesting scenarios are whe the item has type parameters or
references other non-built-in types.  A type definition like:

    #[auto_encode]
    #[auto_decode]
    type spanned<T> = {node: T, span: span};

would yield functions like:

    impl<
        S: Encoder,
        T: Encodable<S>
    > spanned<T>: Encodable<S> {
        fn encode<S: Encoder>(s: &S) {
            do s.emit_rec {
                s.emit_field("node", 0, || self.node.encode(s));
                s.emit_field("span", 1, || self.span.encode(s));
            }
        }
    }

    impl<
        D: Decoder,
        T: Decodable<D>
    > spanned<T>: Decodable<D> {
        static fn decode(d: &D) -> spanned<T> {
            do d.read_rec {
                {
                    node: d.read_field(~"node", 0, || decode(d)),
                    span: d.read_field(~"span", 1, || decode(d)),
                }
            }
        }
    }

FIXME (#2810)--Hygiene. Search for "__" strings.  We also assume "std" is the
standard library.

Misc notes:
-----------

I use move mode arguments for ast nodes that will get inserted as is
into the tree.  This is intended to prevent us from inserting the same
node twice.

*/

use core::prelude::*;

use ast;
use ast_util;
use attr;
use codemap::span;
use ext::base::*;
use parse;

use core::vec;
use std::map;
use std::map::HashMap;

export expand_auto_encode;
export expand_auto_decode;

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    pub use ext;
    pub use parse;
}

fn expand_auto_encode(
    cx: ext_ctxt,
    span: span,
    _mitem: ast::meta_item,
    in_items: ~[@ast::item]
) -> ~[@ast::item] {
    fn is_auto_encode(a: &ast::attribute) -> bool {
        attr::get_attr_name(*a) == ~"auto_encode"
    }

    fn filter_attrs(item: @ast::item) -> @ast::item {
        @ast::item {
            attrs: item.attrs.filtered(|a| !is_auto_encode(a)),
            .. *item
        }
    }

    do vec::flat_map(in_items) |item| {
        if item.attrs.any(is_auto_encode) {
            match item.node {
                ast::item_ty(@{node: ast::ty_rec(ref fields), _}, tps) => {
                    let ser_impl = mk_rec_ser_impl(
                        cx,
                        item.span,
                        item.ident,
                        (*fields),
                        tps
                    );

                    ~[filter_attrs(*item), ser_impl]
                },
                ast::item_struct(@ast::struct_def { fields, _}, tps) => {
                    let ser_impl = mk_struct_ser_impl(
                        cx,
                        item.span,
                        item.ident,
                        fields,
                        tps
                    );

                    ~[filter_attrs(*item), ser_impl]
                },
                ast::item_enum(ref enum_def, tps) => {
                    let ser_impl = mk_enum_ser_impl(
                        cx,
                        item.span,
                        item.ident,
                        (*enum_def),
                        tps
                    );

                    ~[filter_attrs(*item), ser_impl]
                },
                _ => {
                    cx.span_err(span, ~"#[auto_encode] can only be \
                                        applied to structs, record types, \
                                        and enum definitions");
                    ~[*item]
                }
            }
        } else {
            ~[*item]
        }
    }
}

fn expand_auto_decode(
    cx: ext_ctxt,
    span: span,
    _mitem: ast::meta_item,
    in_items: ~[@ast::item]
) -> ~[@ast::item] {
    fn is_auto_decode(a: &ast::attribute) -> bool {
        attr::get_attr_name(*a) == ~"auto_decode"
    }

    fn filter_attrs(item: @ast::item) -> @ast::item {
        @ast::item {
            attrs: item.attrs.filtered(|a| !is_auto_decode(a)),
            .. *item
        }
    }

    do vec::flat_map(in_items) |item| {
        if item.attrs.any(is_auto_decode) {
            match item.node {
                ast::item_ty(@{node: ast::ty_rec(ref fields), _}, tps) => {
                    let deser_impl = mk_rec_deser_impl(
                        cx,
                        item.span,
                        item.ident,
                        (*fields),
                        tps
                    );

                    ~[filter_attrs(*item), deser_impl]
                },
                ast::item_struct(@ast::struct_def { fields, _}, tps) => {
                    let deser_impl = mk_struct_deser_impl(
                        cx,
                        item.span,
                        item.ident,
                        fields,
                        tps
                    );

                    ~[filter_attrs(*item), deser_impl]
                },
                ast::item_enum(ref enum_def, tps) => {
                    let deser_impl = mk_enum_deser_impl(
                        cx,
                        item.span,
                        item.ident,
                        (*enum_def),
                        tps
                    );

                    ~[filter_attrs(*item), deser_impl]
                },
                _ => {
                    cx.span_err(span, ~"#[auto_decode] can only be \
                                        applied to structs, record types, \
                                        and enum definitions");
                    ~[*item]
                }
            }
        } else {
            ~[*item]
        }
    }
}

priv impl ext_ctxt {
    fn bind_path(
        span: span,
        ident: ast::ident,
        path: @ast::path,
        bounds: @~[ast::ty_param_bound]
    ) -> ast::ty_param {
        let bound = ast::TraitTyParamBound(@{
            id: self.next_id(),
            node: ast::ty_path(path, self.next_id()),
            span: span,
        });

        ast::ty_param {
            ident: ident,
            id: self.next_id(),
            bounds: @vec::append(~[bound], *bounds)
        }
    }

    fn expr(span: span, node: ast::expr_) -> @ast::expr {
        @ast::expr {
            id: self.next_id(),
            callee_id: self.next_id(),
            node: node,
            span: span,
        }
    }

    fn path(span: span, strs: ~[ast::ident]) -> @ast::path {
        @ast::path {
            span: span,
            global: false,
            idents: strs,
            rp: None,
            types: ~[]
        }
    }

    fn path_global(span: span, strs: ~[ast::ident]) -> @ast::path {
        @ast::path {
            span: span,
            global: true,
            idents: strs,
            rp: None,
            types: ~[]
        }
    }

    fn path_tps(span: span, strs: ~[ast::ident],
                tps: ~[@ast::Ty]) -> @ast::path {
        @ast::path {
            span: span,
            global: false,
            idents: strs,
            rp: None,
            types: tps
        }
    }

    fn path_tps_global(span: span, strs: ~[ast::ident],
                       tps: ~[@ast::Ty]) -> @ast::path {
        @ast::path {
            span: span,
            global: true,
            idents: strs,
            rp: None,
            types: tps
        }
    }

    fn ty_path(span: span, strs: ~[ast::ident],
               tps: ~[@ast::Ty]) -> @ast::Ty {
        @{id: self.next_id(),
          node: ast::ty_path(self.path_tps(span, strs, tps), self.next_id()),
          span: span}
    }

    fn binder_pat(span: span, nm: ast::ident) -> @ast::pat {
        @ast::pat {
            id: self.next_id(),
            node: ast::pat_ident(
                ast::bind_by_ref(ast::m_imm),
                self.path(span, ~[nm]),
                None),
            span: span,
        }
    }

    fn stmt(expr: @ast::expr) -> @ast::stmt {
        @ast::spanned { node: ast::stmt_semi(expr, self.next_id()),
                       span: expr.span }
    }

    fn lit_str(span: span, s: @~str) -> @ast::expr {
        self.expr(
            span,
            ast::expr_vstore(
                self.expr(
                    span,
                    ast::expr_lit(
                        @ast::spanned { node: ast::lit_str(s),
                                        span: span})),
                ast::expr_vstore_uniq))
    }

    fn lit_uint(span: span, i: uint) -> @ast::expr {
        self.expr(
            span,
            ast::expr_lit(
                @ast::spanned { node: ast::lit_uint(i as u64, ast::ty_u),
                                span: span}))
    }

    fn lambda(blk: ast::blk) -> @ast::expr {
        let ext_cx = self;
        let blk_e = self.expr(blk.span, ast::expr_block(blk));
        quote_expr!( || $blk_e )
    }

    fn blk(span: span, stmts: ~[@ast::stmt]) -> ast::blk {
        ast::spanned {
            node: ast::blk_ {
                view_items: ~[],
                stmts: stmts,
                expr: None,
                id: self.next_id(),
                rules: ast::default_blk,
            },
            span: span,
        }
    }

    fn expr_blk(expr: @ast::expr) -> ast::blk {
        ast::spanned {
            node: ast::blk_ {
                view_items: ~[],
                stmts: ~[],
                expr: Some(expr),
                id: self.next_id(),
                rules: ast::default_blk,
            },
            span: expr.span,
        }
    }

    fn expr_path(span: span, strs: ~[ast::ident]) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path(span, strs)))
    }

    fn expr_path_global(span: span, strs: ~[ast::ident]) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path_global(span, strs)))
    }

    fn expr_var(span: span, var: ~str) -> @ast::expr {
        self.expr_path(span, ~[self.ident_of(var)])
    }

    fn expr_field(
        span: span,
        expr: @ast::expr,
        ident: ast::ident
    ) -> @ast::expr {
        self.expr(span, ast::expr_field(expr, ident, ~[]))
    }

    fn expr_call(
        span: span,
        expr: @ast::expr,
        args: ~[@ast::expr]
    ) -> @ast::expr {
        self.expr(span, ast::expr_call(expr, args, false))
    }

    fn lambda_expr(expr: @ast::expr) -> @ast::expr {
        self.lambda(self.expr_blk(expr))
    }

    fn lambda_stmts(span: span, stmts: ~[@ast::stmt]) -> @ast::expr {
        self.lambda(self.blk(span, stmts))
    }
}

fn mk_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    ty_param: ast::ty_param,
    path: @ast::path,
    tps: ~[ast::ty_param],
    f: fn(@ast::Ty) -> @ast::method
) -> @ast::item {
    // All the type parameters need to bound to the trait.
    let mut trait_tps = vec::append(
        ~[ty_param],
         do tps.map |tp| {
            let t_bound = ast::TraitTyParamBound(@{
                id: cx.next_id(),
                node: ast::ty_path(path, cx.next_id()),
                span: span,
            });

            ast::ty_param {
                ident: tp.ident,
                id: cx.next_id(),
                bounds: @vec::append(~[t_bound], *tp.bounds)
            }
        }
    );

    let opt_trait = Some(@{
        path: path,
        ref_id: cx.next_id(),
    });

    let ty = cx.ty_path(
        span,
        ~[ident],
        tps.map(|tp| cx.ty_path(span, ~[tp.ident], ~[]))
    );

    @ast::item {
        // This is a new-style impl declaration.
        // XXX: clownshoes
        ident: parse::token::special_idents::clownshoes_extensions,
        attrs: ~[],
        id: cx.next_id(),
        node: ast::item_impl(trait_tps, opt_trait, ty, ~[f(ty)]),
        vis: ast::public,
        span: span,
    }
}

fn mk_ser_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    tps: ~[ast::ty_param],
    body: @ast::expr
) -> @ast::item {
    // Make a path to the std::serialize::Encodable typaram.
    let ty_param = cx.bind_path(
        span,
        cx.ident_of(~"__S"),
        cx.path_global(
            span,
            ~[
                cx.ident_of(~"std"),
                cx.ident_of(~"serialize"),
                cx.ident_of(~"Encoder"),
            ]
        ),
        @~[]
    );

    // Make a path to the std::serialize::Encodable trait.
    let path = cx.path_tps_global(
        span,
        ~[
            cx.ident_of(~"std"),
            cx.ident_of(~"serialize"),
            cx.ident_of(~"Encodable"),
        ],
        ~[cx.ty_path(span, ~[cx.ident_of(~"__S")], ~[])]
    );

    mk_impl(
        cx,
        span,
        ident,
        ty_param,
        path,
        tps,
        |_ty| mk_ser_method(cx, span, cx.expr_blk(body))
    )
}

fn mk_deser_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    tps: ~[ast::ty_param],
    body: @ast::expr
) -> @ast::item {
    // Make a path to the std::serialize::Decodable typaram.
    let ty_param = cx.bind_path(
        span,
        cx.ident_of(~"__D"),
        cx.path_global(
            span,
            ~[
                cx.ident_of(~"std"),
                cx.ident_of(~"serialize"),
                cx.ident_of(~"Decoder"),
            ]
        ),
        @~[]
    );

    // Make a path to the std::serialize::Decodable trait.
    let path = cx.path_tps_global(
        span,
        ~[
            cx.ident_of(~"std"),
            cx.ident_of(~"serialize"),
            cx.ident_of(~"Decodable"),
        ],
        ~[cx.ty_path(span, ~[cx.ident_of(~"__D")], ~[])]
    );

    mk_impl(
        cx,
        span,
        ident,
        ty_param,
        path,
        tps,
        |ty| mk_deser_method(cx, span, ty, cx.expr_blk(body))
    )
}

fn mk_ser_method(
    cx: ext_ctxt,
    span: span,
    ser_body: ast::blk
) -> @ast::method {
    let ty_s = @{
        id: cx.next_id(),
        node: ast::ty_rptr(
            @{
                id: cx.next_id(),
                node: ast::re_anon,
            },
            ast::mt {
                ty: cx.ty_path(span, ~[cx.ident_of(~"__S")], ~[]),
                mutbl: ast::m_imm
            }
        ),
        span: span,
    };

    let ser_inputs = ~[{
        mode: ast::infer(cx.next_id()),
        ty: ty_s,
        pat: @ast::pat {
            id: cx.next_id(),
            node: ast::pat_ident(
                ast::bind_by_value,
                ast_util::ident_to_path(span, cx.ident_of(~"__s")),
                None),
            span: span,
        },
        id: cx.next_id(),
    }];

    let ser_output = @{
        id: cx.next_id(),
        node: ast::ty_nil,
        span: span,
    };

    let ser_decl = {
        inputs: ser_inputs,
        output: ser_output,
        cf: ast::return_val,
    };

    @{
        ident: cx.ident_of(~"encode"),
        attrs: ~[],
        tps: ~[],
        self_ty: ast::spanned { node: ast::sty_region(ast::m_imm),
                                span: span },
        purity: ast::impure_fn,
        decl: ser_decl,
        body: ser_body,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: ast::public,
    }
}

fn mk_deser_method(
    cx: ext_ctxt,
    span: span,
    ty: @ast::Ty,
    deser_body: ast::blk
) -> @ast::method {
    let ty_d = @{
        id: cx.next_id(),
        node: ast::ty_rptr(
            @{
                id: cx.next_id(),
                node: ast::re_anon,
            },
            ast::mt {
                ty: cx.ty_path(span, ~[cx.ident_of(~"__D")], ~[]),
                mutbl: ast::m_imm
            }
        ),
        span: span,
    };

    let deser_inputs = ~[{
        mode: ast::infer(cx.next_id()),
        ty: ty_d,
        pat: @ast::pat {
            id: cx.next_id(),
            node: ast::pat_ident(
                ast::bind_by_value,
                ast_util::ident_to_path(span, cx.ident_of(~"__d")),
                None),
            span: span,
        },
        id: cx.next_id(),
    }];

    let deser_decl = {
        inputs: deser_inputs,
        output: ty,
        cf: ast::return_val,
    };

    @{
        ident: cx.ident_of(~"decode"),
        attrs: ~[],
        tps: ~[],
        self_ty: ast::spanned { node: ast::sty_static,
                                span: span },
        purity: ast::impure_fn,
        decl: deser_decl,
        body: deser_body,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: ast::public,
    }
}

fn mk_rec_ser_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    fields: ~[ast::ty_field],
    tps: ~[ast::ty_param]
) -> @ast::item {
    let fields = mk_ser_fields(cx, span, mk_rec_fields(fields));

    // ast for `__s.emit_rec(|| $(fields))`
    let body = cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__s"),
            cx.ident_of(~"emit_rec")
        ),
        ~[cx.lambda_stmts(span, fields)]
    );

    mk_ser_impl(cx, span, ident, tps, body)
}

fn mk_rec_deser_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    fields: ~[ast::ty_field],
    tps: ~[ast::ty_param]
) -> @ast::item {
    let fields = mk_deser_fields(cx, span, mk_rec_fields(fields));

    // ast for `read_rec(|| $(fields))`
    let body = cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__d"),
            cx.ident_of(~"read_rec")
        ),
        ~[
            cx.lambda_expr(
                cx.expr(
                    span,
                    ast::expr_rec(fields, None)
                )
            )
        ]
    );

    mk_deser_impl(cx, span, ident, tps, body)
}

fn mk_struct_ser_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    fields: ~[@ast::struct_field],
    tps: ~[ast::ty_param]
) -> @ast::item {
    let fields = mk_ser_fields(cx, span, mk_struct_fields(fields));

    // ast for `__s.emit_struct($(name), || $(fields))`
    let ser_body = cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__s"),
            cx.ident_of(~"emit_struct")
        ),
        ~[
            cx.lit_str(span, @cx.str_of(ident)),
            cx.lit_uint(span, vec::len(fields)),
            cx.lambda_stmts(span, fields),
        ]
    );

    mk_ser_impl(cx, span, ident, tps, ser_body)
}

fn mk_struct_deser_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    fields: ~[@ast::struct_field],
    tps: ~[ast::ty_param]
) -> @ast::item {
    let fields = mk_deser_fields(cx, span, mk_struct_fields(fields));

    // ast for `read_struct($(name), || $(fields))`
    let body = cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__d"),
            cx.ident_of(~"read_struct")
        ),
        ~[
            cx.lit_str(span, @cx.str_of(ident)),
            cx.lit_uint(span, vec::len(fields)),
            cx.lambda_expr(
                cx.expr(
                    span,
                    ast::expr_struct(
                        cx.path(span, ~[ident]),
                        fields,
                        None
                    )
                )
            ),
        ]
    );

    mk_deser_impl(cx, span, ident, tps, body)
}

// Records and structs don't have the same fields types, but they share enough
// that if we extract the right subfields out we can share the code
// generator code.
type field = { span: span, ident: ast::ident, mutbl: ast::mutability };

fn mk_rec_fields(fields: ~[ast::ty_field]) -> ~[field] {
    do fields.map |field| {
        {
            span: field.span,
            ident: field.node.ident,
            mutbl: field.node.mt.mutbl,
        }
    }
}

fn mk_struct_fields(fields: ~[@ast::struct_field]) -> ~[field] {
    do fields.map |field| {
        let (ident, mutbl) = match field.node.kind {
            ast::named_field(ident, mutbl, _) => (ident, mutbl),
            _ => fail ~"[auto_encode] does not support \
                        unnamed fields",
        };

        {
            span: field.span,
            ident: ident,
            mutbl: match mutbl {
                ast::struct_mutable => ast::m_mutbl,
                ast::struct_immutable => ast::m_imm,
            },
        }
    }
}

fn mk_ser_fields(
    cx: ext_ctxt,
    span: span,
    fields: ~[field]
) -> ~[@ast::stmt] {
    do fields.mapi |idx, field| {
        // ast for `|| self.$(name).encode(__s)`
        let expr_lambda = cx.lambda_expr(
            cx.expr_call(
                span,
                cx.expr_field(
                    span,
                    cx.expr_field(
                        span,
                        cx.expr_var(span, ~"self"),
                        field.ident
                    ),
                    cx.ident_of(~"encode")
                ),
                ~[cx.expr_var(span, ~"__s")]
            )
        );

        // ast for `__s.emit_field($(name), $(idx), $(expr_lambda))`
        cx.stmt(
            cx.expr_call(
                span,
                cx.expr_field(
                    span,
                    cx.expr_var(span, ~"__s"),
                    cx.ident_of(~"emit_field")
                ),
                ~[
                    cx.lit_str(span, @cx.str_of(field.ident)),
                    cx.lit_uint(span, idx),
                    expr_lambda,
                ]
            )
        )
    }
}

fn mk_deser_fields(
    cx: ext_ctxt,
    span: span,
    fields: ~[{ span: span, ident: ast::ident, mutbl: ast::mutability }]
) -> ~[ast::field] {
    do fields.mapi |idx, field| {
        // ast for `|| std::serialize::decode(__d)`
        let expr_lambda = cx.lambda(
            cx.expr_blk(
                cx.expr_call(
                    span,
                    cx.expr_path_global(span, ~[
                        cx.ident_of(~"std"),
                        cx.ident_of(~"serialize"),
                        cx.ident_of(~"Decodable"),
                        cx.ident_of(~"decode"),
                    ]),
                    ~[cx.expr_var(span, ~"__d")]
                )
            )
        );

        // ast for `__d.read_field($(name), $(idx), $(expr_lambda))`
        let expr: @ast::expr = cx.expr_call(
            span,
            cx.expr_field(
                span,
                cx.expr_var(span, ~"__d"),
                cx.ident_of(~"read_field")
            ),
            ~[
                cx.lit_str(span, @cx.str_of(field.ident)),
                cx.lit_uint(span, idx),
                expr_lambda,
            ]
        );

        ast::spanned {
            node: ast::field_ {
                mutbl: field.mutbl,
                ident: field.ident,
                expr: expr,
            },
            span: span,
        }
    }
}

fn mk_enum_ser_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    enum_def: ast::enum_def,
    tps: ~[ast::ty_param]
) -> @ast::item {
    let body = mk_enum_ser_body(
        cx,
        span,
        ident,
        enum_def.variants
    );

    mk_ser_impl(cx, span, ident, tps, body)
}

fn mk_enum_deser_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    enum_def: ast::enum_def,
    tps: ~[ast::ty_param]
) -> @ast::item {
    let body = mk_enum_deser_body(
        cx,
        span,
        ident,
        enum_def.variants
    );

    mk_deser_impl(cx, span, ident, tps, body)
}

fn ser_variant(
    cx: ext_ctxt,
    span: span,
    v_name: ast::ident,
    v_idx: uint,
    args: ~[ast::variant_arg]
) -> ast::arm {
    // Name the variant arguments.
    let names = args.mapi(|i, _arg| cx.ident_of(fmt!("__v%u", i)));

    // Bind the names to the variant argument type.
    let pats = args.mapi(|i, arg| cx.binder_pat(arg.ty.span, names[i]));

    let pat_node = if pats.is_empty() {
        ast::pat_ident(
            ast::bind_by_ref(ast::m_imm),
            cx.path(span, ~[v_name]),
            None
        )
    } else {
        ast::pat_enum(
            cx.path(span, ~[v_name]),
            Some(pats)
        )
    };

    let pat = @ast::pat {
        id: cx.next_id(),
        node: pat_node,
        span: span,
    };

    let stmts = do args.mapi |a_idx, _arg| {
        // ast for `__s.emit_enum_variant_arg`
        let expr_emit = cx.expr_field(
            span,
            cx.expr_var(span, ~"__s"),
            cx.ident_of(~"emit_enum_variant_arg")
        );

        // ast for `|| $(v).encode(__s)`
        let expr_encode = cx.lambda_expr(
             cx.expr_call(
                span,
                cx.expr_field(
                    span,
                    cx.expr_path(span, ~[names[a_idx]]),
                    cx.ident_of(~"encode")
                ),
                ~[cx.expr_var(span, ~"__s")]
            )
        );

        // ast for `$(expr_emit)($(a_idx), $(expr_encode))`
        cx.stmt(
            cx.expr_call(
                span,
                expr_emit,
                ~[cx.lit_uint(span, a_idx), expr_encode]
            )
        )
    };

    // ast for `__s.emit_enum_variant($(name), $(idx), $(sz), $(lambda))`
    let body = cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__s"),
            cx.ident_of(~"emit_enum_variant")
        ),
        ~[
            cx.lit_str(span, @cx.str_of(v_name)),
            cx.lit_uint(span, v_idx),
            cx.lit_uint(span, stmts.len()),
            cx.lambda_stmts(span, stmts),
        ]
    );

    ast::arm { pats: ~[pat], guard: None, body: cx.expr_blk(body) }
}

fn mk_enum_ser_body(
    cx: ext_ctxt,
    span: span,
    name: ast::ident,
    variants: ~[ast::variant]
) -> @ast::expr {
    let arms = do variants.mapi |v_idx, variant| {
        match variant.node.kind {
            ast::tuple_variant_kind(args) =>
                ser_variant(cx, span, variant.node.name, v_idx, args),
            ast::struct_variant_kind(*) =>
                fail ~"struct variants unimplemented",
            ast::enum_variant_kind(*) =>
                fail ~"enum variants unimplemented",
        }
    };

    // ast for `match *self { $(arms) }`
    let match_expr = cx.expr(
        span,
        ast::expr_match(
            cx.expr(
                span,
                ast::expr_unary(ast::deref, cx.expr_var(span, ~"self"))
            ),
            arms
        )
    );

    // ast for `__s.emit_enum($(name), || $(match_expr))`
    cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__s"),
            cx.ident_of(~"emit_enum")
        ),
        ~[
            cx.lit_str(span, @cx.str_of(name)),
            cx.lambda_expr(match_expr),
        ]
    )
}

fn mk_enum_deser_variant_nary(
    cx: ext_ctxt,
    span: span,
    name: ast::ident,
    args: ~[ast::variant_arg]
) -> @ast::expr {
    let args = do args.mapi |idx, _arg| {
        // ast for `|| std::serialize::decode(__d)`
        let expr_lambda = cx.lambda_expr(
            cx.expr_call(
                span,
                cx.expr_path_global(span, ~[
                    cx.ident_of(~"std"),
                    cx.ident_of(~"serialize"),
                    cx.ident_of(~"Decodable"),
                    cx.ident_of(~"decode"),
                ]),
                ~[cx.expr_var(span, ~"__d")]
            )
        );

        // ast for `__d.read_enum_variant_arg($(a_idx), $(expr_lambda))`
        cx.expr_call(
            span,
            cx.expr_field(
                span,
                cx.expr_var(span, ~"__d"),
                cx.ident_of(~"read_enum_variant_arg")
            ),
            ~[cx.lit_uint(span, idx), expr_lambda]
        )
    };

    // ast for `$(name)($(args))`
    cx.expr_call(span, cx.expr_path(span, ~[name]), args)
}

fn mk_enum_deser_body(
    cx: ext_ctxt,
    span: span,
    name: ast::ident,
    variants: ~[ast::variant]
) -> @ast::expr {
    let mut arms = do variants.mapi |v_idx, variant| {
        let body = match variant.node.kind {
            ast::tuple_variant_kind(args) => {
                if args.is_empty() {
                    // for a nullary variant v, do "v"
                    cx.expr_path(span, ~[variant.node.name])
                } else {
                    // for an n-ary variant v, do "v(a_1, ..., a_n)"
                    mk_enum_deser_variant_nary(
                        cx,
                        span,
                        variant.node.name,
                        args
                    )
                }
            },
            ast::struct_variant_kind(*) =>
                fail ~"struct variants unimplemented",
            ast::enum_variant_kind(*) =>
                fail ~"enum variants unimplemented",
        };

        let pat = @ast::pat {
            id: cx.next_id(),
            node: ast::pat_lit(cx.lit_uint(span, v_idx)),
            span: span,
        };

        ast::arm {
            pats: ~[pat],
            guard: None,
            body: cx.expr_blk(body),
        }
    };

    let impossible_case = ast::arm {
        pats: ~[@ast::pat {
            id: cx.next_id(),
            node: ast::pat_wild,
            span: span,
        }],
        guard: None,

        // FIXME(#3198): proper error message
        body: cx.expr_blk(cx.expr(span, ast::expr_fail(None))),
    };

    arms.push(impossible_case);

    // ast for `|i| { match i { $(arms) } }`
    let expr_lambda = cx.expr(
        span,
        ast::expr_fn_block(
            {
                inputs: ~[{
                    mode: ast::infer(cx.next_id()),
                    ty: @{
                        id: cx.next_id(),
                        node: ast::ty_infer,
                        span: span
                    },
                    pat: @ast::pat {
                        id: cx.next_id(),
                        node: ast::pat_ident(
                            ast::bind_by_value,
                            ast_util::ident_to_path(span, cx.ident_of(~"i")),
                            None),
                        span: span,
                    },
                    id: cx.next_id(),
                }],
                output: @{
                    id: cx.next_id(),
                    node: ast::ty_infer,
                    span: span,
                },
                cf: ast::return_val,
            },
            cx.expr_blk(
                cx.expr(
                    span,
                    ast::expr_match(cx.expr_var(span, ~"i"), arms)
                )
            ),
            @~[]
        )
    );

    // ast for `__d.read_enum_variant($(expr_lambda))`
    let expr_lambda = cx.lambda_expr(
        cx.expr_call(
            span,
            cx.expr_field(
                span,
                cx.expr_var(span, ~"__d"),
                cx.ident_of(~"read_enum_variant")
            ),
            ~[expr_lambda]
        )
    );

    // ast for `__d.read_enum($(e_name), $(expr_lambda))`
    cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__d"),
            cx.ident_of(~"read_enum")
        ),
        ~[
            cx.lit_str(span, @cx.str_of(name)),
            expr_lambda
        ]
    )
}
