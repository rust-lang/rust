// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The compiler code necessary to implement the #[auto_encode] and
#[auto_decode] extension.  The idea here is that type-defining items may
be tagged with #[auto_encode] and #[auto_decode], which will cause
us to generate a little companion module with the same name as the item.

For example, a type like:

    #[auto_encode]
    #[auto_decode]
    struct Node {id: uint}

would generate two implementations like:

impl<S:std::serialize::Encoder> Encodable<S> for Node {
    fn encode(&self, s: &S) {
        do s.emit_struct("Node", 1) {
            s.emit_field("id", 0, || s.emit_uint(self.id))
        }
    }
}

impl<D:Decoder> Decodable for node_id {
    fn decode(d: &D) -> Node {
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
    struct spanned<T> {node: T, span: span}

would yield functions like:

    impl<
        S: Encoder,
        T: Encodable<S>
    > spanned<T>: Encodable<S> {
        fn encode<S:Encoder>(s: &S) {
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
        fn decode(d: &D) -> spanned<T> {
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
use codemap;
use codemap::span;
use ext::base::*;
use parse;
use opt_vec;
use opt_vec::OptVec;
use ext::build;

use core::vec;

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    pub use ext;
    pub use parse;
}

pub fn expand_auto_encode(
    cx: @ext_ctxt,
    span: span,
    _mitem: @ast::meta_item,
    in_items: ~[@ast::item]
) -> ~[@ast::item] {
    fn is_auto_encode(a: &ast::attribute) -> bool {
        *attr::get_attr_name(a) == ~"auto_encode"
    }

    fn filter_attrs(item: @ast::item) -> @ast::item {
        @ast::item {
            attrs: item.attrs.filtered(|a| !is_auto_encode(a)),
            .. copy *item
        }
    }

    do vec::flat_map(in_items) |item| {
        if item.attrs.any(is_auto_encode) {
            match item.node {
                ast::item_struct(ref struct_def, ref generics) => {
                    let ser_impl = mk_struct_ser_impl(
                        cx,
                        item.span,
                        item.ident,
                        struct_def.fields,
                        generics
                    );

                    ~[filter_attrs(*item), ser_impl]
                },
                ast::item_enum(ref enum_def, ref generics) => {
                    let ser_impl = mk_enum_ser_impl(
                        cx,
                        item.span,
                        item.ident,
                        copy *enum_def,
                        generics
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

pub fn expand_auto_decode(
    cx: @ext_ctxt,
    span: span,
    _mitem: @ast::meta_item,
    in_items: ~[@ast::item]
) -> ~[@ast::item] {
    fn is_auto_decode(a: &ast::attribute) -> bool {
        *attr::get_attr_name(a) == ~"auto_decode"
    }

    fn filter_attrs(item: @ast::item) -> @ast::item {
        @ast::item {
            attrs: item.attrs.filtered(|a| !is_auto_decode(a)),
            .. copy *item
        }
    }

    do vec::flat_map(in_items) |item| {
        if item.attrs.any(is_auto_decode) {
            match item.node {
                ast::item_struct(ref struct_def, ref generics) => {
                    let deser_impl = mk_struct_deser_impl(
                        cx,
                        item.span,
                        item.ident,
                        struct_def.fields,
                        generics
                    );

                    ~[filter_attrs(*item), deser_impl]
                },
                ast::item_enum(ref enum_def, ref generics) => {
                    let deser_impl = mk_enum_deser_impl(
                        cx,
                        item.span,
                        item.ident,
                        copy *enum_def,
                        generics
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

priv impl @ext_ctxt {
    fn bind_path(
        &self,
        span: span,
        ident: ast::ident,
        path: @ast::path,
        bounds: @OptVec<ast::TyParamBound>
    ) -> ast::TyParam {
        let bound = ast::TraitTyParamBound(@ast::Ty {
            id: self.next_id(),
            node: ast::ty_path(path, self.next_id()),
            span: span,
        });

        ast::TyParam {
            ident: ident,
            id: self.next_id(),
            bounds: @bounds.prepend(bound)
        }
    }

    fn expr(&self, span: span, +node: ast::expr_) -> @ast::expr {
        @ast::expr {
            id: self.next_id(),
            callee_id: self.next_id(),
            node: node,
            span: span,
        }
    }

    fn path(&self, span: span, +strs: ~[ast::ident]) -> @ast::path {
        @ast::path {
            span: span,
            global: false,
            idents: strs,
            rp: None,
            types: ~[]
        }
    }

    fn path_global(&self, span: span, +strs: ~[ast::ident]) -> @ast::path {
        @ast::path {
            span: span,
            global: true,
            idents: strs,
            rp: None,
            types: ~[]
        }
    }

    fn path_tps(
        &self,
        span: span,
        +strs: ~[ast::ident],
        +tps: ~[@ast::Ty]
    ) -> @ast::path {
        @ast::path {
            span: span,
            global: false,
            idents: strs,
            rp: None,
            types: tps
        }
    }

    fn path_tps_global(
        &self,
        span: span,
        +strs: ~[ast::ident],
        +tps: ~[@ast::Ty]
    ) -> @ast::path {
        @ast::path {
            span: span,
            global: true,
            idents: strs,
            rp: None,
            types: tps
        }
    }

    fn ty_path(
        &self,
        span: span,
        +strs: ~[ast::ident],
        +tps: ~[@ast::Ty]
    ) -> @ast::Ty {
        @ast::Ty {
            id: self.next_id(),
            node: ast::ty_path(
                self.path_tps(span, strs, tps),
                self.next_id()),
            span: span,
        }
    }

    fn binder_pat(&self, span: span, nm: ast::ident) -> @ast::pat {
        @ast::pat {
            id: self.next_id(),
            node: ast::pat_ident(
                ast::bind_by_ref(ast::m_imm),
                self.path(span, ~[nm]),
                None),
            span: span,
        }
    }

    fn stmt(&self, expr: @ast::expr) -> @ast::stmt {
        @codemap::spanned { node: ast::stmt_semi(expr, self.next_id()),
                       span: expr.span }
    }

    fn lit_str(&self, span: span, s: @~str) -> @ast::expr {
        self.expr(
            span,
            ast::expr_vstore(
                self.expr(
                    span,
                    ast::expr_lit(
                        @codemap::spanned { node: ast::lit_str(s),
                                        span: span})),
                ast::expr_vstore_uniq))
    }

    fn lit_uint(&self, span: span, i: uint) -> @ast::expr {
        self.expr(
            span,
            ast::expr_lit(
                @codemap::spanned { node: ast::lit_uint(i as u64, ast::ty_u),
                                span: span}))
    }

    fn lambda(&self, +blk: ast::blk) -> @ast::expr {
        let ext_cx = *self;
        let blk_e = self.expr(copy blk.span, ast::expr_block(copy blk));
        quote_expr!( || $blk_e )
    }

    fn blk(&self, span: span, +stmts: ~[@ast::stmt]) -> ast::blk {
        codemap::spanned {
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

    fn expr_blk(&self, expr: @ast::expr) -> ast::blk {
        codemap::spanned {
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

    fn expr_path(&self, span: span, +strs: ~[ast::ident]) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path(span, strs)))
    }

    fn expr_path_global(
        &self,
        span: span,
        +strs: ~[ast::ident]
    ) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path_global(span, strs)))
    }

    fn expr_var(&self, span: span, +var: ~str) -> @ast::expr {
        self.expr_path(span, ~[self.ident_of(var)])
    }

    fn expr_field(
        &self,
        span: span,
        expr: @ast::expr,
        ident: ast::ident
    ) -> @ast::expr {
        self.expr(span, ast::expr_field(expr, ident, ~[]))
    }

    fn expr_call(
        &self,
        span: span,
        expr: @ast::expr,
        +args: ~[@ast::expr]
    ) -> @ast::expr {
        self.expr(span, ast::expr_call(expr, args, ast::NoSugar))
    }

    fn expr_method_call(
        &self,
        span: span,
        expr: @ast::expr,
        ident: ast::ident,
        +args: ~[@ast::expr]
    ) -> @ast::expr {
        self.expr(span, ast::expr_method_call(expr, ident, ~[], args, ast::NoSugar))
    }

    fn lambda_expr(&self, expr: @ast::expr) -> @ast::expr {
        self.lambda(self.expr_blk(expr))
    }

    fn lambda_stmts(&self, span: span, +stmts: ~[@ast::stmt]) -> @ast::expr {
        self.lambda(self.blk(span, stmts))
    }
}

fn mk_impl(
    cx: @ext_ctxt,
    span: span,
    ident: ast::ident,
    ty_param: ast::TyParam,
    path: @ast::path,
    generics: &ast::Generics,
    f: &fn(@ast::Ty) -> @ast::method
) -> @ast::item {
    /*!
     *
     * Given that we are deriving auto-encode a type `T<'a, ...,
     * 'z, A, ..., Z>`, creates an impl like:
     *
     *      impl<'a, ..., 'z, A:Tr, ..., Z:Tr> Tr for T<A, ..., Z> { ... }
     *
     * where Tr is either Serializable and Deserialize.
     *
     * FIXME(#5090): Remove code duplication between this and the code
     * in deriving.rs
     */


    // Copy the lifetimes
    let impl_lifetimes = generics.lifetimes.map(|l| {
        build::mk_lifetime(cx, l.span, l.ident)
    });

    // All the type parameters need to bound to the trait.
    let mut impl_tps = opt_vec::with(ty_param);
    for generics.ty_params.each |tp| {
        let t_bound = ast::TraitTyParamBound(@ast::Ty {
            id: cx.next_id(),
            node: ast::ty_path(path, cx.next_id()),
            span: span,
        });

        impl_tps.push(ast::TyParam {
            ident: tp.ident,
            id: cx.next_id(),
            bounds: @tp.bounds.prepend(t_bound)
        })
    }

    let opt_trait = Some(@ast::trait_ref {
        path: path,
        ref_id: cx.next_id(),
    });

    let ty = cx.ty_path(
        span,
        ~[ident],
        opt_vec::take_vec(generics.ty_params.map(
            |tp| cx.ty_path(span, ~[tp.ident], ~[])))
    );

    let generics = ast::Generics {
        lifetimes: impl_lifetimes,
        ty_params: impl_tps
    };

    @ast::item {
        // This is a new-style impl declaration.
        // XXX: clownshoes
        ident: parse::token::special_idents::clownshoes_extensions,
        attrs: ~[],
        id: cx.next_id(),
        node: ast::item_impl(generics, opt_trait, ty, ~[f(ty)]),
        vis: ast::public,
        span: span,
    }
}

fn mk_ser_impl(
    cx: @ext_ctxt,
    span: span,
    ident: ast::ident,
    generics: &ast::Generics,
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
        @opt_vec::Empty
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
        generics,
        |_ty| mk_ser_method(cx, span, cx.expr_blk(body))
    )
}

fn mk_deser_impl(
    cx: @ext_ctxt,
    span: span,
    ident: ast::ident,
    generics: &ast::Generics,
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
        @opt_vec::Empty
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
        generics,
        |ty| mk_deser_method(cx, span, ty, cx.expr_blk(body))
    )
}

fn mk_ser_method(
    cx: @ext_ctxt,
    span: span,
    +ser_body: ast::blk
) -> @ast::method {
    let ty_s = @ast::Ty {
        id: cx.next_id(),
        node: ast::ty_rptr(
            None,
            ast::mt {
                ty: cx.ty_path(span, ~[cx.ident_of(~"__S")], ~[]),
                mutbl: ast::m_imm
            }
        ),
        span: span,
    };

    let ser_inputs = ~[ast::arg {
        mode: ast::infer(cx.next_id()),
        is_mutbl: false,
        ty: ty_s,
        pat: @ast::pat {
            id: cx.next_id(),
            node: ast::pat_ident(
                ast::bind_by_copy,
                ast_util::ident_to_path(span, cx.ident_of(~"__s")),
                None),
            span: span,
        },
        id: cx.next_id(),
    }];

    let ser_output = @ast::Ty {
        id: cx.next_id(),
        node: ast::ty_nil,
        span: span,
    };

    let ser_decl = ast::fn_decl {
        inputs: ser_inputs,
        output: ser_output,
        cf: ast::return_val,
    };

    @ast::method {
        ident: cx.ident_of(~"encode"),
        attrs: ~[],
        generics: ast_util::empty_generics(),
        self_ty: codemap::spanned {
            node: ast::sty_region(None, ast::m_imm),
            span: span
        },
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
    cx: @ext_ctxt,
    span: span,
    ty: @ast::Ty,
    +deser_body: ast::blk
) -> @ast::method {
    let ty_d = @ast::Ty {
        id: cx.next_id(),
        node: ast::ty_rptr(
            None,
            ast::mt {
                ty: cx.ty_path(span, ~[cx.ident_of(~"__D")], ~[]),
                mutbl: ast::m_imm
            }
        ),
        span: span,
    };

    let deser_inputs = ~[ast::arg {
        mode: ast::infer(cx.next_id()),
        is_mutbl: false,
        ty: ty_d,
        pat: @ast::pat {
            id: cx.next_id(),
            node: ast::pat_ident(
                ast::bind_by_copy,
                ast_util::ident_to_path(span, cx.ident_of(~"__d")),
                None),
            span: span,
        },
        id: cx.next_id(),
    }];

    let deser_decl = ast::fn_decl {
        inputs: deser_inputs,
        output: ty,
        cf: ast::return_val,
    };

    @ast::method {
        ident: cx.ident_of(~"decode"),
        attrs: ~[],
        generics: ast_util::empty_generics(),
        self_ty: codemap::spanned { node: ast::sty_static, span: span },
        purity: ast::impure_fn,
        decl: deser_decl,
        body: deser_body,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: ast::public,
    }
}

fn mk_struct_ser_impl(
    cx: @ext_ctxt,
    span: span,
    ident: ast::ident,
    fields: &[@ast::struct_field],
    generics: &ast::Generics
) -> @ast::item {
    let fields = do mk_struct_fields(fields).mapi |idx, field| {
        // ast for `|| self.$(name).encode(__s)`
        let expr_lambda = cx.lambda_expr(
            cx.expr_method_call(
                span,
                cx.expr_field(
                    span,
                    cx.expr_var(span, ~"self"),
                    field.ident
                ),
                cx.ident_of(~"encode"),
                ~[cx.expr_var(span, ~"__s")]
            )
        );

        // ast for `__s.emit_field($(name), $(idx), $(expr_lambda))`
        cx.stmt(
            cx.expr_method_call(
                span,
                cx.expr_var(span, ~"__s"),
                cx.ident_of(~"emit_field"),
                ~[
                    cx.lit_str(span, @cx.str_of(field.ident)),
                    cx.lit_uint(span, idx),
                    expr_lambda,
                ]
            )
        )
    };

    // ast for `__s.emit_struct($(name), || $(fields))`
    let ser_body = cx.expr_method_call(
        span,
        cx.expr_var(span, ~"__s"),
        cx.ident_of(~"emit_struct"),
        ~[
            cx.lit_str(span, @cx.str_of(ident)),
            cx.lit_uint(span, vec::len(fields)),
            cx.lambda_stmts(span, fields),
        ]
    );

    mk_ser_impl(cx, span, ident, generics, ser_body)
}

fn mk_struct_deser_impl(
    cx: @ext_ctxt,
    span: span,
    ident: ast::ident,
    fields: ~[@ast::struct_field],
    generics: &ast::Generics
) -> @ast::item {
    let fields = do mk_struct_fields(fields).mapi |idx, field| {
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
        let expr: @ast::expr = cx.expr_method_call(
            span,
            cx.expr_var(span, ~"__d"),
            cx.ident_of(~"read_field"),
            ~[
                cx.lit_str(span, @cx.str_of(field.ident)),
                cx.lit_uint(span, idx),
                expr_lambda,
            ]
        );

        codemap::spanned {
            node: ast::field_ {
                mutbl: field.mutbl,
                ident: field.ident,
                expr: expr,
            },
            span: span,
        }
    };

    // ast for `read_struct($(name), || $(fields))`
    let body = cx.expr_method_call(
        span,
        cx.expr_var(span, ~"__d"),
        cx.ident_of(~"read_struct"),
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

    mk_deser_impl(cx, span, ident, generics, body)
}

// Records and structs don't have the same fields types, but they share enough
// that if we extract the right subfields out we can share the code
// generator code.
struct field {
    span: span,
    ident: ast::ident,
    mutbl: ast::mutability,
}

fn mk_struct_fields(fields: &[@ast::struct_field]) -> ~[field] {
    do fields.map |field| {
        let (ident, mutbl) = match field.node.kind {
            ast::named_field(ident, mutbl, _) => (ident, mutbl),
            _ => fail!(~"[auto_encode] does not support \
                        unnamed fields")
        };

        field {
            span: field.span,
            ident: ident,
            mutbl: match mutbl {
                ast::struct_mutable => ast::m_mutbl,
                ast::struct_immutable => ast::m_imm,
            },
        }
    }
}

fn mk_enum_ser_impl(
    cx: @ext_ctxt,
    span: span,
    ident: ast::ident,
    +enum_def: ast::enum_def,
    generics: &ast::Generics
) -> @ast::item {
    let body = mk_enum_ser_body(
        cx,
        span,
        ident,
        copy enum_def.variants
    );

    mk_ser_impl(cx, span, ident, generics, body)
}

fn mk_enum_deser_impl(
    cx: @ext_ctxt,
    span: span,
    ident: ast::ident,
    +enum_def: ast::enum_def,
    generics: &ast::Generics
) -> @ast::item {
    let body = mk_enum_deser_body(
        cx,
        span,
        ident,
        enum_def.variants
    );

    mk_deser_impl(cx, span, ident, generics, body)
}

fn ser_variant(
    cx: @ext_ctxt,
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
            ast::bind_infer,
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
             cx.expr_method_call(
                span,
                 cx.expr_path(span, ~[names[a_idx]]),
                 cx.ident_of(~"encode"),
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
    let body = cx.expr_method_call(
        span,
        cx.expr_var(span, ~"__s"),
        cx.ident_of(~"emit_enum_variant"),
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
    cx: @ext_ctxt,
    span: span,
    name: ast::ident,
    +variants: ~[ast::variant]
) -> @ast::expr {
    let arms = do variants.mapi |v_idx, variant| {
        match variant.node.kind {
            ast::tuple_variant_kind(ref args) =>
                ser_variant(
                    cx,
                    span,
                    variant.node.name,
                    v_idx,
                    /*bad*/ copy *args
                ),
            ast::struct_variant_kind(*) =>
                fail!(~"struct variants unimplemented"),
            ast::enum_variant_kind(*) =>
                fail!(~"enum variants unimplemented"),
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
    cx.expr_method_call(
        span,
        cx.expr_var(span, ~"__s"),
        cx.ident_of(~"emit_enum"),
        ~[
            cx.lit_str(span, @cx.str_of(name)),
            cx.lambda_expr(match_expr),
        ]
    )
}

fn mk_enum_deser_variant_nary(
    cx: @ext_ctxt,
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
        cx.expr_method_call(
            span,
            cx.expr_var(span, ~"__d"),
            cx.ident_of(~"read_enum_variant_arg"),
            ~[cx.lit_uint(span, idx), expr_lambda]
        )
    };

    // ast for `$(name)($(args))`
    cx.expr_call(span, cx.expr_path(span, ~[name]), args)
}

fn mk_enum_deser_body(
    ext_cx: @ext_ctxt,
    span: span,
    name: ast::ident,
    variants: ~[ast::variant]
) -> @ast::expr {
    let expr_arm_names = build::mk_base_vec_e(
        ext_cx,
        span,
         do variants.map |variant| {
            build::mk_base_str(
                ext_cx,
                span,
                ext_cx.str_of(variant.node.name)
            )
        }
    );

    let mut arms = do variants.mapi |v_idx, variant| {
        let body = match variant.node.kind {
            ast::tuple_variant_kind(ref args) => {
                if args.is_empty() {
                    // for a nullary variant v, do "v"
                    ext_cx.expr_path(span, ~[variant.node.name])
                } else {
                    // for an n-ary variant v, do "v(a_1, ..., a_n)"
                    mk_enum_deser_variant_nary(
                        ext_cx,
                        span,
                        variant.node.name,
                        copy *args
                    )
                }
            },
            ast::struct_variant_kind(*) =>
                fail!(~"struct variants unimplemented"),
            ast::enum_variant_kind(*) =>
                fail!(~"enum variants unimplemented")
        };

        let pat = @ast::pat {
            id: ext_cx.next_id(),
            node: ast::pat_lit(ext_cx.lit_uint(span, v_idx)),
            span: span,
        };

        ast::arm {
            pats: ~[pat],
            guard: None,
            body: ext_cx.expr_blk(body),
        }
    };

    let quoted_expr = copy quote_expr!(
      ::core::sys::begin_unwind(~"explicit failure", ~"empty", 1);
    ).node;

    let impossible_case = ast::arm {
        pats: ~[@ast::pat {
            id: ext_cx.next_id(),
            node: ast::pat_wild,
            span: span,
        }],
        guard: None,

        // FIXME(#3198): proper error message
        body: ext_cx.expr_blk(ext_cx.expr(span, quoted_expr)),
    };

    arms.push(impossible_case);

    // ast for `|i| { match i { $(arms) } }`
    let expr_lambda = ext_cx.expr(
        span,
        ast::expr_fn_block(
            ast::fn_decl {
                inputs: ~[ast::arg {
                    mode: ast::infer(ext_cx.next_id()),
                    is_mutbl: false,
                    ty: @ast::Ty {
                        id: ext_cx.next_id(),
                        node: ast::ty_infer,
                        span: span
                    },
                    pat: @ast::pat {
                        id: ext_cx.next_id(),
                        node: ast::pat_ident(
                            ast::bind_by_copy,
                            ast_util::ident_to_path(span,
                                ext_cx.ident_of(~"i")),
                            None),
                        span: span,
                    },
                    id: ext_cx.next_id(),
                }],
                output: @ast::Ty {
                    id: ext_cx.next_id(),
                    node: ast::ty_infer,
                    span: span,
                },
                cf: ast::return_val,
            },
            ext_cx.expr_blk(
                ext_cx.expr(
                    span,
                    ast::expr_match(ext_cx.expr_var(span, ~"i"), arms)
                )
            )
        )
    );

    // ast for `__d.read_enum_variant($expr_arm_names, $(expr_lambda))`
    let expr_lambda = ext_cx.lambda_expr(
        ext_cx.expr_method_call(
            span,
            ext_cx.expr_var(span, ~"__d"),
            ext_cx.ident_of(~"read_enum_variant"),
            ~[expr_arm_names, expr_lambda]
        )
    );

    // ast for `__d.read_enum($(e_name), $(expr_lambda))`
    ext_cx.expr_method_call(
        span,
        ext_cx.expr_var(span, ~"__d"),
        ext_cx.ident_of(~"read_enum"),
        ~[
            ext_cx.lit_str(span, @ext_cx.str_of(name)),
            expr_lambda
        ]
    )
}

#[cfg(test)]
mod test {
    use core::option::{None, Some};
    use std::serialize::Encodable;
    use std::serialize::Encoder;

    // just adding the ones I want to test, for now:
    #[deriving(Eq)]
    pub enum call {
        CallToEmitEnum(~str),
        CallToEmitEnumVariant(~str, uint, uint),
        CallToEmitEnumVariantArg(uint),
        CallToEmitUint(uint),
        CallToEmitNil,
        CallToEmitStruct(~str,uint),
        CallToEmitField(~str,uint),
        CallToEmitOption,
        CallToEmitOptionNone,
        CallToEmitOptionSome,
        // all of the ones I was too lazy to handle:
        CallToOther
    }
    // using `@mut` rather than changing the
    // type of self in every method of every encoder everywhere.
    pub struct TestEncoder {call_log : @mut ~[call]}

    pub impl TestEncoder {
        // these self's should be &mut self's, as well....
        fn add_to_log (&self, c : call) {
            self.call_log.push(copy c);
        }
        fn add_unknown_to_log (&self) {
            self.add_to_log (CallToOther)
        }
    }

    impl Encoder for TestEncoder {
        fn emit_nil(&self) { self.add_to_log(CallToEmitNil) }

        fn emit_uint(&self, +v: uint) {self.add_to_log(CallToEmitUint(v)); }
        fn emit_u64(&self, +_v: u64) { self.add_unknown_to_log(); }
        fn emit_u32(&self, +_v: u32) { self.add_unknown_to_log(); }
        fn emit_u16(&self, +_v: u16) { self.add_unknown_to_log(); }
        fn emit_u8(&self, +_v: u8)   { self.add_unknown_to_log(); }

        fn emit_int(&self, +_v: int) { self.add_unknown_to_log(); }
        fn emit_i64(&self, +_v: i64) { self.add_unknown_to_log(); }
        fn emit_i32(&self, +_v: i32) { self.add_unknown_to_log(); }
        fn emit_i16(&self, +_v: i16) { self.add_unknown_to_log(); }
        fn emit_i8(&self, +_v: i8)   { self.add_unknown_to_log(); }

        fn emit_bool(&self, +_v: bool) { self.add_unknown_to_log(); }

        fn emit_f64(&self, +_v: f64) { self.add_unknown_to_log(); }
        fn emit_f32(&self, +_v: f32) { self.add_unknown_to_log(); }
        fn emit_float(&self, +_v: float) { self.add_unknown_to_log(); }

        fn emit_char(&self, +_v: char) { self.add_unknown_to_log(); }

        fn emit_borrowed_str(&self, +_v: &str) { self.add_unknown_to_log(); }
        fn emit_owned_str(&self, +_v: &str) { self.add_unknown_to_log(); }
        fn emit_managed_str(&self, +_v: &str) { self.add_unknown_to_log(); }

        fn emit_borrowed(&self, f: &fn()) { self.add_unknown_to_log(); f() }
        fn emit_owned(&self, f: &fn()) { self.add_unknown_to_log(); f() }
        fn emit_managed(&self, f: &fn()) { self.add_unknown_to_log(); f() }

        fn emit_enum(&self, name: &str, f: &fn()) {
            self.add_to_log(CallToEmitEnum(name.to_str())); f(); }

        fn emit_enum_variant(&self, name: &str, +id: uint,
                             +cnt: uint, f: &fn()) {
            self.add_to_log(CallToEmitEnumVariant (name.to_str(),id,cnt));
            f();
        }

        fn emit_enum_variant_arg(&self, +idx: uint, f: &fn()) {
            self.add_to_log(CallToEmitEnumVariantArg (idx)); f();
        }

        fn emit_borrowed_vec(&self, +_len: uint, f: &fn()) {
            self.add_unknown_to_log(); f();
        }

        fn emit_owned_vec(&self, +_len: uint, f: &fn()) {
            self.add_unknown_to_log(); f();
        }
        fn emit_managed_vec(&self, +_len: uint, f: &fn()) {
            self.add_unknown_to_log(); f();
        }
        fn emit_vec_elt(&self, +_idx: uint, f: &fn()) {
            self.add_unknown_to_log(); f();
        }

        fn emit_rec(&self, f: &fn()) {
            self.add_unknown_to_log(); f();
        }
        fn emit_struct(&self, name: &str, +len: uint, f: &fn()) {
            self.add_to_log(CallToEmitStruct (name.to_str(),len)); f();
        }
        fn emit_field(&self, name: &str, +idx: uint, f: &fn()) {
            self.add_to_log(CallToEmitField (name.to_str(),idx)); f();
        }

        fn emit_tup(&self, +_len: uint, f: &fn()) {
            self.add_unknown_to_log(); f();
        }
        fn emit_tup_elt(&self, +_idx: uint, f: &fn()) {
            self.add_unknown_to_log(); f();
        }

        fn emit_option(&self, f: &fn()) {
            self.add_to_log(CallToEmitOption);
            f();
        }
        fn emit_option_none(&self) {
            self.add_to_log(CallToEmitOptionNone);
        }
        fn emit_option_some(&self, f: &fn()) {
            self.add_to_log(CallToEmitOptionSome);
            f();
        }
    }


    fn to_call_log<E:Encodable<TestEncoder>>(val: E) -> ~[call] {
        let mut te = TestEncoder {call_log: @mut ~[]};
        val.encode(&te);
        copy *te.call_log
    }

    #[auto_encode]
    enum Written {
        Book(uint,uint),
        Magazine(~str)
    }

    #[test]
    fn test_encode_enum() {
        assert_eq!(
            to_call_log(Book(34,44)),
            ~[
                CallToEmitEnum(~"Written"),
                CallToEmitEnumVariant(~"Book",0,2),
                CallToEmitEnumVariantArg(0),
                CallToEmitUint(34),
                CallToEmitEnumVariantArg(1),
                CallToEmitUint(44),
            ]
        );
    }

    pub struct BPos(uint);

    #[auto_encode]
    pub struct HasPos { pos : BPos }

    #[test]
    fn test_encode_newtype() {
        assert_eq!(
            to_call_log(HasPos { pos:BPos(48) }),
            ~[
                CallToEmitStruct(~"HasPos",1),
                CallToEmitField(~"pos",0),
                CallToEmitUint(48),
            ]
        );
    }

    #[test]
    fn test_encode_option() {
        let mut v = None;

        assert_eq!(
            to_call_log(v),
            ~[
                CallToEmitOption,
                CallToEmitOptionNone,
            ]
        );

        v = Some(54u);
        assert_eq!(
            to_call_log(v),
            ~[
                CallToEmitOption,
                CallToEmitOptionSome,
                CallToEmitUint(54)
            ]
        );
    }
}
