/*

The compiler code necessary to implement the #[auto_serialize2]
extension.  The idea here is that type-defining items may be tagged
with #[auto_serialize2], which will cause us to generate a little
companion module with the same name as the item.

For example, a type like:

    type node_id = uint;

would generate two functions like:

    impl node_id: Serializable {
        fn serialize<S: Serializer>(s: S) {
            s.emit_uint(self)
        }

        static fn deserialize<D: Deserializer>(d: D) -> node_id {
            d.read_uint()
        }
    }

Other interesting scenarios are whe the item has type parameters or
references other non-built-in types.  A type definition like:

    type spanned<T> = {node: T, span: span};

would yield functions like:

    impl<T: Serializable> spanned<T>: Serializable {
        fn serialize<S: Serializer>(s: S) {
            do s.emit_rec {
                s.emit_rec_field("node", 0, self.node.serialize(s));
                s.emit_rec_field("span", 1, self.span.serialize(s));
            }
        }

        static fn deserialize<D: Deserializer>(d: D) -> spanned<T> {
            do d.read_rec {
                {
                    node: d.read_rec_field(~"node", 0, || deserialize(d)),
                    span: d.read_rec_field(~"span", 1, || deserialize(d)),
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

use base::*;
use codemap::span;
use std::map;
use std::map::HashMap;

export expand;

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    pub use ext;
    pub use parse;
}

fn expand(cx: ext_ctxt,
          span: span,
          _mitem: ast::meta_item,
          in_items: ~[@ast::item]) -> ~[@ast::item] {
    fn not_auto_serialize2(a: ast::attribute) -> bool {
        attr::get_attr_name(a) != ~"auto_serialize2"
    }

    fn filter_attrs(item: @ast::item) -> @ast::item {
        @{attrs: vec::filter(item.attrs, not_auto_serialize2),
          .. *item}
    }

    do vec::flat_map(in_items) |item| {
        match item.node {
            ast::item_ty(@{node: ast::ty_rec(fields), _}, tps) => {
                ~[
                    filter_attrs(item),
                    mk_rec_impl(cx, item.span, item.ident, fields, tps),
                ]
            },
            ast::item_class(@{ fields, _}, tps) => {
                ~[
                    filter_attrs(item),
                    mk_struct_impl(cx, item.span, item.ident, fields, tps),
                ]
            },
            ast::item_enum(enum_def, tps) => {
                ~[
                    filter_attrs(item),
                    mk_enum_impl(cx, item.span, item.ident, enum_def, tps),
                ]
            },
            _ => {
                cx.span_err(span, ~"#[auto_serialize2] can only be applied \
                                    to structs, record types, and enum \
                                    definitions");
                ~[item]
            }
        }
    }
}

priv impl ext_ctxt {
    fn expr_path(span: span, strs: ~[ast::ident]) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path(span, strs)))
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
    tps: ~[ast::ty_param],
    ser_body: @ast::expr,
    deser_body: @ast::expr
) -> @ast::item {
    // Make a path to the std::serialization2::Serializable trait.
    let path = cx.path(
        span,
        ~[
            cx.ident_of(~"std"),
            cx.ident_of(~"serialization2"),
            cx.ident_of(~"Serializable"),
        ]
    );

    // All the type parameters need to bound to
    // std::serialization::Serializable.
    let trait_tps = do tps.map |tp| {
        let t_bound = ast::bound_trait(@{
            id: cx.next_id(),
            node: ast::ty_path(path, cx.next_id()),
            span: span,
        });

        {
            ident: tp.ident,
            id: cx.next_id(),
            bounds: @vec::append(~[t_bound], *tp.bounds)
        }
    };

    let opt_trait = Some(@{
        path: path,
        ref_id: cx.next_id(),
        impl_id: cx.next_id(),
    });

    let ty = cx.ty_path(
        span,
        ~[ident],
        tps.map(|tp| cx.ty_path(span, ~[tp.ident], ~[]))
    );

    let methods = ~[
        mk_ser_method(cx, span, cx.expr_blk(ser_body)),
        mk_deser_method(cx, span, ty, cx.expr_blk(deser_body)),
    ];

    @{
        // This is a new-style impl declaration.
        // XXX: clownshoes
        ident: ast::token::special_idents::clownshoes_extensions,
        attrs: ~[],
        id: cx.next_id(),
        node: ast::item_impl(trait_tps, opt_trait, ty, methods),
        vis: ast::public,
        span: span,
    }
}

fn mk_ser_method(
    cx: ext_ctxt,
    span: span,
    ser_body: ast::blk
) -> @ast::method {
    let ser_bound = cx.ty_path(
        span,
        ~[
            cx.ident_of(~"std"),
            cx.ident_of(~"serialization2"),
            cx.ident_of(~"Serializer"),
        ],
        ~[]
    );

    let ser_tps = ~[{
        ident: cx.ident_of(~"__S"),
        id: cx.next_id(),
        bounds: @~[ast::bound_trait(ser_bound)],
    }];

    let ty_s = @{
        id: cx.next_id(),
        node: ast::ty_rptr(
            @{
                id: cx.next_id(),
                node: ast::re_anon,
            },
            {
                ty: cx.ty_path(span, ~[cx.ident_of(~"__S")], ~[]),
                mutbl: ast::m_imm
            }
        ),
        span: span,
    };

    let ser_inputs = ~[{
        mode: ast::infer(cx.next_id()),
        ty: ty_s,
        ident: cx.ident_of(~"__s"),
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
        ident: cx.ident_of(~"serialize"),
        attrs: ~[],
        tps: ser_tps,
        self_ty: { node: ast::sty_region(ast::m_imm), span: span },
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
    ty: @ast::ty,
    deser_body: ast::blk
) -> @ast::method {
    let deser_bound = cx.ty_path(
        span,
        ~[
            cx.ident_of(~"std"),
            cx.ident_of(~"serialization2"),
            cx.ident_of(~"Deserializer"),
        ],
        ~[]
    );

    let deser_tps = ~[{
        ident: cx.ident_of(~"__D"),
        id: cx.next_id(),
        bounds: @~[ast::bound_trait(deser_bound)],
    }];

    let ty_d = @{
        id: cx.next_id(),
        node: ast::ty_rptr(
            @{
                id: cx.next_id(),
                node: ast::re_anon,
            },
            {
                ty: cx.ty_path(span, ~[cx.ident_of(~"__D")], ~[]),
                mutbl: ast::m_imm
            }
        ),
        span: span,
    };

    let deser_inputs = ~[{
        mode: ast::infer(cx.next_id()),
        ty: ty_d,
        ident: cx.ident_of(~"__d"),
        id: cx.next_id(),
    }];

    let deser_decl = {
        inputs: deser_inputs,
        output: ty,
        cf: ast::return_val,
    };

    @{
        ident: cx.ident_of(~"deserialize"),
        attrs: ~[],
        tps: deser_tps,
        self_ty: { node: ast::sty_static, span: span },
        purity: ast::impure_fn,
        decl: deser_decl,
        body: deser_body,
        id: cx.next_id(),
        span: span,
        self_id: cx.next_id(),
        vis: ast::public,
    }
}

fn mk_rec_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    fields: ~[ast::ty_field],
    tps: ~[ast::ty_param]
) -> @ast::item {
    // Records and structs don't have the same fields types, but they share
    // enough that if we extract the right subfields out we can share the
    // serialization generator code.
    let fields = do fields.map |field| {
        {
            span: field.span,
            ident: field.node.ident,
            mutbl: field.node.mt.mutbl,
        }
    };

    let ser_body = mk_ser_fields(cx, span, fields);
    let deser_body = do mk_deser_fields(cx, span, fields) |fields| {
         cx.expr(span, ast::expr_rec(fields, None))
    };

    mk_impl(cx, span, ident, tps, ser_body, deser_body)
}

fn mk_struct_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    fields: ~[@ast::struct_field],
    tps: ~[ast::ty_param]
) -> @ast::item {
    // Records and structs don't have the same fields types, but they share
    // enough that if we extract the right subfields out we can share the
    // serialization generator code.
    let fields = do fields.map |field| {
        let (ident, mutbl) = match field.node.kind {
            ast::named_field(ident, mutbl, _) => (ident, mutbl),
            _ => fail ~"[auto_serialize2] does not support \
                        unnamed fields",
        };

        {
            span: field.span,
            ident: ident,
            mutbl: match mutbl {
                ast::class_mutable => ast::m_mutbl,
                ast::class_immutable => ast::m_imm,
            },
        }
    };

    let ser_body = mk_ser_fields(cx, span, fields);
    let deser_body = do mk_deser_fields(cx, span, fields) |fields| {
        cx.expr(span, ast::expr_struct(cx.path(span, ~[ident]), fields, None))
    };

    mk_impl(cx, span, ident, tps, ser_body, deser_body)
}

fn mk_ser_fields(
    cx: ext_ctxt,
    span: span,
    fields: ~[{ span: span, ident: ast::ident, mutbl: ast::mutability }]
) -> @ast::expr {
    let stmts = do fields.mapi |idx, field| {
        // ast for `|| self.$(name).serialize(__s)`
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
                    cx.ident_of(~"serialize")
                ),
                ~[cx.expr_var(span, ~"__s")]
            )
        );

        // ast for `__s.emit_rec_field($(name), $(idx), $(expr_lambda))`
        cx.stmt(
            cx.expr_call(
                span,
                cx.expr_field(
                    span,
                    cx.expr_var(span, ~"__s"),
                    cx.ident_of(~"emit_rec_field")
                ),
                ~[
                    cx.lit_str(span, @cx.str_of(field.ident)),
                    cx.lit_uint(span, idx),
                    expr_lambda,
                ]
            )
        )
    };

    // ast for `__s.emit_rec(|| $(stmts))`
    cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__s"),
            cx.ident_of(~"emit_rec")
        ),
        ~[cx.lambda_stmts(span, stmts)]
    )
}

fn mk_deser_fields(
    cx: ext_ctxt,
    span: span,
    fields: ~[{ span: span, ident: ast::ident, mutbl: ast::mutability }],
    f: fn(~[ast::field]) -> @ast::expr
) -> @ast::expr {
    let fields = do fields.mapi |idx, field| {
        // ast for `|| std::serialization2::deserialize(__d)`
        let expr_lambda = cx.lambda(
            cx.expr_blk(
                cx.expr_call(
                    span,
                    cx.expr_path(span, ~[
                        cx.ident_of(~"std"),
                        cx.ident_of(~"serialization2"),
                        cx.ident_of(~"deserialize"),
                    ]),
                    ~[cx.expr_var(span, ~"__d")]
                )
            )
        );

        // ast for `__d.read_rec_field($(name), $(idx), $(expr_lambda))`
        let expr: @ast::expr = cx.expr_call(
            span,
            cx.expr_field(
                span,
                cx.expr_var(span, ~"__d"),
                cx.ident_of(~"read_rec_field")
            ),
            ~[
                cx.lit_str(span, @cx.str_of(field.ident)),
                cx.lit_uint(span, idx),
                expr_lambda,
            ]
        );

        {
            node: { mutbl: field.mutbl, ident: field.ident, expr: expr },
            span: span,
        }
    };

    // ast for `__d.read_rec(|| $(fields_expr))`
    cx.expr_call(
        span,
        cx.expr_field(
            span,
            cx.expr_var(span, ~"__d"),
            cx.ident_of(~"read_rec")
        ),
        ~[cx.lambda_expr(f(fields))]
    )
}

fn mk_enum_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    enum_def: ast::enum_def,
    tps: ~[ast::ty_param]
) -> @ast::item {
    let ser_body = mk_enum_ser_body(
        cx,
        span,
        ident,
        enum_def.variants
    );

    let deser_body = mk_enum_deser_body(
        cx,
        span,
        ident,
        enum_def.variants
    );

    mk_impl(cx, span, ident, tps, ser_body, deser_body)
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
            ast::bind_by_implicit_ref,
            cx.path(span, ~[v_name]),
            None
        )
    } else {
        ast::pat_enum(
            cx.path(span, ~[v_name]),
            Some(pats)
        )
    };

    let pat = @{
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

        // ast for `|| $(v).serialize(__s)`
        let expr_serialize = cx.lambda_expr(
             cx.expr_call(
                span,
                cx.expr_field(
                    span,
                    cx.expr_path(span, ~[names[a_idx]]),
                    cx.ident_of(~"serialize")
                ),
                ~[cx.expr_var(span, ~"__s")]
            )
        );

        // ast for `$(expr_emit)($(a_idx), $(expr_serialize))`
        cx.stmt(
            cx.expr_call(
                span,
                expr_emit,
                ~[cx.lit_uint(span, a_idx), expr_serialize]
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

    { pats: ~[pat], guard: None, body: cx.expr_blk(body) }
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
        // ast for `|| std::serialization2::deserialize(__d)`
        let expr_lambda = cx.lambda_expr(
            cx.expr_call(
                span,
                cx.expr_path(span, ~[
                    cx.ident_of(~"std"),
                    cx.ident_of(~"serialization2"),
                    cx.ident_of(~"deserialize"),
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

        let pat = @{
            id: cx.next_id(),
            node: ast::pat_lit(cx.lit_uint(span, v_idx)),
            span: span,
        };

        {
            pats: ~[pat],
            guard: None,
            body: cx.expr_blk(body),
        }
    };

    let impossible_case = {
        pats: ~[@{ id: cx.next_id(), node: ast::pat_wild, span: span}],
        guard: None,

        // FIXME(#3198): proper error message
        body: cx.expr_blk(cx.expr(span, ast::expr_fail(None))),
    };

    vec::push(arms, impossible_case);

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
                    ident: cx.ident_of(~"i"),
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
