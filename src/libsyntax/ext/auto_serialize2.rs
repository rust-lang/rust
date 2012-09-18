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

fn mk_impl(
    cx: ext_ctxt,
    span: span,
    ident: ast::ident,
    tps: ~[ast::ty_param],
    ser_body: @ast::stmt,
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
        mk_ser_method(cx, span, cx.blk(span, ~[ser_body])),
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

    let ser_inputs = ~[{
        mode: ast::expl(ast::by_ref),
        ty: cx.ty_path(span, ~[cx.ident_of(~"__S")], ~[]),
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
        self_ty: { node: ast::sty_by_ref, span: span },
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

    let deser_inputs = ~[{
        mode: ast::expl(ast::by_ref),
        ty: cx.ty_path(span, ~[cx.ident_of(~"__D")], ~[]),
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
         ast::expr_rec(fields, None)
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
        ast::expr_struct(cx.path(span, ~[ident]), fields, None)
    };

    mk_impl(cx, span, ident, tps, ser_body, deser_body)
}

fn mk_ser_fields(
    cx: ext_ctxt,
    span: span,
    fields: ~[{ span: span, ident: ast::ident, mutbl: ast::mutability }]
) -> @ast::stmt {
    let ext_cx = cx; // required for #ast{}

    let stmts = do fields.mapi |idx, field| {
        let name = cx.lit_str(field.span, @cx.str_of(field.ident));
        let idx = cx.lit_uint(field.span, idx);

        // XXX: The next couple stanzas are just to write
        // `self.$(name).serialize(s)`. It'd be nice if the #ast macro could
        // write this for us, but it doesn't appear to support quaziquoting a
        // value inside a field chain.
        let expr_self = cx.expr(
            span,
            ast::expr_path(
                cx.path(span, ~[cx.ident_of(~"self")])
            )
        );

        let expr_name = cx.expr(
            span,
            ast::expr_field(expr_self, field.ident, ~[])
        );

        let expr_serialize = cx.expr(
            span,
            ast::expr_field(expr_name, cx.ident_of(~"serialize"), ~[])
        );

        let expr_arg = cx.expr(
            span,
            ast::expr_path(
                cx.path(span, ~[cx.ident_of(~"__s")])
            )
        );

        let expr = cx.expr(
            span,
            ast::expr_call(expr_serialize, ~[expr_arg], false)
        );

        #ast[stmt]{ __s.emit_rec_field($(name), $(idx), || $(expr))) }
    };

    let fields_lambda = cx.lambda(cx.blk(span, stmts));
    #ast[stmt]{ __s.emit_rec($(fields_lambda)) }
}

fn mk_deser_fields(
    cx: ext_ctxt,
    span: span,
    fields: ~[{ span: span, ident: ast::ident, mutbl: ast::mutability }],
    f: fn(~[ast::field]) -> ast::expr_
) -> @ast::expr {
    let ext_cx = cx; // required for #ast{}

    let fields = do fields.mapi |idx, field| {
        let name = cx.lit_str(
            field.span,
            @cx.str_of(field.ident)
        );
        let idx = cx.lit_uint(field.span, idx);
        let expr = #ast{
             __d.read_rec_field($(name), $(idx), || deserialize(__d))
        };

        {
            node: { mutbl: field.mutbl, ident: field.ident, expr: expr },
            span: field.span,
        }
    };

    let fields_expr = cx.expr(span, f(fields));
    let fields_lambda = cx.lambda(cx.expr_blk(fields_expr));
    #ast{ __d.read_rec($(fields_lambda)) }
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
        ident,
        span,
        enum_def.variants
    );

    let deser_body = mk_enum_deser_body(
        cx,
        ident,
        span,
        enum_def.variants
    );

    mk_impl(cx, span, ident, tps, ser_body, deser_body)
}

fn ser_variant(
    cx: ext_ctxt,
    v_span: span,
    v_name: ast::ident,
    v_idx: @ast::expr,
    args: ~[ast::variant_arg]
) -> ast::arm {
    let ext_cx = cx; // required for #ast{}

    // Name the variant arguments.
    let names = args.mapi(|i, _arg| cx.ident_of(fmt!("__v%u", i)));

    // Bind the names to the variant argument type.
    let pats = args.mapi(|i, arg| cx.binder_pat(arg.ty.span, names[i]));

    let pat_node = if pats.is_empty() {
        ast::pat_ident(
            ast::bind_by_implicit_ref,
            cx.path(v_span, ~[v_name]),
            None
        )
    } else {
        ast::pat_enum(
            cx.path(v_span, ~[v_name]),
            Some(pats)
        )
    };

    let pat = @{
        id: cx.next_id(),
        node: pat_node,
        span: v_span,
    };

    // Create the s.emit_variant_arg statements.
    let stmts = do args.mapi |a_idx, _arg| {
        let v = cx.var_ref(v_span, names[a_idx]);
        let a_idx = cx.lit_uint(v_span, a_idx);

        #ast[stmt]{
            __s.emit_enum_variant_arg($(a_idx), || $(v).serialize(__s));
        }
    };

    let v_name = cx.lit_str(v_span, @cx.str_of(v_name));
    let v_sz = cx.lit_uint(v_span, stmts.len());
    let lambda = cx.lambda(cx.blk(v_span, stmts));
    let body = #ast{
         __s.emit_enum_variant($(v_name), $(v_idx), $(v_sz), $(lambda))
    };

    { pats: ~[pat], guard: None, body: cx.expr_blk(body) }
}

fn mk_enum_ser_body(
    cx: ext_ctxt,
    e_name: ast::ident,
    e_span: span,
    variants: ~[ast::variant]
) -> @ast::stmt {
    let ext_cx = cx; // required for #ast{}

    let arms = do variants.mapi |v_idx, variant| {
        let v_span = variant.span;
        let v_name = variant.node.name;
        let v_idx = cx.lit_uint(v_span, v_idx);

        match variant.node.kind {
            ast::tuple_variant_kind(args) =>
                ser_variant(cx, v_span, v_name, v_idx, args),
            ast::struct_variant_kind(*) =>
                fail ~"struct variants unimplemented",
            ast::enum_variant_kind(*) =>
                fail ~"enum variants unimplemented",
        }
    };

    let match_expr = cx.expr(
        e_span,
        ast::expr_match(#ast{ self }, arms)
    );
    let e_name = cx.lit_str(e_span, @cx.str_of(e_name));

    #ast[stmt]{ __s.emit_enum($(e_name), || $(match_expr)) }
}

fn mk_enum_deser_body(
    cx: ext_ctxt,
    e_name: ast::ident,
    e_span: span,
    variants: ~[ast::variant]
) -> @ast::expr {
    let ext_cx = cx; // required for #ast{}

    let mut arms = do variants.mapi |v_idx, variant| {
        let v_span = variant.span;
        let v_name = variant.node.name;

        let body = match variant.node.kind {
            ast::tuple_variant_kind(args) => {
                let tys = args.map(|a| a.ty);

                if tys.is_empty() {
                    // for a nullary variant v, do "v"
                    cx.var_ref(v_span, v_name)
                } else {
                    // for an n-ary variant v, do "v(a_1, ..., a_n)"

                    let arg_exprs = do tys.mapi |a_idx, _ty| {
                        let a_idx = cx.lit_uint(v_span, a_idx);
                        #ast{
                            __d.read_enum_variant_arg($(a_idx), || {
                                deserialize(__d)
                            })
                        }
                    };

                    cx.expr(
                        v_span,
                        ast::expr_call(
                            cx.var_ref(v_span, v_name),
                            arg_exprs,
                            false
                        )
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
            node: ast::pat_lit(cx.lit_uint(v_span, v_idx)),
            span: v_span,
        };

        {
            pats: ~[pat],
            guard: None,
            body: cx.expr_blk(body),
        }
    };

    let impossible_case = {
        pats: ~[@{ id: cx.next_id(), node: ast::pat_wild, span: e_span}],
        guard: None,

        // FIXME(#3198): proper error message
        body: cx.expr_blk(cx.expr(e_span, ast::expr_fail(None))),
    };

    vec::push(arms, impossible_case);

    let e_name = cx.lit_str(e_span, @cx.str_of(e_name));
    let alt_expr = cx.expr(e_span, ast::expr_match(#ast{ i }, arms));

    #ast{
        __d.read_enum($(e_name), || {
            __d.read_enum_variant(|i| {
                $(alt_expr)
            })
        })
    }
}
