/*

The compiler code necessary to implement the #[auto_serialize]
extension.  The idea here is that type-defining items may be tagged
with #[auto_serialize], which will cause us to generate a little
companion module with the same name as the item.

For example, a type like:

    type node_id = uint;

would generate a companion module like:

    mod node_id {
        use std;
        import std::serialization::serializer;
        import std::serialization::deserializer;
        fn serialize<S: serializer>(s: S, v: node_id) {
             s.emit_uint(v);
        }
        fn deserializer<D: deserializer>(d: D) -> node_id {
             d.read_uint()
        }
    }

Other interesting scenarios are whe the item has type parameters or
references other non-built-in types.  A type definition like:

    type spanned<T> = {node: T, span: span};

would yield a helper module like:

    mod spanned {
        use std;
        import std::serialization::serializer;
        import std::serialization::deserializer;
        fn serialize<S: serializer,T>(s: S, t: fn(T), v: spanned<T>) {
             s.emit_rec(2u) {||
                 s.emit_rec_field("node", 0u) {||
                     t(s.node);
                 };
                 s.emit_rec_field("span", 1u) {||
                     span::serialize(s, s.span);
                 };
             }
        }
        fn deserializer<D: deserializer>(d: D, t: fn() -> T) -> node_id {
             d.read_rec(2u) {||
                 {node: d.read_rec_field("node", 0u, t),
                  span: d.read_rec_field("span", 1u) {||span::deserialize(d)}}
             }
        }
    }

In general, the code to serialize an instance `v` of a non-built-in
type a::b::c<T0,...,Tn> looks like:

    a::b::c::serialize(s, {|v| c_T0}, ..., {|v| c_Tn}, v)

where `c_Ti` is the code to serialize an instance `v` of the type
`Ti`.

Similarly, the code to deserialize an instance of a non-built-in type
`a::b::c<T0,...,Tn>` using the deserializer `d` looks like:

    a::b::c::deserialize(d, {|| c_T0}, ..., {|| c_Tn})

where `c_Ti` is the code to deserialize an instance of `Ti` using the
deserializer `d`.

TODO--Hygiene. Search for "__" strings.

*/
import base::*;
import driver::session::session;
import codemap::span;
import std::map;

export expand_auto_serialize;

enum ser_cx = {
    ext_cx: ext_ctxt,
    tps: map::map<str, fn@(@ast::expr) -> [@ast::stmt]>
};

fn expand_auto_serialize(cx: ext_ctxt,
                         span: span,
                         mitem: ast::meta_item,
                         in_items: [@ast::item]) -> [@ast::item] {
    vec::flat_map(in_items) {|in_item|
        alt in_item.node {
          ast::item_ty(ty, tps) {
            [in_item, ty_module(cx, in_item.ident, ty, tps)]
          }

          ast::item_enum(variants, tps) {
            [in_item, enum_module(cx, in_item.ident, variants, tps)]
          }

          _ {
            cx.session().span_err(span, "#[auto_serialize] can only be \
                                         applied to type and enum \
                                         definitions");
            [in_item]
          }
        }
    }
}

impl helpers for ser_cx {
    fn session() -> session { self.ext_cx.session() }

    fn next_id() -> ast::node_id { self.session().next_node_id() }

    fn path(span: span, strs: [str]) -> @ast::path {
        @{node: {global: false,
                 idents: strs + ["serialize"],
                 types: []},
          span: span}
    }

    fn expr(span: span, node: ast::expr_) -> @ast::expr {
        @{id: self.next_id(), node: node, span: span}
    }

    fn ty_path(span: span, strs: [str]) -> @ast::ty {
        @{node: ast::ty_path(self.path(span, strs), self.next_id()),
          span: span}
    }

    fn var_ref(span: span, name: str) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path(span, [name])))
    }

    fn blk(span: span, stmts: [@ast::stmt]) -> ast::blk {
        {node: {view_items: [],
                stmts: stmts,
                expr: none,
                id: self.next_id(),
                rules: ast::default_blk},
         span: span}
    }

    fn binder_pat(span: span, nm: str) -> @ast::pat {
        let path = @{node: {global: false,
                            idents: [nm],
                            types: []},
                     span: span};
        @{id: self.next_id(),
          node: ast::pat_ident(path, none),
          span: span}
    }

    fn stmt(expr: @ast::expr) -> @ast::stmt {
        @{node: ast::stmt_semi(expr, self.next_id()),
          span: expr.span}
    }

    fn alt_stmt(arms: [ast::arm], span: span, -v: @ast::expr) -> @ast::stmt {
        self.stmt(
            self.expr(
                span,
                ast::expr_alt(v, arms, ast::alt_exhaustive)))
    }

    fn clone(v: @ast::expr) -> @ast::expr {
        let fld = fold::make_fold({
            new_id: {|_id| self.next_id()}
            with *fold::default_ast_fold()
        });
        fld.fold_expr(v)
    }

    fn clone_ty_param(v: ast::ty_param) -> ast::ty_param {
        let fld = fold::make_fold({
            new_id: {|_id| self.next_id()}
            with *fold::default_ast_fold()
        });
        fold::fold_ty_param(v, fld)
    }

    fn at(span: span, expr: @ast::expr) -> @ast::expr {
        fn repl_sp(old_span: span, repl_span: span, with_span: span) -> span {
            if old_span == repl_span {
                with_span
            } else {
                old_span
            }
        }

        let fld = fold::make_fold({
            new_span: repl_sp(_, ast_util::dummy_sp(), span)
            with *fold::default_ast_fold()
        });

        fld.fold_expr(expr)
    }
}

fn serialize_path(cx: ser_cx, path: @ast::path, -s: @ast::expr, -v: @ast::expr)
    -> [@ast::stmt] {
    let ext_cx = cx.ext_cx;

    // We want to take a path like a::b::c<...> and generate a call
    // like a::b::c::serialize(s, ...), as described above.

    let callee =
        cx.expr(
            path.span,
            ast::expr_path(
                cx.path(path.span, path.node.idents + ["serialize"])));

    let ty_args = vec::map(path.node.types) {|ty|
        let sv = serialize_ty(cx, ty, s, #ast(expr){"__v"});
        cx.at(ty.span, #ast(expr){"{|__v| $(sv)}"})
    };

    [cx.stmt(
        cx.expr(
            path.span,
            ast::expr_call(callee, [s] + ty_args + [v], false)))]
}

fn serialize_variant(cx: ser_cx,
                     tys: [@ast::ty],
                     span: span,
                     -s: @ast::expr,
                     pfn: fn([@ast::pat]) -> ast::pat_) -> ast::arm {
    let vnames = vec::init_fn(vec::len(tys)) {|i| #fmt["__v%u", i]};
    let pats = vec::init_fn(vec::len(tys)) {|i|
        cx.binder_pat(tys[i].span, vnames[i])
    };
    let pat: @ast::pat = @{id: cx.next_id(), node: pfn(pats), span: span};
    let stmts = vec::init_fn(vec::len(tys)) {|i|
        let v = cx.var_ref(span, vnames[i]);
        serialize_ty(cx, tys[i], cx.clone(s), v)
    };
    {pats: [pat], guard: none, body: cx.blk(span, vec::concat(stmts))}
}

fn serialize_ty(cx: ser_cx, ty: @ast::ty, -s: @ast::expr, -v: @ast::expr)
    -> [@ast::stmt] {
    let ext_cx = cx.ext_cx;

    alt ty.node {
      ast::ty_nil | ast::ty_bot {
        []
      }

      ast::ty_box(mt) |
      ast::ty_uniq(mt) |
      ast::ty_ptr(mt) {
        serialize_ty(cx, mt.ty, s, #ast(expr){"*$(v)"})
      }

      ast::ty_rec(flds) {
        vec::flat_map(flds) {|fld|
            let vf = cx.expr(
                fld.span,
                ast::expr_field(cx.clone(v), fld.node.ident, []));
            serialize_ty(cx, fld.node.mt.ty, cx.clone(s), vf)
        }
      }

      ast::ty_fn(_, _) {
        cx.session().span_err(
            ty.span, #fmt["Cannot serialize function types"]);
        []
      }

      ast::ty_tup(tys) {
        // Generate code like
        //
        // alt v {
        //    (v1, v2, v3) {
        //       .. serialize v1, v2, v3 ..
        //    }
        // };

        let arms = [
            serialize_variant(cx, tys, ty.span, s,
                              {|pats| ast::pat_tup(pats)})
        ];
        [cx.alt_stmt(arms, ty.span, v)]
      }

      ast::ty_path(path, _) {
        if vec::len(path.node.idents) == 1u &&
            vec::is_empty(path.node.types) {
            let ident = path.node.idents[0];

            alt cx.tps.find(ident) {
              some(f) { f(v) }
              none { serialize_path(cx, path, s, v) }
            }
        } else {
            serialize_path(cx, path, s, v)
        }
      }

      ast::ty_constr(ty, _) {
        serialize_ty(cx, ty, s, v)
      }

      ast::ty_mac(_) {
        cx.session().span_err(
            ty.span, #fmt["Cannot serialize macro types"]);
        []
      }

      ast::ty_infer {
        cx.session().span_err(
            ty.span, #fmt["Cannot serialize inferred types"]);
        []
      }

      ast::ty_vec(mt) {
        let ser_e =
            cx.expr(
                ty.span,
                expr_block(
                    cx.blk(
                        ty.span,
                        serialize_ty(
                            cx, mt.ty,
                            cx.clone(s),
                            cx.at(
                                ty.span,
                                #ast(expr){__e})))));
        [#ast(stmt){ $(s).emit_from_vec($(v), {|__e| $(ser_e) }) }]
      }
    }
}

fn ty_module(ext_cx: ext_ctxt, name: str, -ty: @ast::ty, tps: [ast::ty_param])
    -> @ast::item {

    let cx = ser_cx({ext_cx: ext_cx, tps: map::new_str_hash()});

    let ser_inputs: [ast::arg] =
        [{mode: ast::expl(ast::by_ref),
          ty: cx.ty_path(ty.span, ["__S"]),
          ident: "__s",
          id: cx.next_id()},
         {mode: ast::expl(ast::by_ref),
          ty: ty,
          ident: "__v",
          id: cx.next_id()}] +
        vec::map(tps, {|tp|
            {mode: ast::expl(ast::by_ref),
             ty: cx.ty_path(ty.span, [tp.ident]),
             ident: "__v",
             id: cx.next_id()}});

    let ser_bnds = @[ast::bound_iface(cx.ty_path(ty.span,
                                                 ["__std", "serialization",
                                                  "serializer"]))];
    let ser_tps: [ast::ty_param] =
        [{ident: "__S",
          id: cx.next_id(),
          bounds: ser_bnds}] +
        vec::map(tps) {|tp| cx.clone_ty_param(tp) };

    let ser_output: @ast::ty = @{node: ast::ty_nil,
                                 span: ty.span};

    let ser_blk = cx.blk(ty.span,
                         serialize_ty(cx, ty,
                                      #ast(expr){"__s"}, #ast(expr){"__v"}));

    @{ident: "serialize",
      attrs: [],
      id: cx.next_id(),
      node: ast::item_fn({inputs: ser_inputs,
                          output: ser_output,
                          purity: ast::impure_fn,
                          cf: ast::return_val,
                          constraints: []},
                         ser_tps,
                         ser_blk),
      span: ty.span}
}

fn enum_module(cx: ext_ctxt, name: str,
               variants: [ast::variant], tps: [ast::ty_param])
    -> @ast::item {

}