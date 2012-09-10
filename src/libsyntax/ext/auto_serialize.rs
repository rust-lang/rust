/*

The compiler code necessary to implement the #[auto_serialize]
extension.  The idea here is that type-defining items may be tagged
with #[auto_serialize], which will cause us to generate a little
companion module with the same name as the item.

For example, a type like:

    type node_id = uint;

would generate two functions like:

    fn serialize_node_id<S: serializer>(s: S, v: node_id) {
        s.emit_uint(v);
    }
    fn deserialize_node_id<D: deserializer>(d: D) -> node_id {
        d.read_uint()
    }

Other interesting scenarios are whe the item has type parameters or
references other non-built-in types.  A type definition like:

    type spanned<T> = {node: T, span: span};

would yield functions like:

    fn serialize_spanned<S: serializer,T>(s: S, v: spanned<T>, t: fn(T)) {
         s.emit_rec(2u) {||
             s.emit_rec_field("node", 0u) {||
                 t(s.node);
             };
             s.emit_rec_field("span", 1u) {||
                 serialize_span(s, s.span);
             };
         }
    }
    fn deserialize_spanned<D: deserializer>(d: D, t: fn() -> T) -> node_id {
         d.read_rec(2u) {||
             {node: d.read_rec_field("node", 0u, t),
              span: d.read_rec_field("span", 1u) {||deserialize_span(d)}}
         }
    }

In general, the code to serialize an instance `v` of a non-built-in
type a::b::c<T0,...,Tn> looks like:

    a::b::serialize_c(s, {|v| c_T0}, ..., {|v| c_Tn}, v)

where `c_Ti` is the code to serialize an instance `v` of the type
`Ti`.

Similarly, the code to deserialize an instance of a non-built-in type
`a::b::c<T0,...,Tn>` using the deserializer `d` looks like:

    a::b::deserialize_c(d, {|| c_T0}, ..., {|| c_Tn})

where `c_Ti` is the code to deserialize an instance of `Ti` using the
deserializer `d`.

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
    import ext;
    export ext;
    import parse;
    export parse;
}

type ser_tps_map = map::HashMap<ast::ident, fn@(@ast::expr) -> ~[@ast::stmt]>;
type deser_tps_map = map::HashMap<ast::ident, fn@() -> @ast::expr>;

fn expand(cx: ext_ctxt,
          span: span,
          _mitem: ast::meta_item,
          in_items: ~[@ast::item]) -> ~[@ast::item] {
    fn not_auto_serialize(a: ast::attribute) -> bool {
        attr::get_attr_name(a) != ~"auto_serialize"
    }

    fn filter_attrs(item: @ast::item) -> @ast::item {
        @{attrs: vec::filter(item.attrs, not_auto_serialize),
          .. *item}
    }

    do vec::flat_map(in_items) |in_item| {
        match in_item.node {
          ast::item_ty(ty, tps) => {
            vec::append(~[filter_attrs(in_item)],
                        ty_fns(cx, in_item.ident, ty, tps))
          }

          ast::item_enum(enum_definition, tps) => {
            vec::append(~[filter_attrs(in_item)],
                        enum_fns(cx, in_item.ident,
                                 in_item.span, enum_definition.variants, tps))
          }

          _ => {
            cx.span_err(span, ~"#[auto_serialize] can only be \
                               applied to type and enum \
                               definitions");
            ~[in_item]
          }
        }
    }
}

trait ext_ctxt_helpers {
    fn helper_path(base_path: @ast::path, helper_name: ~str) -> @ast::path;
    fn path(span: span, strs: ~[ast::ident]) -> @ast::path;
    fn path_tps(span: span, strs: ~[ast::ident],
                tps: ~[@ast::ty]) -> @ast::path;
    fn ty_path(span: span, strs: ~[ast::ident], tps: ~[@ast::ty]) -> @ast::ty;
    fn ty_fn(span: span,
             -input_tys: ~[@ast::ty],
             -output: @ast::ty) -> @ast::ty;
    fn ty_nil(span: span) -> @ast::ty;
    fn expr(span: span, node: ast::expr_) -> @ast::expr;
    fn var_ref(span: span, name: ast::ident) -> @ast::expr;
    fn blk(span: span, stmts: ~[@ast::stmt]) -> ast::blk;
    fn expr_blk(expr: @ast::expr) -> ast::blk;
    fn binder_pat(span: span, nm: ast::ident) -> @ast::pat;
    fn stmt(expr: @ast::expr) -> @ast::stmt;
    fn alt_stmt(arms: ~[ast::arm], span: span, -v: @ast::expr) -> @ast::stmt;
    fn lit_str(span: span, s: @~str) -> @ast::expr;
    fn lit_uint(span: span, i: uint) -> @ast::expr;
    fn lambda(blk: ast::blk) -> @ast::expr;
    fn clone_folder() -> fold::ast_fold;
    fn clone(v: @ast::expr) -> @ast::expr;
    fn clone_ty(v: @ast::ty) -> @ast::ty;
    fn clone_ty_param(v: ast::ty_param) -> ast::ty_param;
    fn at(span: span, expr: @ast::expr) -> @ast::expr;
}

impl ext_ctxt: ext_ctxt_helpers {
    fn helper_path(base_path: @ast::path,
                   helper_name: ~str) -> @ast::path {
        let head = vec::init(base_path.idents);
        let tail = vec::last(base_path.idents);
        self.path(base_path.span,
                  vec::append(head,
                              ~[self.parse_sess().interner.
                                intern(@(helper_name + ~"_" +
                                         *self.parse_sess().interner.get(
                                             tail)))]))
    }

    fn path(span: span, strs: ~[ast::ident]) -> @ast::path {
        @{span: span, global: false, idents: strs, rp: None, types: ~[]}
    }

    fn path_tps(span: span, strs: ~[ast::ident],
                tps: ~[@ast::ty]) -> @ast::path {
        @{span: span, global: false, idents: strs, rp: None, types: tps}
    }

    fn ty_path(span: span, strs: ~[ast::ident],
               tps: ~[@ast::ty]) -> @ast::ty {
        @{id: self.next_id(),
          node: ast::ty_path(self.path_tps(span, strs, tps), self.next_id()),
          span: span}
    }

    fn ty_fn(span: span,
             -input_tys: ~[@ast::ty],
             -output: @ast::ty) -> @ast::ty {
        let args = do vec::map(input_tys) |ty| {
            {mode: ast::expl(ast::by_ref),
             ty: ty,
             ident: parse::token::special_idents::invalid,
             id: self.next_id()}
        };

        @{id: self.next_id(),
          node: ast::ty_fn(ast::proto_block,
                           ast::impure_fn,
                           @~[],
                           {inputs: args,
                            output: output,
                            cf: ast::return_val}),
          span: span}
    }

    fn ty_nil(span: span) -> @ast::ty {
        @{id: self.next_id(), node: ast::ty_nil, span: span}
    }

    fn expr(span: span, node: ast::expr_) -> @ast::expr {
        @{id: self.next_id(), callee_id: self.next_id(),
          node: node, span: span}
    }

    fn var_ref(span: span, name: ast::ident) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path(span, ~[name])))
    }

    fn blk(span: span, stmts: ~[@ast::stmt]) -> ast::blk {
        {node: {view_items: ~[],
                stmts: stmts,
                expr: None,
                id: self.next_id(),
                rules: ast::default_blk},
         span: span}
    }

    fn expr_blk(expr: @ast::expr) -> ast::blk {
        {node: {view_items: ~[],
                stmts: ~[],
                expr: Some(expr),
                id: self.next_id(),
                rules: ast::default_blk},
         span: expr.span}
    }

    fn binder_pat(span: span, nm: ast::ident) -> @ast::pat {
        let path = @{span: span, global: false, idents: ~[nm],
                     rp: None, types: ~[]};
        @{id: self.next_id(),
          node: ast::pat_ident(ast::bind_by_implicit_ref,
                               path,
                               None),
          span: span}
    }

    fn stmt(expr: @ast::expr) -> @ast::stmt {
        @{node: ast::stmt_semi(expr, self.next_id()),
          span: expr.span}
    }

    fn alt_stmt(arms: ~[ast::arm],
                span: span, -v: @ast::expr) -> @ast::stmt {
        self.stmt(
            self.expr(
                span,
                ast::expr_match(v, arms)))
    }

    fn lit_str(span: span, s: @~str) -> @ast::expr {
        self.expr(
            span,
            ast::expr_vstore(
                self.expr(
                    span,
                    ast::expr_lit(
                        @{node: ast::lit_str(s),
                          span: span})),
                ast::vstore_uniq))
    }

    fn lit_uint(span: span, i: uint) -> @ast::expr {
        self.expr(
            span,
            ast::expr_lit(
                @{node: ast::lit_uint(i as u64, ast::ty_u),
                  span: span}))
    }

    fn lambda(blk: ast::blk) -> @ast::expr {
        let ext_cx = self;
        let blk_e = self.expr(blk.span, ast::expr_block(blk));
        #ast{ || $(blk_e) }
    }

    fn clone_folder() -> fold::ast_fold {
        fold::make_fold(@{
            new_id: |_id| self.next_id(),
            .. *fold::default_ast_fold()
        })
    }

    fn clone(v: @ast::expr) -> @ast::expr {
        let fld = self.clone_folder();
        fld.fold_expr(v)
    }

    fn clone_ty(v: @ast::ty) -> @ast::ty {
        let fld = self.clone_folder();
        fld.fold_ty(v)
    }

    fn clone_ty_param(v: ast::ty_param) -> ast::ty_param {
        let fld = self.clone_folder();
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

        let fld = fold::make_fold(@{
            new_span: |a| repl_sp(a, ast_util::dummy_sp(), span),
            .. *fold::default_ast_fold()
        });

        fld.fold_expr(expr)
    }
}

fn ser_path(cx: ext_ctxt, tps: ser_tps_map, path: @ast::path,
                  -s: @ast::expr, -v: @ast::expr)
    -> ~[@ast::stmt] {
    let ext_cx = cx; // required for #ast{}

    // We want to take a path like a::b::c<...> and generate a call
    // like a::b::c::serialize(s, ...), as described above.

    let callee =
        cx.expr(
            path.span,
            ast::expr_path(
                cx.helper_path(path, ~"serialize")));

    let ty_args = do vec::map(path.types) |ty| {
        let sv_stmts = ser_ty(cx, tps, ty, cx.clone(s), #ast{ __v });
        let sv = cx.expr(path.span,
                         ast::expr_block(cx.blk(path.span, sv_stmts)));
        cx.at(ty.span, #ast{ |__v| $(sv) })
    };

    ~[cx.stmt(
        cx.expr(
            path.span,
            ast::expr_call(callee, vec::append(~[s, v], ty_args), false)))]
}

fn ser_variant(cx: ext_ctxt,
               tps: ser_tps_map,
               tys: ~[@ast::ty],
               span: span,
               -s: @ast::expr,
               pfn: fn(~[@ast::pat]) -> ast::pat_,
               bodyfn: fn(-@ast::expr, ast::blk) -> @ast::expr,
               argfn: fn(-@ast::expr, uint, ast::blk) -> @ast::expr)
    -> ast::arm {
    let vnames = do vec::from_fn(vec::len(tys)) |i| {
        cx.parse_sess().interner.intern(@fmt!("__v%u", i))
    };
    let pats = do vec::from_fn(vec::len(tys)) |i| {
        cx.binder_pat(tys[i].span, vnames[i])
    };
    let pat: @ast::pat = @{id: cx.next_id(), node: pfn(pats), span: span};
    let stmts = do vec::from_fn(vec::len(tys)) |i| {
        let v = cx.var_ref(span, vnames[i]);
        let arg_blk =
            cx.blk(
                span,
                ser_ty(cx, tps, tys[i], cx.clone(s), v));
        cx.stmt(argfn(cx.clone(s), i, arg_blk))
    };

    let body_blk = cx.blk(span, stmts);
    let body = cx.blk(span, ~[cx.stmt(bodyfn(s, body_blk))]);

    {pats: ~[pat], guard: None, body: body}
}

fn ser_lambda(cx: ext_ctxt, tps: ser_tps_map, ty: @ast::ty,
              -s: @ast::expr, -v: @ast::expr) -> @ast::expr {
    cx.lambda(cx.blk(ty.span, ser_ty(cx, tps, ty, s, v)))
}

fn is_vec_or_str(ty: @ast::ty) -> bool {
    match ty.node {
      ast::ty_vec(_) => true,
      // This may be wrong if the user has shadowed (!) str
      ast::ty_path(@{span: _, global: _, idents: ids,
                             rp: None, types: _}, _)
      if ids == ~[parse::token::special_idents::str] => true,
      _ => false
    }
}

fn ser_ty(cx: ext_ctxt, tps: ser_tps_map,
          ty: @ast::ty, -s: @ast::expr, -v: @ast::expr)
    -> ~[@ast::stmt] {

    let ext_cx = cx; // required for #ast{}

    match ty.node {
      ast::ty_nil => {
        ~[#ast[stmt]{$(s).emit_nil()}]
      }

      ast::ty_bot => {
        cx.span_err(
            ty.span, fmt!("Cannot serialize bottom type"));
        ~[]
      }

      ast::ty_box(mt) => {
        let l = ser_lambda(cx, tps, mt.ty, cx.clone(s), #ast{ *$(v) });
        ~[#ast[stmt]{$(s).emit_box($(l));}]
      }

      // For unique evecs/estrs, just pass through to underlying vec or str
      ast::ty_uniq(mt) if is_vec_or_str(mt.ty) => {
        ser_ty(cx, tps, mt.ty, s, v)
      }

      ast::ty_uniq(mt) => {
        let l = ser_lambda(cx, tps, mt.ty, cx.clone(s), #ast{ *$(v) });
        ~[#ast[stmt]{$(s).emit_uniq($(l));}]
      }

      ast::ty_ptr(_) | ast::ty_rptr(_, _) => {
        cx.span_err(ty.span, ~"cannot serialize pointer types");
        ~[]
      }

      ast::ty_rec(flds) => {
        let fld_stmts = do vec::from_fn(vec::len(flds)) |fidx| {
            let fld = flds[fidx];
            let vf = cx.expr(fld.span,
                             ast::expr_field(cx.clone(v),
                                             fld.node.ident,
                                             ~[]));
            let s = cx.clone(s);
            let f = cx.lit_str(fld.span, cx.parse_sess().interner.get(
                fld.node.ident));
            let i = cx.lit_uint(fld.span, fidx);
            let l = ser_lambda(cx, tps, fld.node.mt.ty, cx.clone(s), vf);
            #ast[stmt]{$(s).emit_rec_field($(f), $(i), $(l));}
        };
        let fld_lambda = cx.lambda(cx.blk(ty.span, fld_stmts));
        ~[#ast[stmt]{$(s).emit_rec($(fld_lambda));}]
      }

      ast::ty_fn(*) => {
        cx.span_err(ty.span, ~"cannot serialize function types");
        ~[]
      }

      ast::ty_tup(tys) => {
        // Generate code like
        //
        // match v {
        //    (v1, v2, v3) {
        //       .. serialize v1, v2, v3 ..
        //    }
        // };

        let arms = ~[
            ser_variant(

                cx, tps, tys, ty.span, s,

                // Generate pattern (v1, v2, v3)
                |pats| ast::pat_tup(pats),

                // Generate body s.emit_tup(3, {|| blk })
                |-s, blk| {
                    let sz = cx.lit_uint(ty.span, vec::len(tys));
                    let body = cx.lambda(blk);
                    #ast{ $(s).emit_tup($(sz), $(body)) }
                },

                // Generate s.emit_tup_elt(i, {|| blk })
                |-s, i, blk| {
                    let idx = cx.lit_uint(ty.span, i);
                    let body = cx.lambda(blk);
                    #ast{ $(s).emit_tup_elt($(idx), $(body)) }
                })
        ];
        ~[cx.alt_stmt(arms, ty.span, v)]
      }

      ast::ty_path(path, _) => {
        if vec::len(path.idents) == 1u &&
            vec::is_empty(path.types) {
            let ident = path.idents[0];

            match tps.find(ident) {
              Some(f) => f(v),
              None => ser_path(cx, tps, path, s, v)
            }
        } else {
            ser_path(cx, tps, path, s, v)
        }
      }

      ast::ty_mac(_) => {
        cx.span_err(ty.span, ~"cannot serialize macro types");
        ~[]
      }

      ast::ty_infer => {
        cx.span_err(ty.span, ~"cannot serialize inferred types");
        ~[]
      }

      ast::ty_vec(mt) => {
        let ser_e =
            cx.expr(
                ty.span,
                ast::expr_block(
                    cx.blk(
                        ty.span,
                        ser_ty(
                            cx, tps, mt.ty,
                            cx.clone(s),
                            cx.at(ty.span, #ast{ __e })))));

        ~[#ast[stmt]{
            std::serialization::emit_from_vec($(s), $(v), |__e| $(ser_e))
        }]
      }

      ast::ty_fixed_length(_, _) => {
        cx.span_unimpl(ty.span, ~"serialization for fixed length types");
      }
    }
}

fn mk_ser_fn(cx: ext_ctxt, span: span, name: ast::ident,
             tps: ~[ast::ty_param],
             f: fn(ext_ctxt, ser_tps_map,
                   -@ast::expr, -@ast::expr) -> ~[@ast::stmt])
    -> @ast::item {
    let ext_cx = cx; // required for #ast

    let tp_types = vec::map(tps, |tp| cx.ty_path(span, ~[tp.ident], ~[]));
    let v_ty = cx.ty_path(span, ~[name], tp_types);

    let tp_inputs =
        vec::map(tps, |tp|
            {mode: ast::expl(ast::by_ref),
             ty: cx.ty_fn(span,
                          ~[cx.ty_path(span, ~[tp.ident], ~[])],
                          cx.ty_nil(span)),
             ident: cx.ident_of(~"__s" + cx.str_of(tp.ident)),
             id: cx.next_id()});

    debug!("tp_inputs = %?", tp_inputs);


    let ser_inputs: ~[ast::arg] =
        vec::append(~[{mode: ast::expl(ast::by_ref),
                      ty: cx.ty_path(span, ~[cx.ident_of(~"__S")], ~[]),
                      ident: cx.ident_of(~"__s"),
                      id: cx.next_id()},
                     {mode: ast::expl(ast::by_ref),
                      ty: v_ty,
                      ident: cx.ident_of(~"__v"),
                      id: cx.next_id()}],
                    tp_inputs);

    let tps_map = map::uint_hash();
    do vec::iter2(tps, tp_inputs) |tp, arg| {
        let arg_ident = arg.ident;
        tps_map.insert(
            tp.ident,
            fn@(v: @ast::expr) -> ~[@ast::stmt] {
                let f = cx.var_ref(span, arg_ident);
                debug!("serializing type arg %s", cx.str_of(arg_ident));
                ~[#ast[stmt]{$(f)($(v));}]
            });
    }

    let ser_bnds = @~[
        ast::bound_trait(cx.ty_path(span,
                                    ~[cx.ident_of(~"std"),
                                      cx.ident_of(~"serialization"),
                                      cx.ident_of(~"serializer")],
                                    ~[]))];

    let ser_tps: ~[ast::ty_param] =
        vec::append(~[{ident: cx.ident_of(~"__S"),
                      id: cx.next_id(),
                      bounds: ser_bnds}],
                    vec::map(tps, |tp| cx.clone_ty_param(tp)));

    let ser_output: @ast::ty = @{id: cx.next_id(),
                                 node: ast::ty_nil,
                                 span: span};

    let ser_blk = cx.blk(span,
                         f(cx, tps_map, #ast{ __s }, #ast{ __v }));

    @{ident: cx.ident_of(~"serialize_" + cx.str_of(name)),
      attrs: ~[],
      id: cx.next_id(),
      node: ast::item_fn({inputs: ser_inputs,
                          output: ser_output,
                          cf: ast::return_val},
                         ast::impure_fn,
                         ser_tps,
                         ser_blk),
      vis: ast::public,
      span: span}
}

// ______________________________________________________________________

fn deser_path(cx: ext_ctxt, tps: deser_tps_map, path: @ast::path,
                    -d: @ast::expr) -> @ast::expr {
    // We want to take a path like a::b::c<...> and generate a call
    // like a::b::c::deserialize(d, ...), as described above.

    let callee =
        cx.expr(
            path.span,
            ast::expr_path(
                cx.helper_path(path, ~"deserialize")));

    let ty_args = do vec::map(path.types) |ty| {
        let dv_expr = deser_ty(cx, tps, ty, cx.clone(d));
        cx.lambda(cx.expr_blk(dv_expr))
    };

    cx.expr(path.span, ast::expr_call(callee, vec::append(~[d], ty_args),
                                      false))
}

fn deser_lambda(cx: ext_ctxt, tps: deser_tps_map, ty: @ast::ty,
                -d: @ast::expr) -> @ast::expr {
    cx.lambda(cx.expr_blk(deser_ty(cx, tps, ty, d)))
}

fn deser_ty(cx: ext_ctxt, tps: deser_tps_map,
                  ty: @ast::ty, -d: @ast::expr) -> @ast::expr {

    let ext_cx = cx; // required for #ast{}

    match ty.node {
      ast::ty_nil => {
        #ast{ $(d).read_nil() }
      }

      ast::ty_bot => {
        #ast{ fail }
      }

      ast::ty_box(mt) => {
        let l = deser_lambda(cx, tps, mt.ty, cx.clone(d));
        #ast{ @$(d).read_box($(l)) }
      }

      // For unique evecs/estrs, just pass through to underlying vec or str
      ast::ty_uniq(mt) if is_vec_or_str(mt.ty) => {
        deser_ty(cx, tps, mt.ty, d)
      }

      ast::ty_uniq(mt) => {
        let l = deser_lambda(cx, tps, mt.ty, cx.clone(d));
        #ast{ ~$(d).read_uniq($(l)) }
      }

      ast::ty_ptr(_) | ast::ty_rptr(_, _) => {
        #ast{ fail }
      }

      ast::ty_rec(flds) => {
        let fields = do vec::from_fn(vec::len(flds)) |fidx| {
            let fld = flds[fidx];
            let d = cx.clone(d);
            let f = cx.lit_str(fld.span, @cx.str_of(fld.node.ident));
            let i = cx.lit_uint(fld.span, fidx);
            let l = deser_lambda(cx, tps, fld.node.mt.ty, cx.clone(d));
            {node: {mutbl: fld.node.mt.mutbl,
                    ident: fld.node.ident,
                    expr: #ast{ $(d).read_rec_field($(f), $(i), $(l))} },
             span: fld.span}
        };
        let fld_expr = cx.expr(ty.span, ast::expr_rec(fields, None));
        let fld_lambda = cx.lambda(cx.expr_blk(fld_expr));
        #ast{ $(d).read_rec($(fld_lambda)) }
      }

      ast::ty_fn(*) => {
        #ast{ fail }
      }

      ast::ty_tup(tys) => {
        // Generate code like
        //
        // d.read_tup(3u) {||
        //   (d.read_tup_elt(0u, {||...}),
        //    d.read_tup_elt(1u, {||...}),
        //    d.read_tup_elt(2u, {||...}))
        // }

        let arg_exprs = do vec::from_fn(vec::len(tys)) |i| {
            let idx = cx.lit_uint(ty.span, i);
            let body = deser_lambda(cx, tps, tys[i], cx.clone(d));
            #ast{ $(d).read_tup_elt($(idx), $(body)) }
        };
        let body =
            cx.lambda(cx.expr_blk(
                cx.expr(ty.span, ast::expr_tup(arg_exprs))));
        let sz = cx.lit_uint(ty.span, vec::len(tys));
        #ast{ $(d).read_tup($(sz), $(body)) }
      }

      ast::ty_path(path, _) => {
        if vec::len(path.idents) == 1u &&
            vec::is_empty(path.types) {
            let ident = path.idents[0];

            match tps.find(ident) {
              Some(f) => f(),
              None => deser_path(cx, tps, path, d)
            }
        } else {
            deser_path(cx, tps, path, d)
        }
      }

      ast::ty_mac(_) => {
        #ast{ fail }
      }

      ast::ty_infer => {
        #ast{ fail }
      }

      ast::ty_vec(mt) => {
        let l = deser_lambda(cx, tps, mt.ty, cx.clone(d));
        #ast{ std::serialization::read_to_vec($(d), $(l)) }
      }

      ast::ty_fixed_length(_, _) => {
        cx.span_unimpl(ty.span, ~"deserialization for fixed length types");
      }
    }
}

fn mk_deser_fn(cx: ext_ctxt, span: span,
               name: ast::ident, tps: ~[ast::ty_param],
               f: fn(ext_ctxt, deser_tps_map, -@ast::expr) -> @ast::expr)
    -> @ast::item {
    let ext_cx = cx; // required for #ast

    let tp_types = vec::map(tps, |tp| cx.ty_path(span, ~[tp.ident], ~[]));
    let v_ty = cx.ty_path(span, ~[name], tp_types);

    let tp_inputs =
        vec::map(tps, |tp|
            {mode: ast::expl(ast::by_ref),
             ty: cx.ty_fn(span,
                          ~[],
                          cx.ty_path(span, ~[tp.ident], ~[])),
             ident: cx.ident_of(~"__d" + cx.str_of(tp.ident)),
             id: cx.next_id()});

    debug!("tp_inputs = %?", tp_inputs);

    let deser_inputs: ~[ast::arg] =
        vec::append(~[{mode: ast::expl(ast::by_ref),
                      ty: cx.ty_path(span, ~[cx.ident_of(~"__D")], ~[]),
                      ident: cx.ident_of(~"__d"),
                      id: cx.next_id()}],
                    tp_inputs);

    let tps_map = map::uint_hash();
    do vec::iter2(tps, tp_inputs) |tp, arg| {
        let arg_ident = arg.ident;
        tps_map.insert(
            tp.ident,
            fn@() -> @ast::expr {
                let f = cx.var_ref(span, arg_ident);
                #ast{ $(f)() }
            });
    }

    let deser_bnds = @~[
        ast::bound_trait(cx.ty_path(
            span,
            ~[cx.ident_of(~"std"), cx.ident_of(~"serialization"),
              cx.ident_of(~"deserializer")],
            ~[]))];

    let deser_tps: ~[ast::ty_param] =
        vec::append(~[{ident: cx.ident_of(~"__D"),
                      id: cx.next_id(),
                      bounds: deser_bnds}],
                    vec::map(tps, |tp| {
                        let cloned = cx.clone_ty_param(tp);
                        {bounds: @(vec::append(*cloned.bounds,
                                               ~[ast::bound_copy])),
                         .. cloned}
                    }));

    let deser_blk = cx.expr_blk(f(cx, tps_map, #ast[expr]{__d}));

    @{ident: cx.ident_of(~"deserialize_" + cx.str_of(name)),
      attrs: ~[],
      id: cx.next_id(),
      node: ast::item_fn({inputs: deser_inputs,
                          output: v_ty,
                          cf: ast::return_val},
                         ast::impure_fn,
                         deser_tps,
                         deser_blk),
      vis: ast::public,
      span: span}
}

fn ty_fns(cx: ext_ctxt, name: ast::ident,
          ty: @ast::ty, tps: ~[ast::ty_param])
    -> ~[@ast::item] {

    let span = ty.span;
    ~[
        mk_ser_fn(cx, span, name, tps, |a,b,c,d| ser_ty(a, b, ty, c, d)),
        mk_deser_fn(cx, span, name, tps, |a,b,c| deser_ty(a, b, ty, c))
    ]
}

fn ser_enum(cx: ext_ctxt, tps: ser_tps_map, e_name: ast::ident,
            e_span: span, variants: ~[ast::variant],
            -s: @ast::expr, -v: @ast::expr) -> ~[@ast::stmt] {
    let ext_cx = cx;
    let arms = do vec::from_fn(vec::len(variants)) |vidx| {
        let variant = variants[vidx];
        let v_span = variant.span;
        let v_name = variant.node.name;

        match variant.node.kind {
            ast::tuple_variant_kind(args) => {
                let variant_tys = vec::map(args, |a| a.ty);

                ser_variant(
                    cx, tps, variant_tys, v_span, cx.clone(s),

                    // Generate pattern var(v1, v2, v3)
                    |pats| {
                        if vec::is_empty(pats) {
                            ast::pat_ident(ast::bind_by_implicit_ref,
                                           cx.path(v_span, ~[v_name]),
                                           None)
                        } else {
                            ast::pat_enum(cx.path(v_span, ~[v_name]),
                                                  Some(pats))
                        }
                    },

                    // Generate body s.emit_enum_variant("foo", 0u,
                    //                                   3u, {|| blk })
                    |-s, blk| {
                        let v_name = cx.lit_str(v_span, @cx.str_of(v_name));
                        let v_id = cx.lit_uint(v_span, vidx);
                        let sz = cx.lit_uint(v_span, vec::len(variant_tys));
                        let body = cx.lambda(blk);
                        #ast[expr]{
                            $(s).emit_enum_variant($(v_name), $(v_id),
                                                   $(sz), $(body))
                        }
                    },

                    // Generate s.emit_enum_variant_arg(i, {|| blk })
                    |-s, i, blk| {
                        let idx = cx.lit_uint(v_span, i);
                        let body = cx.lambda(blk);
                        #ast[expr]{
                            $(s).emit_enum_variant_arg($(idx), $(body))
                        }
                    })
            }
            _ =>
                fail ~"struct variants unimplemented for auto serialize"
        }
    };
    let lam = cx.lambda(cx.blk(e_span, ~[cx.alt_stmt(arms, e_span, v)]));
    let e_name = cx.lit_str(e_span, @cx.str_of(e_name));
    ~[#ast[stmt]{ $(s).emit_enum($(e_name), $(lam)) }]
}

fn deser_enum(cx: ext_ctxt, tps: deser_tps_map, e_name: ast::ident,
              e_span: span, variants: ~[ast::variant],
              -d: @ast::expr) -> @ast::expr {
    let ext_cx = cx;
    let mut arms: ~[ast::arm] = do vec::from_fn(vec::len(variants)) |vidx| {
        let variant = variants[vidx];
        let v_span = variant.span;
        let v_name = variant.node.name;

        let body;
        match variant.node.kind {
            ast::tuple_variant_kind(args) => {
                let tys = vec::map(args, |a| a.ty);

                let arg_exprs = do vec::from_fn(vec::len(tys)) |i| {
                    let idx = cx.lit_uint(v_span, i);
                    let body = deser_lambda(cx, tps, tys[i], cx.clone(d));
                    #ast{ $(d).read_enum_variant_arg($(idx), $(body)) }
                };

                body = {
                    if vec::is_empty(tys) {
                        // for a nullary variant v, do "v"
                        cx.var_ref(v_span, v_name)
                    } else {
                        // for an n-ary variant v, do "v(a_1, ..., a_n)"
                        cx.expr(v_span, ast::expr_call(
                            cx.var_ref(v_span, v_name), arg_exprs, false))
                    }
                };
            }
            ast::struct_variant_kind(*) =>
                fail ~"struct variants unimplemented",
            ast::enum_variant_kind(*) =>
                fail ~"enum variants unimplemented"
        }

        {pats: ~[@{id: cx.next_id(),
                  node: ast::pat_lit(cx.lit_uint(v_span, vidx)),
                  span: v_span}],
         guard: None,
         body: cx.expr_blk(body)}
    };

    let impossible_case = {pats: ~[@{id: cx.next_id(),
                                     node: ast::pat_wild,
                                     span: e_span}],
                        guard: None,
                        // FIXME #3198: proper error message
                           body: cx.expr_blk(cx.expr(e_span,
                                                     ast::expr_fail(None)))};
    arms += ~[impossible_case];

    // Generate code like:
    let e_name = cx.lit_str(e_span, @cx.str_of(e_name));
    let alt_expr = cx.expr(e_span,
                           ast::expr_match(#ast{__i}, arms));
    let var_lambda = #ast{ |__i| $(alt_expr) };
    let read_var = #ast{ $(cx.clone(d)).read_enum_variant($(var_lambda)) };
    let read_lambda = cx.lambda(cx.expr_blk(read_var));
    #ast{ $(d).read_enum($(e_name), $(read_lambda)) }
}

fn enum_fns(cx: ext_ctxt, e_name: ast::ident, e_span: span,
               variants: ~[ast::variant], tps: ~[ast::ty_param])
    -> ~[@ast::item] {
    ~[
        mk_ser_fn(cx, e_span, e_name, tps,
                  |a,b,c,d| ser_enum(a, b, e_name, e_span, variants, c, d)),
        mk_deser_fn(cx, e_span, e_name, tps,
                    |a,b,c| deser_enum(a, b, e_name, e_span, variants, c))
    ]
}
