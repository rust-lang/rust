
use ast::*;
use codemap::span;

// Context-passing AST walker. Each overridden visit method has full control
// over what happens with its node, it can do its own traversal of the node's
// children (potentially passing in different contexts to each), call
// visit::visit_* to apply the default traversal algorithm (again, it can
// override the context), or prevent deeper traversal by doing nothing.

// Our typesystem doesn't do circular types, so the visitor record can not
// hold functions that take visitors. A vt enum is used to break the cycle.
enum vt<E> { mk_vt(visitor<E>), }

enum fn_kind {
    fk_item_fn(ident, ~[ty_param], purity), //< an item declared with fn()
    fk_method(ident, ~[ty_param], @method),
    fk_anon(proto, capture_clause),  //< an anonymous function like fn@(...)
    fk_fn_block(capture_clause),     //< a block {||...}
    fk_ctor(ident, ~[attribute], ~[ty_param], node_id /* self id */,
            def_id /* parent class id */), // class constructor
    fk_dtor(~[ty_param], ~[attribute], node_id /* self id */,
            def_id /* parent class id */) // class destructor

}

fn name_of_fn(fk: fn_kind) -> ident {
    match fk {
      fk_item_fn(name, _, _) | fk_method(name, _, _)
          | fk_ctor(name, _, _, _, _) =>  /* FIXME (#2543) */ copy name,
      fk_anon(*) | fk_fn_block(*) => parse::token::special_idents::anon,
      fk_dtor(*)                  => parse::token::special_idents::dtor
    }
}

fn tps_of_fn(fk: fn_kind) -> ~[ty_param] {
    match fk {
      fk_item_fn(_, tps, _) | fk_method(_, tps, _)
          | fk_ctor(_, _, tps, _, _) | fk_dtor(tps, _, _, _) => {
          /* FIXME (#2543) */ copy tps
      }
      fk_anon(*) | fk_fn_block(*) => ~[]
    }
}

type visitor<E> =
    @{visit_mod: fn@(_mod, span, node_id, E, vt<E>),
      visit_view_item: fn@(@view_item, E, vt<E>),
      visit_foreign_item: fn@(@foreign_item, E, vt<E>),
      visit_item: fn@(@item, E, vt<E>),
      visit_local: fn@(@local, E, vt<E>),
      visit_block: fn@(ast::blk, E, vt<E>),
      visit_stmt: fn@(@stmt, E, vt<E>),
      visit_arm: fn@(arm, E, vt<E>),
      visit_pat: fn@(@pat, E, vt<E>),
      visit_decl: fn@(@decl, E, vt<E>),
      visit_expr: fn@(@expr, E, vt<E>),
      visit_expr_post: fn@(@expr, E, vt<E>),
      visit_ty: fn@(@ty, E, vt<E>),
      visit_ty_params: fn@(~[ty_param], E, vt<E>),
      visit_fn: fn@(fn_kind, fn_decl, blk, span, node_id, E, vt<E>),
      visit_ty_method: fn@(ty_method, E, vt<E>),
      visit_trait_method: fn@(trait_method, E, vt<E>),
      visit_struct_def: fn@(@struct_def, ident, ~[ty_param], node_id, E,
                            vt<E>),
      visit_struct_field: fn@(@struct_field, E, vt<E>),
      visit_struct_method: fn@(@method, E, vt<E>)};

fn default_visitor<E>() -> visitor<E> {
    return @{visit_mod: |a,b,c,d,e|visit_mod::<E>(a, b, c, d, e),
          visit_view_item: |a,b,c|visit_view_item::<E>(a, b, c),
          visit_foreign_item: |a,b,c|visit_foreign_item::<E>(a, b, c),
          visit_item: |a,b,c|visit_item::<E>(a, b, c),
          visit_local: |a,b,c|visit_local::<E>(a, b, c),
          visit_block: |a,b,c|visit_block::<E>(a, b, c),
          visit_stmt: |a,b,c|visit_stmt::<E>(a, b, c),
          visit_arm: |a,b,c|visit_arm::<E>(a, b, c),
          visit_pat: |a,b,c|visit_pat::<E>(a, b, c),
          visit_decl: |a,b,c|visit_decl::<E>(a, b, c),
          visit_expr: |a,b,c|visit_expr::<E>(a, b, c),
          visit_expr_post: |_a,_b,_c| (),
          visit_ty: |a,b,c|skip_ty::<E>(a, b, c),
          visit_ty_params: |a,b,c|visit_ty_params::<E>(a, b, c),
          visit_fn: |a,b,c,d,e,f,g|visit_fn::<E>(a, b, c, d, e, f, g),
          visit_ty_method: |a,b,c|visit_ty_method::<E>(a, b, c),
          visit_trait_method: |a,b,c|visit_trait_method::<E>(a, b, c),
          visit_struct_def: |a,b,c,d,e,f|visit_struct_def::<E>(a, b, c,
                                                               d, e, f),
          visit_struct_field: |a,b,c|visit_struct_field::<E>(a, b, c),
          visit_struct_method: |a,b,c|visit_struct_method::<E>(a, b, c)};
}

fn visit_crate<E>(c: crate, e: E, v: vt<E>) {
    v.visit_mod(c.node.module, c.span, crate_node_id, e, v);
}

fn visit_crate_directive<E>(cd: @crate_directive, e: E, v: vt<E>) {
    match cd.node {
      cdir_src_mod(_, _, _) => (),
      cdir_dir_mod(_, _, cdirs, _) => for cdirs.each |cdir| {
        visit_crate_directive(*cdir, e, v);
      },
      cdir_view_item(vi) => v.visit_view_item(vi, e, v),
      cdir_syntax(_) => ()
    }
}

fn visit_mod<E>(m: _mod, _sp: span, _id: node_id, e: E, v: vt<E>) {
    for m.view_items.each |vi| { v.visit_view_item(*vi, e, v); }
    for m.items.each |i| { v.visit_item(*i, e, v); }
}

fn visit_view_item<E>(_vi: @view_item, _e: E, _v: vt<E>) { }

fn visit_local<E>(loc: @local, e: E, v: vt<E>) {
    v.visit_pat(loc.node.pat, e, v);
    v.visit_ty(loc.node.ty, e, v);
    match loc.node.init {
      None => (),
      Some(i) => v.visit_expr(i.expr, e, v)
    }
}

fn visit_item<E>(i: @item, e: E, v: vt<E>) {
    match i.node {
      item_const(t, ex) => { v.visit_ty(t, e, v); v.visit_expr(ex, e, v); }
      item_fn(decl, purity, tp, body) => {
        v.visit_fn(fk_item_fn(/* FIXME (#2543) */ copy i.ident,
                              /* FIXME (#2543) */ copy tp,
                              purity), decl, body,
                   i.span, i.id, e, v);
      }
      item_mod(m) => v.visit_mod(m, i.span, i.id, e, v),
      item_foreign_mod(nm) => {
        for nm.view_items.each |vi| { v.visit_view_item(*vi, e, v); }
        for nm.items.each |ni| { v.visit_foreign_item(*ni, e, v); }
      }
      item_ty(t, tps) => {
        v.visit_ty(t, e, v);
        v.visit_ty_params(tps, e, v);
      }
      item_enum(enum_definition, tps) => {
        v.visit_ty_params(tps, e, v);
        visit_enum_def(enum_definition, tps, e, v);
      }
      item_impl(tps, traits, ty, methods) => {
        v.visit_ty_params(tps, e, v);
        for traits.each |p| {
            visit_path(p.path, e, v);
        }
        v.visit_ty(ty, e, v);
        for methods.each |m| {
            visit_method_helper(*m, e, v)
        }
      }
      item_class(struct_def, tps) => {
        v.visit_ty_params(tps, e, v);
        v.visit_struct_def(struct_def, i.ident, tps, i.id, e, v);
      }
      item_trait(tps, traits, methods) => {
        v.visit_ty_params(tps, e, v);
        for traits.each |p| { visit_path(p.path, e, v); }
        for methods.each |m| {
            v.visit_trait_method(*m, e, v);
        }
      }
      item_mac(m) => visit_mac(m, e, v)
    }
}

fn visit_enum_def<E>(enum_definition: ast::enum_def, tps: ~[ast::ty_param],
                     e: E, v: vt<E>) {
    for enum_definition.variants.each |vr| {
        match vr.node.kind {
            tuple_variant_kind(variant_args) => {
                for variant_args.each |va| { v.visit_ty(va.ty, e, v); }
            }
            struct_variant_kind(struct_def) => {
                v.visit_struct_def(struct_def, vr.node.name, tps,
                                   vr.node.id, e, v);
            }
            enum_variant_kind(enum_definition) => {
                visit_enum_def(enum_definition, tps, e, v);
            }
        }
    }
}

fn skip_ty<E>(_t: @ty, _e: E, _v: vt<E>) {}

fn visit_ty<E>(t: @ty, e: E, v: vt<E>) {
    match t.node {
      ty_box(mt) | ty_uniq(mt) |
      ty_vec(mt) | ty_ptr(mt) | ty_rptr(_, mt) => {
        v.visit_ty(mt.ty, e, v);
      }
      ty_rec(flds) => for flds.each |f| {
        v.visit_ty(f.node.mt.ty, e, v);
      },
      ty_tup(ts) => for ts.each |tt| {
        v.visit_ty(*tt, e, v);
      },
      ty_fn(_, _, bounds, decl) => {
        for decl.inputs.each |a| { v.visit_ty(a.ty, e, v); }
        visit_ty_param_bounds(bounds, e, v);
        v.visit_ty(decl.output, e, v);
      }
      ty_path(p, _) => visit_path(p, e, v),
      ty_fixed_length(t, _) => v.visit_ty(t, e, v),
      ty_nil |
      ty_bot |
      ty_mac(_) |
      ty_infer => ()
    }
}

fn visit_path<E>(p: @path, e: E, v: vt<E>) {
    for p.types.each |tp| { v.visit_ty(*tp, e, v); }
}

fn visit_pat<E>(p: @pat, e: E, v: vt<E>) {
    match p.node {
      pat_enum(path, children) => {
        visit_path(path, e, v);
        do option::iter(&children) |children| {
            for children.each |child| { v.visit_pat(*child, e, v); }}
      }
      pat_rec(fields, _) => for fields.each |f| {
        v.visit_pat(f.pat, e, v)
      },
      pat_struct(path, fields, _) => {
        visit_path(path, e, v);
        for fields.each |f| {
            v.visit_pat(f.pat, e, v);
        }
      }
      pat_tup(elts) => for elts.each |elt| {
        v.visit_pat(*elt, e, v)
      },
      pat_box(inner) | pat_uniq(inner) | pat_region(inner) =>
          v.visit_pat(inner, e, v),
      pat_ident(_, path, inner) => {
          visit_path(path, e, v);
          do option::iter(&inner) |subpat| { v.visit_pat(subpat, e, v)};
      }
      pat_lit(ex) => v.visit_expr(ex, e, v),
      pat_range(e1, e2) => { v.visit_expr(e1, e, v); v.visit_expr(e2, e, v); }
      pat_wild => ()
    }
}

fn visit_foreign_item<E>(ni: @foreign_item, e: E, v: vt<E>) {
    match ni.node {
      foreign_item_fn(fd, _, tps) => {
        v.visit_ty_params(tps, e, v);
        visit_fn_decl(fd, e, v);
      }
      foreign_item_const(t) => {
        v.visit_ty(t, e, v);
      }
    }
}

fn visit_ty_param_bounds<E>(bounds: @~[ty_param_bound], e: E, v: vt<E>) {
    for vec::each(*bounds) |bound| {
        match *bound {
          bound_trait(t) => v.visit_ty(t, e, v),
          bound_copy | bound_send | bound_const | bound_owned => ()
        }
    }
}

fn visit_ty_params<E>(tps: ~[ty_param], e: E, v: vt<E>) {
    for tps.each |tp| {
        visit_ty_param_bounds(tp.bounds, e, v);
    }
}

fn visit_fn_decl<E>(fd: fn_decl, e: E, v: vt<E>) {
    for fd.inputs.each |a| { v.visit_ty(a.ty, e, v); }
    v.visit_ty(fd.output, e, v);
}

// Note: there is no visit_method() method in the visitor, instead override
// visit_fn() and check for fk_method().  I named this visit_method_helper()
// because it is not a default impl of any method, though I doubt that really
// clarifies anything. - Niko
fn visit_method_helper<E>(m: @method, e: E, v: vt<E>) {
    v.visit_fn(fk_method(/* FIXME (#2543) */ copy m.ident,
                         /* FIXME (#2543) */ copy m.tps, m),
               m.decl, m.body, m.span, m.id, e, v);
}

// Similar logic to the comment on visit_method_helper - Tim
fn visit_class_ctor_helper<E>(ctor: class_ctor, nm: ident, tps: ~[ty_param],
                              parent_id: def_id, e: E, v: vt<E>) {
    v.visit_fn(fk_ctor(/* FIXME (#2543) */ copy nm,
                       ctor.node.attrs,
                       /* FIXME (#2543) */ copy tps,
                       ctor.node.self_id, parent_id),
        ctor.node.dec, ctor.node.body, ctor.span, ctor.node.id, e, v)

}

fn visit_class_dtor_helper<E>(dtor: class_dtor, tps: ~[ty_param],
                              parent_id: def_id, e: E, v: vt<E>) {
    v.visit_fn(fk_dtor(/* FIXME (#2543) */ copy tps, dtor.node.attrs,
                       dtor.node.self_id, parent_id), ast_util::dtor_dec(),
               dtor.node.body, dtor.span, dtor.node.id, e, v)

}

fn visit_fn<E>(fk: fn_kind, decl: fn_decl, body: blk, _sp: span,
               _id: node_id, e: E, v: vt<E>) {
    visit_fn_decl(decl, e, v);
    v.visit_ty_params(tps_of_fn(fk), e, v);
    v.visit_block(body, e, v);
}

fn visit_ty_method<E>(m: ty_method, e: E, v: vt<E>) {
    for m.decl.inputs.each |a| { v.visit_ty(a.ty, e, v); }
    v.visit_ty_params(m.tps, e, v);
    v.visit_ty(m.decl.output, e, v);
}

fn visit_trait_method<E>(m: trait_method, e: E, v: vt<E>) {
    match m {
      required(ty_m) => v.visit_ty_method(ty_m, e, v),
      provided(m) => visit_method_helper(m, e, v)
    }
}

fn visit_struct_def<E>(sd: @struct_def, nm: ast::ident, tps: ~[ty_param],
                       id: node_id, e: E, v: vt<E>) {
    for sd.fields.each |f| {
        v.visit_struct_field(*f, e, v);
    }
    for sd.methods.each |m| {
        v.visit_struct_method(*m, e, v);
    }
    for sd.traits.each |p| {
        visit_path(p.path, e, v);
    }
    do option::iter(&sd.ctor) |ctor| {
      visit_class_ctor_helper(ctor, nm, tps, ast_util::local_def(id), e, v);
    };
    do option::iter(&sd.dtor) |dtor| {
      visit_class_dtor_helper(dtor, tps, ast_util::local_def(id), e, v)
    };
}

fn visit_struct_field<E>(sf: @struct_field, e: E, v: vt<E>) {
    v.visit_ty(sf.node.ty, e, v);
}

fn visit_struct_method<E>(m: @method, e: E, v: vt<E>) {
    visit_method_helper(m, e, v);
}

fn visit_block<E>(b: ast::blk, e: E, v: vt<E>) {
    for b.node.view_items.each |vi| {
        v.visit_view_item(*vi, e, v);
    }
    for b.node.stmts.each |s| {
        v.visit_stmt(*s, e, v);
    }
    visit_expr_opt(b.node.expr, e, v);
}

fn visit_stmt<E>(s: @stmt, e: E, v: vt<E>) {
    match s.node {
      stmt_decl(d, _) => v.visit_decl(d, e, v),
      stmt_expr(ex, _) => v.visit_expr(ex, e, v),
      stmt_semi(ex, _) => v.visit_expr(ex, e, v)
    }
}

fn visit_decl<E>(d: @decl, e: E, v: vt<E>) {
    match d.node {
      decl_local(locs) => for locs.each |loc| {
        v.visit_local(*loc, e, v)
      },
      decl_item(it) => v.visit_item(it, e, v)
    }
}

fn visit_expr_opt<E>(eo: Option<@expr>, e: E, v: vt<E>) {
    match eo { None => (), Some(ex) => v.visit_expr(ex, e, v) }
}

fn visit_exprs<E>(exprs: ~[@expr], e: E, v: vt<E>) {
    for exprs.each |ex| { v.visit_expr(*ex, e, v); }
}

fn visit_mac<E>(m: mac, e: E, v: vt<E>) {
    match m.node {
      ast::mac_invoc(_, arg, _) => {
        option::map(&arg, |arg| v.visit_expr(*arg, e, v)); }
      ast::mac_invoc_tt(*) => { /* no user-serviceable parts inside */ }
      ast::mac_ellipsis => (),
      ast::mac_aq(*) => { /* FIXME: maybe visit (Issue #2340) */ }
      ast::mac_var(_) => ()
    }
}

fn visit_expr<E>(ex: @expr, e: E, v: vt<E>) {
    match ex.node {
      expr_vstore(x, _) => v.visit_expr(x, e, v),
      expr_vec(es, _) => visit_exprs(es, e, v),
      expr_repeat(element, count, _) => {
        v.visit_expr(element, e, v);
        v.visit_expr(count, e, v);
      }
      expr_rec(flds, base) => {
        for flds.each |f| { v.visit_expr(f.node.expr, e, v); }
        visit_expr_opt(base, e, v);
      }
      expr_struct(p, flds, base) => {
        visit_path(p, e, v);
        for flds.each |f| { v.visit_expr(f.node.expr, e, v); }
        visit_expr_opt(base, e, v);
      }
      expr_tup(elts) => for elts.each |el| { v.visit_expr(*el, e, v); },
      expr_call(callee, args, _) => {
        visit_exprs(args, e, v);
        v.visit_expr(callee, e, v);
      }
      expr_binary(_, a, b) => {
        v.visit_expr(a, e, v); v.visit_expr(b, e, v);
      }
      expr_addr_of(_, x) | expr_unary(_, x) |
      expr_loop_body(x) | expr_do_body(x) |
      expr_assert(x) => v.visit_expr(x, e, v),
      expr_lit(_) => (),
      expr_cast(x, t) => { v.visit_expr(x, e, v); v.visit_ty(t, e, v); }
      expr_if(x, b, eo) => {
        v.visit_expr(x, e, v);
        v.visit_block(b, e, v);
        visit_expr_opt(eo, e, v);
      }
      expr_while(x, b) => { v.visit_expr(x, e, v); v.visit_block(b, e, v); }
      expr_loop(b, _) => v.visit_block(b, e, v),
      expr_match(x, arms) => {
        v.visit_expr(x, e, v);
        for arms.each |a| { v.visit_arm(*a, e, v); }
      }
      expr_fn(proto, decl, body, cap_clause) => {
        v.visit_fn(fk_anon(proto, cap_clause), decl, body,
                   ex.span, ex.id, e, v);
      }
      expr_fn_block(decl, body, cap_clause) => {
        v.visit_fn(fk_fn_block(cap_clause), decl, body,
                   ex.span, ex.id, e, v);
      }
      expr_block(b) => v.visit_block(b, e, v),
      expr_assign(a, b) => { v.visit_expr(b, e, v); v.visit_expr(a, e, v); }
      expr_copy(a) => v.visit_expr(a, e, v),
      expr_unary_move(a) => v.visit_expr(a, e, v),
      expr_move(a, b) => { v.visit_expr(b, e, v); v.visit_expr(a, e, v); }
      expr_swap(a, b) => { v.visit_expr(a, e, v); v.visit_expr(b, e, v); }
      expr_assign_op(_, a, b) => {
        v.visit_expr(b, e, v);
        v.visit_expr(a, e, v);
      }
      expr_field(x, _, tys) => {
        v.visit_expr(x, e, v);
        for tys.each |tp| { v.visit_ty(*tp, e, v); }
      }
      expr_index(a, b) => { v.visit_expr(a, e, v); v.visit_expr(b, e, v); }
      expr_path(p) => visit_path(p, e, v),
      expr_fail(eo) => visit_expr_opt(eo, e, v),
      expr_break(_) => (),
      expr_again(_) => (),
      expr_ret(eo) => visit_expr_opt(eo, e, v),
      expr_log(_, lv, x) => {
        v.visit_expr(lv, e, v);
        v.visit_expr(x, e, v);
      }
      expr_mac(mac) => visit_mac(mac, e, v),
    }
    v.visit_expr_post(ex, e, v);
}

fn visit_arm<E>(a: arm, e: E, v: vt<E>) {
    for a.pats.each |p| { v.visit_pat(*p, e, v); }
    visit_expr_opt(a.guard, e, v);
    v.visit_block(a.body, e, v);
}

// Simpler, non-context passing interface. Always walks the whole tree, simply
// calls the given functions on the nodes.

type simple_visitor =
    @{visit_mod: fn@(_mod, span, node_id),
      visit_view_item: fn@(@view_item),
      visit_foreign_item: fn@(@foreign_item),
      visit_item: fn@(@item),
      visit_local: fn@(@local),
      visit_block: fn@(ast::blk),
      visit_stmt: fn@(@stmt),
      visit_arm: fn@(arm),
      visit_pat: fn@(@pat),
      visit_decl: fn@(@decl),
      visit_expr: fn@(@expr),
      visit_expr_post: fn@(@expr),
      visit_ty: fn@(@ty),
      visit_ty_params: fn@(~[ty_param]),
      visit_fn: fn@(fn_kind, fn_decl, blk, span, node_id),
      visit_ty_method: fn@(ty_method),
      visit_trait_method: fn@(trait_method),
      visit_struct_def: fn@(@struct_def, ident, ~[ty_param], node_id),
      visit_struct_field: fn@(@struct_field),
      visit_struct_method: fn@(@method)};

fn simple_ignore_ty(_t: @ty) {}

fn default_simple_visitor() -> simple_visitor {
    return @{visit_mod: fn@(_m: _mod, _sp: span, _id: node_id) { },
          visit_view_item: fn@(_vi: @view_item) { },
          visit_foreign_item: fn@(_ni: @foreign_item) { },
          visit_item: fn@(_i: @item) { },
          visit_local: fn@(_l: @local) { },
          visit_block: fn@(_b: ast::blk) { },
          visit_stmt: fn@(_s: @stmt) { },
          visit_arm: fn@(_a: arm) { },
          visit_pat: fn@(_p: @pat) { },
          visit_decl: fn@(_d: @decl) { },
          visit_expr: fn@(_e: @expr) { },
          visit_expr_post: fn@(_e: @expr) { },
          visit_ty: simple_ignore_ty,
          visit_ty_params: fn@(_ps: ~[ty_param]) {},
          visit_fn: fn@(_fk: fn_kind, _d: fn_decl, _b: blk, _sp: span,
                        _id: node_id) { },
          visit_ty_method: fn@(_m: ty_method) { },
          visit_trait_method: fn@(_m: trait_method) { },
          visit_struct_def: fn@(_sd: @struct_def, _nm: ident,
                                _tps: ~[ty_param], _id: node_id) { },
          visit_struct_field: fn@(_f: @struct_field) { },
          visit_struct_method: fn@(_m: @method) { }
         };
}

fn mk_simple_visitor(v: simple_visitor) -> vt<()> {
    fn v_mod(f: fn@(_mod, span, node_id), m: _mod, sp: span, id: node_id,
             &&e: (), v: vt<()>) {
        f(m, sp, id);
        visit_mod(m, sp, id, e, v);
    }
    fn v_view_item(f: fn@(@view_item), vi: @view_item, &&e: (), v: vt<()>) {
        f(vi);
        visit_view_item(vi, e, v);
    }
    fn v_foreign_item(f: fn@(@foreign_item), ni: @foreign_item, &&e: (),
                     v: vt<()>) {
        f(ni);
        visit_foreign_item(ni, e, v);
    }
    fn v_item(f: fn@(@item), i: @item, &&e: (), v: vt<()>) {
        f(i);
        visit_item(i, e, v);
    }
    fn v_local(f: fn@(@local), l: @local, &&e: (), v: vt<()>) {
        f(l);
        visit_local(l, e, v);
    }
    fn v_block(f: fn@(ast::blk), bl: ast::blk, &&e: (), v: vt<()>) {
        f(bl);
        visit_block(bl, e, v);
    }
    fn v_stmt(f: fn@(@stmt), st: @stmt, &&e: (), v: vt<()>) {
        f(st);
        visit_stmt(st, e, v);
    }
    fn v_arm(f: fn@(arm), a: arm, &&e: (), v: vt<()>) {
        f(a);
        visit_arm(a, e, v);
    }
    fn v_pat(f: fn@(@pat), p: @pat, &&e: (), v: vt<()>) {
        f(p);
        visit_pat(p, e, v);
    }
    fn v_decl(f: fn@(@decl), d: @decl, &&e: (), v: vt<()>) {
        f(d);
        visit_decl(d, e, v);
    }
    fn v_expr(f: fn@(@expr), ex: @expr, &&e: (), v: vt<()>) {
        f(ex);
        visit_expr(ex, e, v);
    }
    fn v_expr_post(f: fn@(@expr), ex: @expr, &&_e: (), _v: vt<()>) {
        f(ex);
    }
    fn v_ty(f: fn@(@ty), ty: @ty, &&e: (), v: vt<()>) {
        f(ty);
        visit_ty(ty, e, v);
    }
    fn v_ty_method(f: fn@(ty_method), ty: ty_method, &&e: (), v: vt<()>) {
        f(ty);
        visit_ty_method(ty, e, v);
    }
    fn v_trait_method(f: fn@(trait_method), m: trait_method, &&e: (),
                      v: vt<()>) {
        f(m);
        visit_trait_method(m, e, v);
    }
    fn v_struct_def(f: fn@(@struct_def, ident, ~[ty_param], node_id),
                    sd: @struct_def, nm: ident, tps: ~[ty_param], id: node_id,
                    &&e: (), v: vt<()>) {
        f(sd, nm, tps, id);
        visit_struct_def(sd, nm, tps, id, e, v);
    }
    fn v_ty_params(f: fn@(~[ty_param]),
                   ps: ~[ty_param],
                   &&e: (), v: vt<()>) {
        f(ps);
        visit_ty_params(ps, e, v);
    }
    fn v_fn(f: fn@(fn_kind, fn_decl, blk, span, node_id),
            fk: fn_kind, decl: fn_decl, body: blk, sp: span,
            id: node_id, &&e: (), v: vt<()>) {
        f(fk, decl, body, sp, id);
        visit_fn(fk, decl, body, sp, id, e, v);
    }
    let visit_ty = |a,b,c| v_ty(v.visit_ty, a, b, c);
    fn v_struct_field(f: fn@(@struct_field), sf: @struct_field, &&e: (),
                      v: vt<()>) {
        f(sf);
        visit_struct_field(sf, e, v);
    }
    fn v_struct_method(f: fn@(@method), m: @method, &&e: (), v: vt<()>) {
        f(m);
        visit_struct_method(m, e, v);
    }
    return mk_vt(@{visit_mod: |a,b,c,d,e|v_mod(v.visit_mod, a, b, c, d, e),
                visit_view_item: |a,b,c|
                    v_view_item(v.visit_view_item, a, b, c),
                visit_foreign_item:
                    |a,b,c|v_foreign_item(v.visit_foreign_item, a, b, c),
                visit_item: |a,b,c|v_item(v.visit_item, a, b, c),
                visit_local: |a,b,c|v_local(v.visit_local, a, b, c),
                visit_block: |a,b,c|v_block(v.visit_block, a, b, c),
                visit_stmt: |a,b,c|v_stmt(v.visit_stmt, a, b, c),
                visit_arm: |a,b,c|v_arm(v.visit_arm, a, b, c),
                visit_pat: |a,b,c|v_pat(v.visit_pat, a, b, c),
                visit_decl: |a,b,c|v_decl(v.visit_decl, a, b, c),
                visit_expr: |a,b,c|v_expr(v.visit_expr, a, b, c),
                visit_expr_post: |a,b,c| v_expr_post(v.visit_expr_post,
                                                     a, b, c),
                visit_ty: visit_ty,
                visit_ty_params: |a,b,c|
                    v_ty_params(v.visit_ty_params, a, b, c),
                visit_fn: |a,b,c,d,e,f,g|
                    v_fn(v.visit_fn, a, b, c, d, e, f, g),
                visit_ty_method: |a,b,c|
                    v_ty_method(v.visit_ty_method, a, b, c),
                visit_trait_method: |a,b,c|
                    v_trait_method(v.visit_trait_method, a, b, c),
                visit_struct_def: |a,b,c,d,e,f|
                    v_struct_def(v.visit_struct_def, a, b, c, d, e, f),
                visit_struct_field: |a,b,c|
                    v_struct_field(v.visit_struct_field, a, b, c),
                visit_struct_method: |a,b,c|
                    v_struct_method(v.visit_struct_method, a, b, c)
               });
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
