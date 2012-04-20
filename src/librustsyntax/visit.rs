
import ast::*;
import codemap::span;

// Context-passing AST walker. Each overridden visit method has full control
// over what happens with its node, it can do its own traversal of the node's
// children (potentially passing in different contexts to each), call
// visit::visit_* to apply the default traversal algorithm (again, it can
// override the context), or prevent deeper traversal by doing nothing.

// Our typesystem doesn't do circular types, so the visitor record can not
// hold functions that take visitors. A vt enum is used to break the cycle.
enum vt<E> { mk_vt(visitor<E>), }

enum fn_kind {
    fk_item_fn(ident, [ty_param]), //< an item declared with fn()
    fk_method(ident, [ty_param], @method),
    fk_res(ident, [ty_param], region_param),
    fk_anon(proto),  //< an anonymous function like fn@(...)
    fk_fn_block,     //< a block {||...}
    fk_ctor(ident, [ty_param], node_id /* self id */,
            def_id /* parent class id */) // class constructor
}

fn name_of_fn(fk: fn_kind) -> ident {
    alt fk {
      fk_item_fn(name, _) | fk_method(name, _, _) | fk_res(name, _, _)
          | fk_ctor(name, _, _, _) { name }
      fk_anon(_) | fk_fn_block { "anon" }
    }
}

fn tps_of_fn(fk: fn_kind) -> [ty_param] {
    alt fk {
      fk_item_fn(_, tps) | fk_method(_, tps, _) | fk_res(_, tps, _)
          | fk_ctor(_, tps, _, _) { tps }
      fk_anon(_) | fk_fn_block { [] }
    }
}

type visitor<E> =
    // takes the components so that one function can be
    // generic over constr and ty_constr
    @{visit_mod: fn@(_mod, span, node_id, E, vt<E>),
      visit_view_item: fn@(@view_item, E, vt<E>),
      visit_native_item: fn@(@native_item, E, vt<E>),
      visit_item: fn@(@item, E, vt<E>),
      visit_local: fn@(@local, E, vt<E>),
      visit_block: fn@(ast::blk, E, vt<E>),
      visit_stmt: fn@(@stmt, E, vt<E>),
      visit_arm: fn@(arm, E, vt<E>),
      visit_pat: fn@(@pat, E, vt<E>),
      visit_decl: fn@(@decl, E, vt<E>),
      visit_expr: fn@(@expr, E, vt<E>),
      visit_ty: fn@(@ty, E, vt<E>),
      visit_ty_params: fn@([ty_param], E, vt<E>),
      visit_constr: fn@(@path, span, node_id, E, vt<E>),
      visit_fn: fn@(fn_kind, fn_decl, blk, span, node_id, E, vt<E>),
      visit_class_item: fn@(@class_member, E, vt<E>)};

fn default_visitor<E>() -> visitor<E> {
    ret @{visit_mod: bind visit_mod::<E>(_, _, _, _, _),
          visit_view_item: bind visit_view_item::<E>(_, _, _),
          visit_native_item: bind visit_native_item::<E>(_, _, _),
          visit_item: bind visit_item::<E>(_, _, _),
          visit_local: bind visit_local::<E>(_, _, _),
          visit_block: bind visit_block::<E>(_, _, _),
          visit_stmt: bind visit_stmt::<E>(_, _, _),
          visit_arm: bind visit_arm::<E>(_, _, _),
          visit_pat: bind visit_pat::<E>(_, _, _),
          visit_decl: bind visit_decl::<E>(_, _, _),
          visit_expr: bind visit_expr::<E>(_, _, _),
          visit_ty: bind skip_ty::<E>(_, _, _),
          visit_ty_params: bind visit_ty_params::<E>(_, _, _),
          visit_constr: bind visit_constr::<E>(_, _, _, _, _),
          visit_fn: bind visit_fn::<E>(_, _, _, _, _, _, _),
          visit_class_item: bind visit_class_item::<E>(_,_,_)};
}

fn visit_crate<E>(c: crate, e: E, v: vt<E>) {
    v.visit_mod(c.node.module, c.span, crate_node_id, e, v);
}

fn visit_crate_directive<E>(cd: @crate_directive, e: E, v: vt<E>) {
    alt cd.node {
      cdir_src_mod(_, _) { }
      cdir_dir_mod(_, cdirs, _) {
        for cdirs.each {|cdir|
            visit_crate_directive(cdir, e, v);
        }
      }
      cdir_view_item(vi) { v.visit_view_item(vi, e, v); }
      cdir_syntax(_) { }
    }
}

fn visit_mod<E>(m: _mod, _sp: span, _id: node_id, e: E, v: vt<E>) {
    for m.view_items.each {|vi| v.visit_view_item(vi, e, v); }
    for m.items.each {|i| v.visit_item(i, e, v); }
}

fn visit_view_item<E>(_vi: @view_item, _e: E, _v: vt<E>) { }

fn visit_local<E>(loc: @local, e: E, v: vt<E>) {
    v.visit_pat(loc.node.pat, e, v);
    v.visit_ty(loc.node.ty, e, v);
    alt loc.node.init { none { } some(i) { v.visit_expr(i.expr, e, v); } }
}

fn visit_item<E>(i: @item, e: E, v: vt<E>) {
    alt i.node {
      item_const(t, ex) { v.visit_ty(t, e, v); v.visit_expr(ex, e, v); }
      item_fn(decl, tp, body) {
        v.visit_fn(fk_item_fn(i.ident, tp), decl, body, i.span, i.id, e, v);
      }
      item_mod(m) { v.visit_mod(m, i.span, i.id, e, v); }
      item_native_mod(nm) {
        for nm.view_items.each {|vi| v.visit_view_item(vi, e, v); }
        for nm.items.each {|ni| v.visit_native_item(ni, e, v); }
      }
      item_ty(t, tps, rp) {
        v.visit_ty(t, e, v);
        v.visit_ty_params(tps, e, v);
      }
      item_res(decl, tps, body, dtor_id, _, rp) {
        v.visit_fn(fk_res(i.ident, tps, rp), decl, body, i.span,
                   dtor_id, e, v);
      }
      item_enum(variants, tps, _) {
        v.visit_ty_params(tps, e, v);
        for variants.each {|vr|
            for vr.node.args.each {|va| v.visit_ty(va.ty, e, v); }
        }
      }
      item_impl(tps, ifce, ty, methods) {
        v.visit_ty_params(tps, e, v);
        alt ifce { some(ty) { v.visit_ty(ty, e, v); } none {} }
        v.visit_ty(ty, e, v);
        for methods.each {|m|
            visit_method_helper(m, e, v)
        }
      }
      item_class(tps, ifaces, members, ctor, _) {
          v.visit_ty_params(tps, e, v);
          for members.each {|m|
             v.visit_class_item(m, e, v);
          }
          for ifaces.each {|p| visit_path(p.path, e, v); }
          visit_class_ctor_helper(ctor, i.ident, tps,
                                  ast_util::local_def(i.id), e, v);
      }
      item_iface(tps, methods) {
        v.visit_ty_params(tps, e, v);
        for methods.each {|m|
            for m.decl.inputs.each {|a| v.visit_ty(a.ty, e, v); }
            v.visit_ty(m.decl.output, e, v);
        }
      }
    }
}

fn visit_class_item<E>(cm: @class_member, e:E, v:vt<E>) {
    alt cm.node {
        instance_var(_, t, _, _, _) {
            v.visit_ty(t, e, v);
        }
        class_method(m) {
            visit_method_helper(m, e, v);
        }
    }
}

fn skip_ty<E>(_t: @ty, _e: E, _v: vt<E>) {}

fn visit_ty<E>(t: @ty, e: E, v: vt<E>) {
    alt t.node {
      ty_box(mt) | ty_uniq(mt) |
      ty_vec(mt) | ty_ptr(mt) | ty_rptr(_, mt) {
        v.visit_ty(mt.ty, e, v);
      }
      ty_rec(flds) {
        for flds.each {|f| v.visit_ty(f.node.mt.ty, e, v); }
      }
      ty_tup(ts) { for ts.each {|tt| v.visit_ty(tt, e, v); } }
      ty_fn(_, decl) {
        for decl.inputs.each {|a| v.visit_ty(a.ty, e, v); }
        for decl.constraints.each {|c|
            v.visit_constr(c.node.path, c.span, c.node.id, e, v);
        }
        v.visit_ty(decl.output, e, v);
      }
      ty_path(p, _) { visit_path(p, e, v); }
      ty_vstore(t, _) {
        v.visit_ty(t, e, v);
      }
      ty_constr(t, cs) {
        v.visit_ty(t, e, v);
        for cs.each {|tc|
            v.visit_constr(tc.node.path, tc.span, tc.node.id, e, v);
        }
      }
      ty_nil |
      ty_bot |
      ty_mac(_) |
      ty_infer {
      }
    }
}

fn visit_constr<E>(_operator: @path, _sp: span, _id: node_id, _e: E,
                   _v: vt<E>) {
    // default
}

fn visit_path<E>(p: @path, e: E, v: vt<E>) {
    for p.node.types.each {|tp| v.visit_ty(tp, e, v); }
}

fn visit_pat<E>(p: @pat, e: E, v: vt<E>) {
    alt p.node {
      pat_enum(path, children) {
        visit_path(path, e, v);
        option::iter(children) {|children|
                for children.each {|child| v.visit_pat(child, e, v); }}
      }
      pat_rec(fields, _) {
        for fields.each {|f| v.visit_pat(f.pat, e, v); }
      }
      pat_tup(elts) { for elts.each {|elt| v.visit_pat(elt, e, v); } }
      pat_box(inner) | pat_uniq(inner) {
        v.visit_pat(inner, e, v);
      }
      pat_ident(path, inner) {
          visit_path(path, e, v);
          option::iter(inner) {|subpat| v.visit_pat(subpat, e, v)};
      }
      pat_lit(ex) { v.visit_expr(ex, e, v); }
      pat_range(e1, e2) { v.visit_expr(e1, e, v); v.visit_expr(e2, e, v); }
      pat_wild {}
    }
}

fn visit_native_item<E>(ni: @native_item, e: E, v: vt<E>) {
    alt ni.node {
      native_item_fn(fd, tps) {
        v.visit_ty_params(tps, e, v);
        visit_fn_decl(fd, e, v);
      }
    }
}

fn visit_ty_params<E>(tps: [ty_param], e: E, v: vt<E>) {
    for tps.each {|tp|
        for vec::each(*tp.bounds) {|bound|
            alt bound {
              bound_iface(t) { v.visit_ty(t, e, v); }
              bound_copy | bound_send { }
            }
        }
    }
}

fn visit_fn_decl<E>(fd: fn_decl, e: E, v: vt<E>) {
    for fd.inputs.each {|a| v.visit_ty(a.ty, e, v); }
    for fd.constraints.each {|c|
        v.visit_constr(c.node.path, c.span, c.node.id, e, v);
    }
    v.visit_ty(fd.output, e, v);
}

// Note: there is no visit_method() method in the visitor, instead override
// visit_fn() and check for fk_method().  I named this visit_method_helper()
// because it is not a default impl of any method, though I doubt that really
// clarifies anything. - Niko
fn visit_method_helper<E>(m: @method, e: E, v: vt<E>) {
    v.visit_fn(fk_method(m.ident, m.tps, m), m.decl, m.body, m.span,
               m.id, e, v);
}

// Similar logic to the comment on visit_method_helper - Tim
fn visit_class_ctor_helper<E>(ctor: class_ctor, nm: ident, tps: [ty_param],
                              parent_id: def_id, e: E, v: vt<E>) {
    v.visit_fn(visit::fk_ctor(nm, tps, ctor.node.self_id,
                              parent_id), ctor.node.dec,
               ctor.node.body, ctor.span, ctor.node.id, e, v)

}

fn visit_fn<E>(fk: fn_kind, decl: fn_decl, body: blk, _sp: span,
               _id: node_id, e: E, v: vt<E>) {
    visit_fn_decl(decl, e, v);
    v.visit_ty_params(tps_of_fn(fk), e, v);
    v.visit_block(body, e, v);
}

fn visit_block<E>(b: ast::blk, e: E, v: vt<E>) {
    for b.node.view_items.each {|vi| v.visit_view_item(vi, e, v); }
    for b.node.stmts.each {|s| v.visit_stmt(s, e, v); }
    visit_expr_opt(b.node.expr, e, v);
}

fn visit_stmt<E>(s: @stmt, e: E, v: vt<E>) {
    alt s.node {
      stmt_decl(d, _) { v.visit_decl(d, e, v); }
      stmt_expr(ex, _) { v.visit_expr(ex, e, v); }
      stmt_semi(ex, _) { v.visit_expr(ex, e, v); }
    }
}

fn visit_decl<E>(d: @decl, e: E, v: vt<E>) {
    alt d.node {
      decl_local(locs) {
        for locs.each {|loc| v.visit_local(loc, e, v); }
      }
      decl_item(it) { v.visit_item(it, e, v); }
    }
}

fn visit_expr_opt<E>(eo: option<@expr>, e: E, v: vt<E>) {
    alt eo { none { } some(ex) { v.visit_expr(ex, e, v); } }
}

fn visit_exprs<E>(exprs: [@expr], e: E, v: vt<E>) {
    for exprs.each {|ex| v.visit_expr(ex, e, v); }
}

fn visit_mac<E>(m: mac, e: E, v: vt<E>) {
    alt m.node {
      ast::mac_invoc(pth, arg, body) {
        option::map(arg) {|arg| v.visit_expr(arg, e, v)}; }
      ast::mac_embed_type(ty) { v.visit_ty(ty, e, v); }
      ast::mac_embed_block(blk) { v.visit_block(blk, e, v); }
      ast::mac_ellipsis { }
      ast::mac_aq(_, e) { /* FIXME: maybe visit */ }
      ast::mac_var(_) { }
    }
}

fn visit_expr<E>(ex: @expr, e: E, v: vt<E>) {
    alt ex.node {
      expr_new(pool, _, val) {
        v.visit_expr(pool, e, v);
        v.visit_expr(val, e, v);
      }
      expr_vstore(x, _) { v.visit_expr(x, e, v); }
      expr_vec(es, _) { visit_exprs(es, e, v); }
      expr_rec(flds, base) {
        for flds.each {|f| v.visit_expr(f.node.expr, e, v); }
        visit_expr_opt(base, e, v);
      }
      expr_tup(elts) { for elts.each {|el| v.visit_expr(el, e, v); } }
      expr_call(callee, args, _) {
        visit_exprs(args, e, v);
        v.visit_expr(callee, e, v);
      }
      expr_bind(callee, args) {
        v.visit_expr(callee, e, v);
        for args.each {|eo| visit_expr_opt(eo, e, v); }
      }
      expr_binary(_, a, b) { v.visit_expr(a, e, v); v.visit_expr(b, e, v); }
      expr_addr_of(_, x) | expr_unary(_, x) | expr_loop_body(x) |
      expr_check(_, x) | expr_assert(x) {
        v.visit_expr(x, e, v);
      }
      expr_lit(_) { }
      expr_cast(x, t) { v.visit_expr(x, e, v); v.visit_ty(t, e, v); }
      expr_if(x, b, eo) {
        v.visit_expr(x, e, v);
        v.visit_block(b, e, v);
        visit_expr_opt(eo, e, v);
      }
      expr_if_check(x, b, eo) {
        v.visit_expr(x, e, v);
        v.visit_block(b, e, v);
        visit_expr_opt(eo, e, v);
      }
      expr_while(x, b) { v.visit_expr(x, e, v); v.visit_block(b, e, v); }
      expr_loop(b) { v.visit_block(b, e, v); }
      expr_do_while(b, x) { v.visit_block(b, e, v); v.visit_expr(x, e, v); }
      expr_alt(x, arms, _) {
        v.visit_expr(x, e, v);
        for arms.each {|a| v.visit_arm(a, e, v); }
      }
      expr_fn(proto, decl, body, _) {
        v.visit_fn(fk_anon(proto), decl, body, ex.span, ex.id, e, v);
      }
      expr_fn_block(decl, body) {
        v.visit_fn(fk_fn_block, decl, body, ex.span, ex.id, e, v);
      }
      expr_block(b) { v.visit_block(b, e, v); }
      expr_assign(a, b) { v.visit_expr(b, e, v); v.visit_expr(a, e, v); }
      expr_copy(a) { v.visit_expr(a, e, v); }
      expr_move(a, b) { v.visit_expr(b, e, v); v.visit_expr(a, e, v); }
      expr_swap(a, b) { v.visit_expr(a, e, v); v.visit_expr(b, e, v); }
      expr_assign_op(_, a, b) {
        v.visit_expr(b, e, v);
        v.visit_expr(a, e, v);
      }
      expr_field(x, _, tys) {
        v.visit_expr(x, e, v);
        for tys.each {|tp| v.visit_ty(tp, e, v); }
      }
      expr_index(a, b) { v.visit_expr(a, e, v); v.visit_expr(b, e, v); }
      expr_path(p) { visit_path(p, e, v); }
      expr_fail(eo) { visit_expr_opt(eo, e, v); }
      expr_break { }
      expr_cont { }
      expr_ret(eo) { visit_expr_opt(eo, e, v); }
      expr_be(x) { v.visit_expr(x, e, v); }
      expr_log(_, lv, x) {
        v.visit_expr(lv, e, v);
        v.visit_expr(x, e, v);
      }
      expr_mac(mac) { visit_mac(mac, e, v); }
    }
}

fn visit_arm<E>(a: arm, e: E, v: vt<E>) {
    for a.pats.each {|p| v.visit_pat(p, e, v); }
    visit_expr_opt(a.guard, e, v);
    v.visit_block(a.body, e, v);
}

// Simpler, non-context passing interface. Always walks the whole tree, simply
// calls the given functions on the nodes.

type simple_visitor =
    // takes the components so that one function can be
    // generic over constr and ty_constr
    @{visit_mod: fn@(_mod, span, node_id),
      visit_view_item: fn@(@view_item),
      visit_native_item: fn@(@native_item),
      visit_item: fn@(@item),
      visit_local: fn@(@local),
      visit_block: fn@(ast::blk),
      visit_stmt: fn@(@stmt),
      visit_arm: fn@(arm),
      visit_pat: fn@(@pat),
      visit_decl: fn@(@decl),
      visit_expr: fn@(@expr),
      visit_ty: fn@(@ty),
      visit_ty_params: fn@([ty_param]),
      visit_constr: fn@(@path, span, node_id),
      visit_fn: fn@(fn_kind, fn_decl, blk, span, node_id),
      visit_class_item: fn@(@class_member)};

fn simple_ignore_ty(_t: @ty) {}

fn default_simple_visitor() -> simple_visitor {
    ret @{visit_mod: fn@(_m: _mod, _sp: span, _id: node_id) { },
          visit_view_item: fn@(_vi: @view_item) { },
          visit_native_item: fn@(_ni: @native_item) { },
          visit_item: fn@(_i: @item) { },
          visit_local: fn@(_l: @local) { },
          visit_block: fn@(_b: ast::blk) { },
          visit_stmt: fn@(_s: @stmt) { },
          visit_arm: fn@(_a: arm) { },
          visit_pat: fn@(_p: @pat) { },
          visit_decl: fn@(_d: @decl) { },
          visit_expr: fn@(_e: @expr) { },
          visit_ty: simple_ignore_ty,
          visit_ty_params: fn@(_ps: [ty_param]) {},
          visit_constr: fn@(_p: @path, _sp: span, _id: node_id) { },
          visit_fn: fn@(_fk: fn_kind, _d: fn_decl, _b: blk, _sp: span,
                        _id: node_id) { },
          visit_class_item: fn@(_c: @class_member) {}
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
    fn v_native_item(f: fn@(@native_item), ni: @native_item, &&e: (),
                     v: vt<()>) {
        f(ni);
        visit_native_item(ni, e, v);
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
    fn v_ty(f: fn@(@ty), ty: @ty, &&e: (), v: vt<()>) {
        f(ty);
        visit_ty(ty, e, v);
    }
    fn v_ty_params(f: fn@([ty_param]), ps: [ty_param], &&e: (), v: vt<()>) {
        f(ps);
        visit_ty_params(ps, e, v);
    }
    fn v_constr(f: fn@(@path, span, node_id), pt: @path, sp: span,
                id: node_id, &&e: (), v: vt<()>) {
        f(pt, sp, id);
        visit_constr(pt, sp, id, e, v);
    }
    fn v_fn(f: fn@(fn_kind, fn_decl, blk, span, node_id),
            fk: fn_kind, decl: fn_decl, body: blk, sp: span,
            id: node_id, &&e: (), v: vt<()>) {
        f(fk, decl, body, sp, id);
        visit_fn(fk, decl, body, sp, id, e, v);
    }
    let visit_ty = if v.visit_ty == simple_ignore_ty {
        bind skip_ty(_, _, _)
    } else {
        bind v_ty(v.visit_ty, _, _, _)
    };
    fn v_class_item(f: fn@(@class_member),
                    cm: @class_member, &&e: (),
                    v: vt<()>) {
        f(cm);
        visit_class_item(cm, e, v);
    }
    ret mk_vt(@{visit_mod: bind v_mod(v.visit_mod, _, _, _, _, _),
                visit_view_item: bind v_view_item(v.visit_view_item, _, _, _),
                visit_native_item:
                    bind v_native_item(v.visit_native_item, _, _, _),
                visit_item: bind v_item(v.visit_item, _, _, _),
                visit_local: bind v_local(v.visit_local, _, _, _),
                visit_block: bind v_block(v.visit_block, _, _, _),
                visit_stmt: bind v_stmt(v.visit_stmt, _, _, _),
                visit_arm: bind v_arm(v.visit_arm, _, _, _),
                visit_pat: bind v_pat(v.visit_pat, _, _, _),
                visit_decl: bind v_decl(v.visit_decl, _, _, _),
                visit_expr: bind v_expr(v.visit_expr, _, _, _),
                visit_ty: visit_ty,
                visit_ty_params: bind v_ty_params(v.visit_ty_params, _, _, _),
                visit_constr: bind v_constr(v.visit_constr, _, _, _, _, _),
                visit_fn: bind v_fn(v.visit_fn, _, _, _, _, _, _, _),
                visit_class_item: bind v_class_item(v.visit_class_item, _, _,
                                                    _)
               });
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
