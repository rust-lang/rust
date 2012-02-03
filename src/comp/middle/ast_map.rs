import option;
import std::map;
import syntax::ast::*;
import syntax::ast_util;
import syntax::{visit, codemap};

enum path_elt { path_mod(str), path_name(str) }
type path = [path_elt];

enum ast_node {
    node_item(@item, @path),
    node_native_item(@native_item, @path),
    node_method(@method, @path),
    node_expr(@expr),
    // Locals are numbered, because the alias analysis needs to know in which
    // order they are introduced.
    node_arg(arg, uint),
    node_local(uint),
    node_res_ctor(@item),
}

type map = std::map::map<node_id, ast_node>;
type ctx = {map: map, mutable path: path, mutable local_id: uint};
type vt = visit::vt<ctx>;

fn map_crate(c: crate) -> map {
    let cx = {map: std::map::new_int_hash(),
              mutable path: [],
              mutable local_id: 0u};
    visit::visit_crate(c, cx, visit::mk_vt(@{
        visit_item: map_item,
        visit_native_item: map_native_item,
        visit_expr: map_expr,
        visit_fn: map_fn,
        visit_local: map_local,
        visit_arm: map_arm
        with *visit::default_visitor()
    }));
    ret cx.map;
}

fn map_fn(fk: visit::fn_kind, decl: fn_decl, body: blk,
          sp: codemap::span, id: node_id, cx: ctx, v: vt) {
    for a in decl.inputs {
        cx.map.insert(a.id, node_arg(a, cx.local_id));
        cx.local_id += 1u;
    }
    visit::visit_fn(fk, decl, body, sp, id, cx, v);
}

fn map_local(loc: @local, cx: ctx, v: vt) {
    pat_util::pat_bindings(loc.node.pat) {|p_id, _s, _p|
        cx.map.insert(p_id, node_local(cx.local_id));
        cx.local_id += 1u;
    };
    visit::visit_local(loc, cx, v);
}

fn map_arm(arm: arm, cx: ctx, v: vt) {
    pat_util::pat_bindings(arm.pats[0]) {|p_id, _s, _p|
        cx.map.insert(p_id, node_local(cx.local_id));
        cx.local_id += 1u;
    };
    visit::visit_arm(arm, cx, v);
}

fn map_item(i: @item, cx: ctx, v: vt) {
    cx.map.insert(i.id, node_item(i, @cx.path));
    alt i.node {
      item_impl(_, _, _, ms) {
        for m in ms { cx.map.insert(m.id, node_method(m, @cx.path)); }
      }
      item_res(_, _, _, dtor_id, ctor_id) {
        cx.map.insert(ctor_id, node_res_ctor(i));
        cx.map.insert(dtor_id, node_item(i, @cx.path));
      }
      _ { }
    }
    alt i.node {
      item_mod(_) | item_native_mod(_) { cx.path += [path_mod(i.ident)]; }
      _ { cx.path += [path_name(i.ident)]; }
    }
    visit::visit_item(i, cx, v);
    vec::pop(cx.path);
}

fn map_native_item(i: @native_item, cx: ctx, v: vt) {
    cx.map.insert(i.id, node_native_item(i, @cx.path));
    visit::visit_native_item(i, cx, v);
}

fn map_expr(ex: @expr, cx: ctx, v: vt) {
    cx.map.insert(ex.id, node_expr(ex));
    visit::visit_expr(ex, cx, v);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
