import std::map;
import std::map::hashmap;
import syntax::ast::*;
import syntax::print::pprust;
import syntax::ast_util;
import middle::pat_util::*;
import syntax::ast_util::inlined_item_methods;
import syntax::{visit, codemap};
import driver::session::session;
import syntax::attr;

enum path_elt { path_mod(str), path_name(str) }
type path = [path_elt];

fn path_to_str_with_sep(p: path, sep: str) -> str {
    let strs = vec::map(p) {|e|
        alt e {
          path_mod(s) { s }
          path_name(s) { s }
        }
    };
    str::connect(strs, sep)
}

fn path_ident_to_str(p: path, i: ident) -> str {
    if vec::is_empty(p) {
        i
    } else {
        #fmt["%s::%s", path_to_str(p), i]
    }
}

fn path_to_str(p: path) -> str {
    path_to_str_with_sep(p, "::")
}

enum ast_node {
    node_item(@item, @path),
    node_native_item(@native_item, native_abi, @path),
    node_method(@method, def_id /* impl did */, @path /* path to the impl */),
    node_variant(variant, @item, @path),
    node_expr(@expr),
    node_export(@view_path, @path),
    // Locals are numbered, because the alias analysis needs to know in which
    // order they are introduced.
    node_arg(arg, uint),
    node_local(uint),
    // Constructor for either a resource or a class
    node_ctor(ident, [ty_param], a_ctor, @path),
    node_block(blk),
}

enum a_ctor {
  res_ctor(fn_decl, node_id, codemap::span),
  class_ctor(@class_ctor, def_id /* ID for parent class */),
}

type map = std::map::hashmap<node_id, ast_node>;
type ctx = {map: map, mut path: path,
            mut local_id: uint, sess: session};
type vt = visit::vt<ctx>;

fn extend(cx: ctx, elt: str) -> @path {
    @(cx.path + [path_name(elt)])
}

fn mk_ast_map_visitor() -> vt {
    ret visit::mk_vt(@{
        visit_item: map_item,
        visit_expr: map_expr,
        visit_fn: map_fn,
        visit_local: map_local,
        visit_arm: map_arm,
        visit_view_item: map_view_item,
        visit_block: map_block
        with *visit::default_visitor()
    });
}

fn map_crate(sess: session, c: crate) -> map {
    let cx = {map: std::map::int_hash(),
              mut path: [],
              mut local_id: 0u,
              sess: sess};
    visit::visit_crate(c, cx, mk_ast_map_visitor());
    ret cx.map;
}

// Used for items loaded from external crate that are being inlined into this
// crate.  The `path` should be the path to the item but should not include
// the item itself.
fn map_decoded_item(sess: session, map: map, path: path, ii: inlined_item) {
    // I believe it is ok for the local IDs of inlined items from other crates
    // to overlap with the local ids from this crate, so just generate the ids
    // starting from 0.  (In particular, I think these ids are only used in
    // alias analysis, which we will not be running on the inlined items, and
    // even if we did I think it only needs an ordering between local
    // variables that are simultaneously in scope).
    let cx = {map: map,
              mut path: path,
              mut local_id: 0u,
              sess: sess};
    let v = mk_ast_map_visitor();

    // methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now:
    alt ii {
      ii_item(_) | ii_ctor(_,_,_,_) { /* fallthrough */ }
      ii_native(i) {
        cx.map.insert(i.id, node_native_item(i, native_abi_rust_intrinsic,
                                             @path));
      }
      ii_method(impl_did, m) {
        map_method(impl_did, @path, m, cx);
      }
    }

    // visit the item / method contents and add those to the map:
    ii.accept(cx, v);
}

fn map_fn(fk: visit::fn_kind, decl: fn_decl, body: blk,
          sp: codemap::span, id: node_id, cx: ctx, v: vt) {
    for decl.inputs.each {|a|
        cx.map.insert(a.id, node_arg(a, cx.local_id));
        cx.local_id += 1u;
    }
    alt fk {
      visit::fk_ctor(nm, tps, self_id, parent_id) {
          let ct = @{node: {id: id, self_id: self_id,
                           dec: decl, body: body},
                    span: sp};
          cx.map.insert(id, node_ctor(nm, tps, class_ctor(ct, parent_id),
                                      @cx.path));
       }
       _ {}
    }
    visit::visit_fn(fk, decl, body, sp, id, cx, v);
}

fn map_block(b: blk, cx: ctx, v: vt) {
    cx.map.insert(b.node.id, node_block(b));
    visit::visit_block(b, cx, v);
}

fn number_pat(cx: ctx, pat: @pat) {
    pat_util::walk_pat(pat) {|p|
        alt p.node {
          pat_ident(_, _) {
            cx.map.insert(p.id, node_local(cx.local_id));
            cx.local_id += 1u;
          }
          _ {}
        }
    };
}

fn map_local(loc: @local, cx: ctx, v: vt) {
    number_pat(cx, loc.node.pat);
    visit::visit_local(loc, cx, v);
}

fn map_arm(arm: arm, cx: ctx, v: vt) {
    number_pat(cx, arm.pats[0]);
    visit::visit_arm(arm, cx, v);
}

fn map_method(impl_did: def_id, impl_path: @path,
              m: @method, cx: ctx) {
    cx.map.insert(m.id, node_method(m, impl_did, impl_path));
    cx.map.insert(m.self_id, node_local(cx.local_id));
    cx.local_id += 1u;
}

fn map_item(i: @item, cx: ctx, v: vt) {
    let item_path = @cx.path;
    cx.map.insert(i.id, node_item(i, item_path));
    alt i.node {
      item_impl(_, _, _, ms) {
        let impl_did = ast_util::local_def(i.id);
        for ms.each {|m|
            map_method(impl_did, extend(cx, i.ident), m, cx);
        }
      }
      item_res(decl, tps, _, dtor_id, ctor_id) {
        cx.map.insert(ctor_id, node_ctor(i.ident, tps,
                                         res_ctor(decl, ctor_id, i.span),
                                         item_path));
        cx.map.insert(dtor_id, node_item(i, item_path));
      }
      item_enum(vs, _) {
        for vs.each {|v|
            cx.map.insert(v.node.id, node_variant(
                v, i, extend(cx, i.ident)));
        }
      }
      item_native_mod(nm) {
        let abi = alt attr::native_abi(i.attrs) {
          either::left(msg) { cx.sess.span_fatal(i.span, msg); }
          either::right(abi) { abi }
        };
        for nm.items.each {|nitem|
            cx.map.insert(nitem.id, node_native_item(nitem, abi, @cx.path));
        }
      }
      item_class(_, _, items, ctor) {
          let d_id = ast_util::local_def(i.id);
          let p = extend(cx, i.ident);
          for items.each {|ci|
           // only need to handle methods
           alt ci.node {
             class_method(m) { map_method(d_id, p, m, cx); }
             _ {}
           }
          }
      }
      _ { }
    }
    alt i.node {
      item_mod(_) | item_native_mod(_) {
        cx.path += [path_mod(i.ident)];
      }
      _ { cx.path += [path_name(i.ident)]; }
    }
    visit::visit_item(i, cx, v);
    vec::pop(cx.path);
}

fn map_view_item(vi: @view_item, cx: ctx, _v: vt) {
    alt vi.node {
      view_item_export(vps) {
        for vps.each {|vp|
            let (id, name) = alt vp.node {
              view_path_simple(nm, _, id) { (id, nm) }
              view_path_glob(pth, id) | view_path_list(pth, _, id) {
                (id, path_to_ident(pth))
              }
            };
            cx.map.insert(id, node_export(vp, extend(cx, name)));
        }
      }
      _ {}
    }
}

fn map_expr(ex: @expr, cx: ctx, v: vt) {
    cx.map.insert(ex.id, node_expr(ex));
    visit::visit_expr(ex, cx, v);
}

fn node_id_to_str(map: map, id: node_id) -> str {
    alt map.find(id) {
      none {
        #fmt["unknown node (id=%d)", id]
      }
      some(node_item(item, path)) {
        #fmt["item %s (id=%?)", path_ident_to_str(*path, item.ident), id]
      }
      some(node_native_item(item, abi, path)) {
        #fmt["native item %s with abi %? (id=%?)",
             path_ident_to_str(*path, item.ident), abi, id]
      }
      some(node_method(m, impl_did, path)) {
        #fmt["method %s in %s (id=%?)",
             m.ident, path_to_str(*path), id]
      }
      some(node_variant(variant, def_id, path)) {
        #fmt["variant %s in %s (id=%?)",
             variant.node.name, path_to_str(*path), id]
      }
      some(node_expr(expr)) {
        #fmt["expr %s (id=%?)",
             pprust::expr_to_str(expr), id]
      }
      some(node_export(_, path)) {
        #fmt["export %s (id=%?)", // FIXME: add more info here
             path_to_str(*path), id]
      }
      some(node_arg(_, _)) { // FIXME: add more info here
        #fmt["arg (id=%?)", id]
      }
      some(node_local(_)) { // FIXME: add more info here
        #fmt["local (id=%?)", id]
      }
      some(node_ctor(_, _, _, _)) { // FIXME: add more info here
        #fmt["node_ctor (id=%?)", id]
      }
      some(node_block(_)) {
        #fmt["block"]
      }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
