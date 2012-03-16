import std::map;
import std::map::hashmap;
import syntax::ast::*;
import syntax::ast_util;
import syntax::ast_util::inlined_item_methods;
import syntax::{visit, codemap};
import driver::session::session;
import front::attr;

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
    node_ctor(@item),
    node_block(blk),
}

type map = std::map::hashmap<node_id, ast_node>;
type ctx = {map: map, mutable path: path,
            mutable local_id: uint, sess: session};
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
        visit_view_item: map_view_item
        with *visit::default_visitor()
    });
}

fn map_crate(sess: session, c: crate) -> map {
    let cx = {map: std::map::int_hash(),
              mutable path: [],
              mutable local_id: 0u,
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
              mutable path: path,
              mutable local_id: 0u,
              sess: sess};
    let v = mk_ast_map_visitor();

    // methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now:
    alt ii {
      ii_item(i) { /* fallthrough */ }
      ii_method(impl_did, m) {
        map_method(impl_did, @path, m, cx);
      }
    }

    // visit the item / method contents and add those to the map:
    ii.accept(cx, v);
}

fn map_fn(fk: visit::fn_kind, decl: fn_decl, body: blk,
          sp: codemap::span, id: node_id, cx: ctx, v: vt) {
    for a in decl.inputs {
        cx.map.insert(a.id, node_arg(a, cx.local_id));
        cx.local_id += 1u;
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
        for m in ms {
            map_method(impl_did, extend(cx, i.ident), m, cx);
        }
      }
      item_res(_, _, _, dtor_id, ctor_id) {
        cx.map.insert(ctor_id, node_ctor(i));
        cx.map.insert(dtor_id, node_item(i, item_path));
      }
      item_enum(vs, _) {
        for v in vs {
            cx.map.insert(v.node.id, node_variant(
                v, i, extend(cx, i.ident)));
        }
      }
      item_native_mod(nm) {
        let abi = alt attr::native_abi(i.attrs) {
          either::left(msg) { cx.sess.span_fatal(i.span, msg); }
          either::right(abi) { abi }
        };
        for nitem in nm.items {
            cx.map.insert(nitem.id, node_native_item(nitem, abi, @cx.path));
        }
      }
      item_class(_, _, ctor) {
        cx.map.insert(ctor.node.id, node_ctor(i));
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
        for vp in vps {
            let (id, name) = alt vp.node {
              view_path_simple(nm, _, id) { (id, nm) }
              view_path_glob(pth, id) | view_path_list(pth, _, id) {
                  // should be a constraint on the type
                assert (vec::is_not_empty(*pth));
                (id, vec::last(*pth))
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

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
