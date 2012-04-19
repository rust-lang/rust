// Finds items that are externally reachable, to determine which items
// need to have their metadata (and possibly their AST) serialized.
// All items that can be referred to through an exported name are
// reachable, and when a reachable thing is inline or generic, it
// makes all other generics or inline functions that it references
// reachable as well.

import syntax::ast::*;
import syntax::{visit, ast_util};
import syntax::ast_util::def_id_of_def;
import syntax::attr;
import std::map::hashmap;

export map, find_reachable;

type map = std::map::hashmap<node_id, ()>;

type ctx = {exp_map: resolve::exp_map,
            tcx: ty::ctxt,
            method_map: typeck::method_map,
            rmap: map};

fn find_reachable(crate_mod: _mod, exp_map: resolve::exp_map,
                  tcx: ty::ctxt, method_map: typeck::method_map) -> map {
    let rmap = std::map::int_hash();
    let cx = {exp_map: exp_map, tcx: tcx, method_map: method_map, rmap: rmap};
    traverse_public_mod(cx, crate_mod);
    traverse_all_resources(cx, crate_mod);
    rmap
}

fn traverse_exports(cx: ctx, vis: [@view_item]) -> bool {
    let mut found_export = false;
    for vec::each(vis) {|vi|
        alt vi.node {
          view_item_export(vps) {
            found_export = true;
            for vec::each(vps) {|vp|
                alt vp.node {
                  view_path_simple(_, _, id) | view_path_glob(_, id) |
                  view_path_list(_, _, id) {
                    traverse_export(cx, id);
                  }
                }
            }
          }
          _ {}
        }
    }
    found_export
}

fn traverse_export(cx: ctx, exp_id: node_id) {
    option::iter(cx.exp_map.find(exp_id)) {|defs|
        for vec::each(defs) {|def| traverse_def_id(cx, def.id); }
    }
}

fn traverse_def_id(cx: ctx, did: def_id) {
    if did.crate != local_crate { ret; }
    alt cx.tcx.items.get(did.node) {
      ast_map::node_item(item, _) { traverse_public_item(cx, item); }
      ast_map::node_method(_, impl_id, _) { traverse_def_id(cx, impl_id); }
      ast_map::node_native_item(item, _, _) { cx.rmap.insert(item.id, ()); }
      ast_map::node_variant(v, _, _) { cx.rmap.insert(v.node.id, ()); }
      _ {}
    }
}

fn traverse_public_mod(cx: ctx, m: _mod) {
    if !traverse_exports(cx, m.view_items) {
        // No exports, so every local item is exported
        for vec::each(m.items) {|item| traverse_public_item(cx, item); }
    }
}

fn traverse_public_item(cx: ctx, item: @item) {
    if cx.rmap.contains_key(item.id) { ret; }
    cx.rmap.insert(item.id, ());
    alt item.node {
      item_mod(m) { traverse_public_mod(cx, m); }
      item_native_mod(nm) {
          if !traverse_exports(cx, nm.view_items) {
              for vec::each(nm.items) {|item| cx.rmap.insert(item.id, ()); }
          }
      }
      item_res(_, tps, blk, _, _, _) {
        // resources seem to be unconditionally inlined
        traverse_inline_body(cx, blk);
      }
      item_fn(_, tps, blk) {
        if tps.len() > 0u ||
           attr::find_inline_attr(item.attrs) != attr::ia_none {
            traverse_inline_body(cx, blk);
        }
      }
      item_impl(tps, _, _, ms) {
        for vec::each(ms) {|m|
            if tps.len() > 0u || m.tps.len() > 0u ||
               attr::find_inline_attr(m.attrs) != attr::ia_none {
                traverse_inline_body(cx, m.body);
            }
        }
      }
      item_class(tps, _ifaces, items, ctor, _) {
        cx.rmap.insert(ctor.node.id, ());
        for vec::each(items) {|item|
            alt item.node {
              class_method(m) {
                cx.rmap.insert(m.id, ());
                if tps.len() > 0u ||
                   attr::find_inline_attr(m.attrs) != attr::ia_none {
                    traverse_inline_body(cx, m.body);
                }
              }
              _ {}
            }
        }
      }
      item_const(_, _) | item_ty(_, _, _) |
      item_enum(_, _, _) | item_iface(_, _) {}
    }
}

fn traverse_inline_body(cx: ctx, body: blk) {
    fn traverse_expr(e: @expr, cx: ctx, v: visit::vt<ctx>) {
        alt e.node {
          expr_path(_) {
            traverse_def_id(cx, def_id_of_def(cx.tcx.def_map.get(e.id)));
          }
          expr_field(_, _, _) {
            alt cx.method_map.find(e.id) {
              some(typeck::method_static(did)) { traverse_def_id(cx, did); }
              _ {}
            }
          }
          _ {}
        }
        visit::visit_expr(e, cx, v);
    }
    // Ignore nested items
    fn traverse_item(_i: @item, _cx: ctx, _v: visit::vt<ctx>) {}
    visit::visit_block(body, cx, visit::mk_vt(@{
        visit_expr: traverse_expr,
        visit_item: traverse_item
        with *visit::default_visitor()
    }));
}

fn traverse_all_resources(cx: ctx, crate_mod: _mod) {
    visit::visit_mod(crate_mod, ast_util::dummy_sp(), 0, cx, visit::mk_vt(@{
        visit_expr: {|_e, _cx, _v|},
        visit_item: {|i, cx, v|
            visit::visit_item(i, cx, v);
            alt i.node {
              item_res(_, _, _, _, _, _) {
                traverse_public_item(cx, i);
              }
              _ {}
            }
        }
        with *visit::default_visitor()
    }));
}
