// Finds items that are externally reachable, to determine which items
// need to have their metadata (and possibly their AST) serialized.
// All items that can be referred to through an exported name are
// reachable, and when a reachable thing is inline or generic, it
// makes all other generics or inline functions that it references
// reachable as well.

import middle::{resolve, ast_map, typeck};
import syntax::ast::*;
import syntax::visit;
import syntax::ast_util::def_id_of_def;
import front::attr;
import std::map::hashmap;

export map, find_reachable;

type map = std::map::hashmap<node_id, ()>;

type ctx = {ccx: @middle::trans::common::crate_ctxt,
            rmap: map};

fn find_reachable(ccx: @middle::trans::common::crate_ctxt, crate_mod: _mod)
    -> map {
    let rmap = std::map::new_int_hash();
    traverse_public_mod({ccx: ccx, rmap: rmap}, crate_mod);
    rmap
}

fn traverse_exports(cx: ctx, vis: [@view_item]) -> bool {
    let found_export = false;
    for vi in vis {
        alt vi.node {
          view_item_export(vps) {
            found_export = true;
            for vp in vps {
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
    option::may(cx.ccx.exp_map.find(exp_id)) {|defs|
        for def in defs { traverse_def_id(cx, def.id); }
    }
}

fn traverse_def_id(cx: ctx, did: def_id) {
    if did.crate != local_crate { ret; }
    alt cx.ccx.tcx.items.get(did.node) {
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
        for item in m.items { traverse_public_item(cx, item); }
    }
}

fn traverse_public_item(cx: ctx, item: @item) {
    if cx.rmap.contains_key(item.id) { ret; }
    cx.rmap.insert(item.id, ());
    alt item.node {
      item_mod(m) { traverse_public_mod(cx, m); }
      item_native_mod(nm) {
          if !traverse_exports(cx, nm.view_items) {
              for item in nm.items { cx.rmap.insert(item.id, ()); }
          }
      }
      item_res(_, tps, blk, _, _) | item_fn(_, tps, blk) {
        if tps.len() > 0u ||
           attr::find_inline_attr(item.attrs) != attr::ia_none {
            traverse_inline_body(cx, blk);
        }
      }
      item_impl(tps, _, _, ms) {
        for m in ms {
            if tps.len() > 0u || m.tps.len() > 0u ||
               attr::find_inline_attr(m.attrs) != attr::ia_none {
                traverse_inline_body(cx, m.body);
            }
        }
      }
      item_class(_tps, _items, _) {} // FIXME handle these when stable
      item_const(_, _) | item_ty(_, _) | item_enum(_, _) | item_iface(_, _) {}
    }
}

fn traverse_inline_body(cx: ctx, body: blk) {
    fn traverse_expr(e: @expr, cx: ctx, v: visit::vt<ctx>) {
        alt e.node {
          expr_path(_) {
            traverse_def_id(cx, def_id_of_def(cx.ccx.tcx.def_map.get(e.id)));
          }
          expr_field(_, _, _) {
            alt cx.ccx.maps.method_map.find(e.id) {
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
