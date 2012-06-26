// Finds items that are externally reachable, to determine which items
// need to have their metadata (and possibly their AST) serialized.
// All items that can be referred to through an exported name are
// reachable, and when a reachable thing is inline or generic, it
// makes all other generics or inline functions that it references
// reachable as well.

import syntax::ast::*;
import syntax::{visit, ast_util, ast_map};
import syntax::ast_util::def_id_of_def;
import syntax::attr;
import syntax::print::pprust::expr_to_str;
import std::map::hashmap;
import driver::session::*;

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

fn traverse_exports(cx: ctx, vis: [@view_item]/~) -> bool {
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
    let n = alt cx.tcx.items.find(did.node) {
        none { ret; } // This can happen for self, for example
        some(n) { n }
    };
    alt n {
      ast_map::node_item(item, _) { traverse_public_item(cx, item); }
      ast_map::node_method(_, impl_id, _) { traverse_def_id(cx, impl_id); }
      ast_map::node_foreign_item(item, _, _) { cx.rmap.insert(item.id, ()); }
      ast_map::node_variant(v, _, _) { cx.rmap.insert(v.node.id, ()); }
      // If it's a ctor, consider the parent reachable
      ast_map::node_ctor(_, _, _, parent_id, _) {
        traverse_def_id(cx, parent_id);
      }
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
      item_foreign_mod(nm) {
          if !traverse_exports(cx, nm.view_items) {
              for vec::each(nm.items) {|item| cx.rmap.insert(item.id, ()); }
          }
      }
      item_fn(_, tps, blk) {
        if tps.len() > 0u ||
           attr::find_inline_attr(item.attrs) != attr::ia_none {
            traverse_inline_body(cx, blk);
        }
      }
      item_impl(tps, _, _, _, ms) {
        for vec::each(ms) {|m|
            if tps.len() > 0u || m.tps.len() > 0u ||
               attr::find_inline_attr(m.attrs) != attr::ia_none {
                cx.rmap.insert(m.id, ());
                traverse_inline_body(cx, m.body);
            }
        }
      }
      item_class(tps, _ifaces, items, ctor, m_dtor, _) {
        cx.rmap.insert(ctor.node.id, ());
        option::iter(m_dtor) {|dtor|
            cx.rmap.insert(dtor.node.id, ());
            // dtors don't have attrs
            if tps.len() > 0u {
                traverse_inline_body(cx, dtor.node.body);
            }
        }
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
      item_ty(t, _, _) {
        traverse_ty(t, cx, mk_ty_visitor());
      }
      item_const(*) |
      item_enum(*) | item_iface(*) {}
    }
}

fn mk_ty_visitor() -> visit::vt<ctx> {
    visit::mk_vt(@{visit_ty: traverse_ty with *visit::default_visitor()})
}

fn traverse_ty(ty: @ty, cx: ctx, v: visit::vt<ctx>) {
    if cx.rmap.contains_key(ty.id) { ret; }
    cx.rmap.insert(ty.id, ());

    alt ty.node {
      ty_path(p, p_id) {
        alt cx.tcx.def_map.find(p_id) {
          // Kind of a hack to check this here, but I'm not sure what else
          // to do
          some(def_prim_ty(_)) { /* do nothing */ }
          some(d) { traverse_def_id(cx, def_id_of_def(d)); }
          none    { /* do nothing -- but should we fail here? */ }
        }
        for p.types.each {|t| v.visit_ty(t, cx, v); };
      }
      _ { visit::visit_ty(ty, cx, v); }
    }
}

fn traverse_inline_body(cx: ctx, body: blk) {
    fn traverse_expr(e: @expr, cx: ctx, v: visit::vt<ctx>) {
        alt e.node {
          expr_path(_) {
            alt cx.tcx.def_map.find(e.id) {
                some(d) {
                  traverse_def_id(cx, def_id_of_def(d));
                }
                none      { cx.tcx.sess.span_bug(e.span, #fmt("Unbound node \
                  id %? while traversing %s", e.id, expr_to_str(e))); }
            }
          }
          expr_field(_, _, _) {
            alt cx.method_map.find(e.id) {
              some({origin: typeck::method_static(did), _}) {
                traverse_def_id(cx, did);
              }
              _ {}
            }
          }
          _ {}
        }
        visit::visit_expr(e, cx, v);
    }
    // Don't ignore nested items: for example if a generic fn contains a
    // generic impl (as in deque::create), we need to monomorphize the
    // impl as well
    fn traverse_item(i: @item, cx: ctx, _v: visit::vt<ctx>) {
      traverse_public_item(cx, i);
    }
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
              item_class(_, _, _, _, some(_), _) {
                traverse_public_item(cx, i);
              }
              _ {}
            }
        }
        with *visit::default_visitor()
    }));
}
