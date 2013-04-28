// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Finds items that are externally reachable, to determine which items
// need to have their metadata (and possibly their AST) serialized.
// All items that can be referred to through an exported name are
// reachable, and when a reachable thing is inline or generic, it
// makes all other generics or inline functions that it references
// reachable as well.


use middle::resolve;
use middle::ty;
use middle::typeck;

use core::hashmap::HashSet;
use syntax::ast;
use syntax::ast::*;
use syntax::ast_util::def_id_of_def;
use syntax::attr;
use syntax::codemap;
use syntax::print::pprust::expr_to_str;
use syntax::{visit, ast_map};

pub type map = @HashSet<node_id>;

struct ctx<'self> {
    exp_map2: resolve::ExportMap2,
    tcx: ty::ctxt,
    method_map: typeck::method_map,
    rmap: &'self mut HashSet<node_id>,
}

pub fn find_reachable(crate_mod: &_mod, exp_map2: resolve::ExportMap2,
                      tcx: ty::ctxt, method_map: typeck::method_map) -> map {
    let mut rmap = HashSet::new();
    {
        let cx = ctx {
            exp_map2: exp_map2,
            tcx: tcx,
            method_map: method_map,
            rmap: &mut rmap
        };
        traverse_public_mod(&cx, ast::crate_node_id, crate_mod);
        traverse_all_resources_and_impls(&cx, crate_mod);
    }
    return @rmap;
}

fn traverse_exports(cx: &ctx, mod_id: node_id) -> bool {
    let mut found_export = false;
    match cx.exp_map2.find(&mod_id) {
      Some(ref exp2s) => {
        for (*exp2s).each |e2| {
            found_export = true;
            traverse_def_id(cx, e2.def_id)
        };
      }
      None => ()
    }
    return found_export;
}

fn traverse_def_id(cx: &ctx, did: def_id) {
    if did.crate != local_crate { return; }
    match cx.tcx.items.find(&did.node) {
        None => (), // This can happen for self, for example
        Some(&ast_map::node_item(item, _)) => traverse_public_item(cx, item),
        Some(&ast_map::node_method(_, impl_id, _)) => traverse_def_id(cx, impl_id),
        Some(&ast_map::node_foreign_item(item, _, _, _)) => {
            cx.rmap.insert(item.id);
        }
        Some(&ast_map::node_variant(ref v, _, _)) => {
            cx.rmap.insert(v.node.id);
        }
        _ => ()
    }
}

fn traverse_public_mod(cx: &ctx, mod_id: node_id, m: &_mod) {
    if !traverse_exports(cx, mod_id) {
        // No exports, so every local item is exported
        for m.items.each |item| {
            traverse_public_item(cx, *item);
        }
    }
}

fn traverse_public_item(cx: &ctx, item: @item) {
    // FIXME #6021: naming rmap shouldn't be necessary
    let rmap: &mut HashSet<node_id> = cx.rmap;
    if rmap.contains(&item.id) { return; }
    rmap.insert(item.id);
    match item.node {
      item_mod(ref m) => traverse_public_mod(cx, item.id, m),
      item_foreign_mod(ref nm) => {
          if !traverse_exports(cx, item.id) {
              for nm.items.each |item| {
                  cx.rmap.insert(item.id);
              }
          }
      }
      item_fn(_, _, _, ref generics, ref blk) => {
        if generics.ty_params.len() > 0u ||
           attr::find_inline_attr(item.attrs) != attr::ia_none {
            traverse_inline_body(cx, blk);
        }
      }
      item_impl(ref generics, _, _, ref ms) => {
        for ms.each |m| {
            if generics.ty_params.len() > 0u ||
                m.generics.ty_params.len() > 0u ||
                attr::find_inline_attr(m.attrs) != attr::ia_none
            {
                cx.rmap.insert(m.id);
                traverse_inline_body(cx, &m.body);
            }
        }
      }
      item_struct(ref struct_def, ref generics) => {
        for struct_def.ctor_id.each |&ctor_id| {
            cx.rmap.insert(ctor_id);
        }
        for struct_def.dtor.each |dtor| {
            cx.rmap.insert(dtor.node.id);
            if generics.ty_params.len() > 0u ||
                attr::find_inline_attr(dtor.node.attrs) != attr::ia_none
            {
                traverse_inline_body(cx, &dtor.node.body);
            }
        }
      }
      item_ty(t, _) => {
        traverse_ty(t, cx,
                    visit::mk_vt(@visit::Visitor {visit_ty: traverse_ty,
                                                  ..*visit::default_visitor()}))
      }
      item_const(*) |
      item_enum(*) | item_trait(*) => (),
      item_mac(*) => fail!(~"item macros unimplemented")
    }
}

fn traverse_ty<'a, 'b>(ty: @Ty, cx: &'b ctx<'a>, v: visit::vt<&'b ctx<'a>>) {
    // FIXME #6021: naming rmap shouldn't be necessary
    let rmap: &mut HashSet<node_id> = cx.rmap;
    if rmap.contains(&ty.id) { return; }
    rmap.insert(ty.id);

    match ty.node {
      ty_path(p, p_id) => {
        match cx.tcx.def_map.find(&p_id) {
          // Kind of a hack to check this here, but I'm not sure what else
          // to do
          Some(&def_prim_ty(_)) => { /* do nothing */ }
          Some(&d) => traverse_def_id(cx, def_id_of_def(d)),
          None    => { /* do nothing -- but should we fail here? */ }
        }
        for p.types.each |t| {
            (v.visit_ty)(*t, cx, v);
        }
      }
      _ => visit::visit_ty(ty, cx, v)
    }
}

fn traverse_inline_body(cx: &ctx, body: &blk) {
    fn traverse_expr<'a, 'b>(e: @expr, cx: &'b ctx<'a>,
                             v: visit::vt<&'b ctx<'a>>) {
        match e.node {
          expr_path(_) => {
            match cx.tcx.def_map.find(&e.id) {
                Some(&d) => {
                  traverse_def_id(cx, def_id_of_def(d));
                }
                None      => cx.tcx.sess.span_bug(e.span, fmt!("Unbound node \
                  id %? while traversing %s", e.id,
                  expr_to_str(e, cx.tcx.sess.intr())))
            }
          }
          expr_field(_, _, _) => {
            match cx.method_map.find(&e.id) {
              Some(&typeck::method_map_entry {
                  origin: typeck::method_static(did),
                  _
                }) => {
                traverse_def_id(cx, did);
              }
              _ => ()
            }
          }
          expr_method_call(*) => {
            match cx.method_map.find(&e.id) {
              Some(&typeck::method_map_entry {
                  origin: typeck::method_static(did),
                  _
                }) => {
                traverse_def_id(cx, did);
              }
              Some(_) => {}
              None => {
                cx.tcx.sess.span_bug(e.span, ~"expr_method_call not in \
                                               method map");
              }
            }
          }
          _ => ()
        }
        visit::visit_expr(e, cx, v);
    }
    // Don't ignore nested items: for example if a generic fn contains a
    // generic impl (as in deque::create), we need to monomorphize the
    // impl as well
    fn traverse_item(i: @item, cx: &ctx, _v: visit::vt<&ctx>) {
      traverse_public_item(cx, i);
    }
    visit::visit_block(body, cx, visit::mk_vt(@visit::Visitor {
        visit_expr: traverse_expr,
        visit_item: traverse_item,
         ..*visit::default_visitor()
    }));
}

fn traverse_all_resources_and_impls(cx: &ctx, crate_mod: &_mod) {
    visit::visit_mod(
        crate_mod,
        codemap::dummy_sp(),
        0,
        cx,
        visit::mk_vt(@visit::Visitor {
            visit_expr: |_e, _cx, _v| { },
            visit_item: |i, cx, v| {
                visit::visit_item(i, cx, v);
                match i.node {
                    item_struct(sdef, _) if sdef.dtor.is_some() => {
                        traverse_public_item(cx, i);
                    }
                    item_impl(*) => {
                        traverse_public_item(cx, i);
                    }
                    _ => ()
                }
            },
            ..*visit::default_visitor()
        }));
}

