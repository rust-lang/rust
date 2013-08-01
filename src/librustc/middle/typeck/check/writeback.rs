// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.


use middle::pat_util;
use middle::ty;
use middle::typeck::check::{FnCtxt, SelfInfo};
use middle::typeck::infer::{force_all, resolve_all, resolve_region};
use middle::typeck::infer::resolve_type;
use middle::typeck::infer;
use middle::typeck::{vtable_res, vtable_origin};
use middle::typeck::{vtable_static, vtable_param};
use middle::typeck::method_map_entry;
use middle::typeck::write_substs_to_tcx;
use middle::typeck::write_ty_to_tcx;
use util::ppaux;

use syntax::ast;
use syntax::codemap::span;
use syntax::print::pprust::pat_to_str;
use syntax::visit;

fn resolve_type_vars_in_type(fcx: @mut FnCtxt, sp: span, typ: ty::t)
                          -> Option<ty::t> {
    if !ty::type_needs_infer(typ) { return Some(typ); }
    match resolve_type(fcx.infcx(), typ, resolve_all | force_all) {
        Ok(new_type) => return Some(new_type),
        Err(e) => {
            if !fcx.ccx.tcx.sess.has_errors() {
                fcx.ccx.tcx.sess.span_err(
                    sp,
                    fmt!("cannot determine a type \
                          for this expression: %s",
                         infer::fixup_err_to_str(e)))
            }
            return None;
        }
    }
}

fn resolve_type_vars_in_types(fcx: @mut FnCtxt, sp: span, tys: &[ty::t])
                          -> ~[ty::t] {
    tys.map(|t| {
        match resolve_type_vars_in_type(fcx, sp, *t) {
            Some(t1) => t1,
            None => ty::mk_err()
        }
    })
}

fn resolve_method_map_entry(fcx: @mut FnCtxt, sp: span, id: ast::NodeId) {
    // Resolve any method map entry
    match fcx.inh.method_map.find(&id) {
        None => {}
        Some(mme) => {
            {
                let r = resolve_type_vars_in_type(fcx, sp, mme.self_ty);
                foreach t in r.iter() {
                    let method_map = fcx.ccx.method_map;
                    let new_entry = method_map_entry { self_ty: *t, ..*mme };
                    debug!("writeback::resolve_method_map_entry(id=%?, \
                            new_entry=%?)",
                           id, new_entry);
                    method_map.insert(id, new_entry);
                }
            }
        }
    }
}

fn resolve_vtable_map_entry(fcx: @mut FnCtxt, sp: span, id: ast::NodeId) {
    // Resolve any method map entry
    match fcx.inh.vtable_map.find(&id) {
        None => {}
        Some(origins) => {
            let r_origins = resolve_origins(fcx, sp, *origins);
            let vtable_map = fcx.ccx.vtable_map;
            vtable_map.insert(id, r_origins);
            debug!("writeback::resolve_vtable_map_entry(id=%d, vtables=%?)",
                   id, r_origins.repr(fcx.tcx()));
        }
    }

    fn resolve_origins(fcx: @mut FnCtxt, sp: span,
                       vtbls: vtable_res) -> vtable_res {
        @vtbls.map(|os| @os.map(|o| resolve_origin(fcx, sp, o)))
    }

    fn resolve_origin(fcx: @mut FnCtxt,
                      sp: span,
                      origin: &vtable_origin) -> vtable_origin {
        match origin {
            &vtable_static(def_id, ref tys, origins) => {
                let r_tys = resolve_type_vars_in_types(fcx, sp, *tys);
                let r_origins = resolve_origins(fcx, sp, origins);
                vtable_static(def_id, r_tys, r_origins)
            }
            &vtable_param(n, b) => {
                vtable_param(n, b)
            }
        }
    }
}

fn resolve_type_vars_for_node(wbcx: @mut WbCtxt, sp: span, id: ast::NodeId)
                           -> Option<ty::t> {
    let fcx = wbcx.fcx;
    let tcx = fcx.ccx.tcx;

    // Resolve any borrowings for the node with id `id`
    match fcx.inh.adjustments.find(&id) {
        None => (),

        Some(&@ty::AutoAddEnv(r, s)) => {
            match resolve_region(fcx.infcx(), r, resolve_all | force_all) {
                Err(e) => {
                    // This should not, I think, happen:
                    fcx.ccx.tcx.sess.span_err(
                        sp, fmt!("cannot resolve bound for closure: %s",
                                 infer::fixup_err_to_str(e)));
                }
                Ok(r1) => {
                    let resolved_adj = @ty::AutoAddEnv(r1, s);
                    debug!("Adjustments for node %d: %?", id, resolved_adj);
                    fcx.tcx().adjustments.insert(id, resolved_adj);
                }
            }
        }

        Some(&@ty::AutoDerefRef(adj)) => {
            let fixup_region = |r| {
                match resolve_region(fcx.infcx(), r, resolve_all | force_all) {
                    Ok(r1) => r1,
                    Err(e) => {
                        // This should not, I think, happen.
                        fcx.ccx.tcx.sess.span_err(
                            sp, fmt!("cannot resolve scope of borrow: %s",
                                     infer::fixup_err_to_str(e)));
                        r
                    }
                }
            };

            let resolved_autoref = match adj.autoref {
                None => None,
                Some(ref r) => Some(r.map_region(fixup_region))
            };

            let resolved_adj = @ty::AutoDerefRef(ty::AutoDerefRef {
                autoderefs: adj.autoderefs,
                autoref: resolved_autoref,
            });
            debug!("Adjustments for node %d: %?", id, resolved_adj);
            fcx.tcx().adjustments.insert(id, resolved_adj);
        }
    }

    // Resolve the type of the node with id `id`
    let n_ty = fcx.node_ty(id);
    match resolve_type_vars_in_type(fcx, sp, n_ty) {
      None => {
        wbcx.success = false;
        return None;
      }

      Some(t) => {
        debug!("resolve_type_vars_for_node(id=%d, n_ty=%s, t=%s)",
               id, ppaux::ty_to_str(tcx, n_ty), ppaux::ty_to_str(tcx, t));
        write_ty_to_tcx(tcx, id, t);
        for fcx.opt_node_ty_substs(id) |substs| {
          let mut new_tps = ~[];
          foreach subst in substs.tps.iter() {
              match resolve_type_vars_in_type(fcx, sp, *subst) {
                Some(t) => new_tps.push(t),
                None => { wbcx.success = false; return None; }
              }
          }
          write_substs_to_tcx(tcx, id, new_tps);
        }
        return Some(t);
      }
    }
}

fn maybe_resolve_type_vars_for_node(wbcx: @mut WbCtxt,
                                    sp: span,
                                    id: ast::NodeId)
                                 -> Option<ty::t> {
    if wbcx.fcx.inh.node_types.contains_key(&id) {
        resolve_type_vars_for_node(wbcx, sp, id)
    } else {
        None
    }
}

struct WbCtxt {
    fcx: @mut FnCtxt,

    // As soon as we hit an error we have to stop resolving
    // the entire function.
    success: bool,
}

type wb_vt = visit::vt<@mut WbCtxt>;

fn visit_stmt(s: @ast::stmt, (wbcx, v): (@mut WbCtxt, wb_vt)) {
    if !wbcx.success { return; }
    resolve_type_vars_for_node(wbcx, s.span, ty::stmt_node_id(s));
    visit::visit_stmt(s, (wbcx, v));
}

fn visit_expr(e: @ast::expr, (wbcx, v): (@mut WbCtxt, wb_vt)) {
    if !wbcx.success {
        return;
    }

    resolve_type_vars_for_node(wbcx, e.span, e.id);

    resolve_method_map_entry(wbcx.fcx, e.span, e.id);
    {
        let r = e.get_callee_id();
        foreach callee_id in r.iter() {
            resolve_method_map_entry(wbcx.fcx, e.span, *callee_id);
        }
    }

    resolve_vtable_map_entry(wbcx.fcx, e.span, e.id);
    {
        let r = e.get_callee_id();
        foreach callee_id in r.iter() {
            resolve_vtable_map_entry(wbcx.fcx, e.span, *callee_id);
        }
    }

    match e.node {
        ast::expr_fn_block(ref decl, _) => {
            foreach input in decl.inputs.iter() {
                let _ = resolve_type_vars_for_node(wbcx, e.span, input.id);
            }
        }

        ast::expr_binary(callee_id, _, _, _) |
        ast::expr_unary(callee_id, _, _) |
        ast::expr_assign_op(callee_id, _, _, _) |
        ast::expr_index(callee_id, _, _) => {
            maybe_resolve_type_vars_for_node(wbcx, e.span, callee_id);
        }

        ast::expr_method_call(callee_id, _, _, _, _, _) => {
            // We must always have written in a callee ID type for these.
            resolve_type_vars_for_node(wbcx, e.span, callee_id);
        }

        _ => ()
    }

    visit::visit_expr(e, (wbcx, v));
}

fn visit_block(b: &ast::Block, (wbcx, v): (@mut WbCtxt, wb_vt)) {
    if !wbcx.success {
        return;
    }

    resolve_type_vars_for_node(wbcx, b.span, b.id);
    visit::visit_block(b, (wbcx, v));
}

fn visit_pat(p: @ast::pat, (wbcx, v): (@mut WbCtxt, wb_vt)) {
    if !wbcx.success {
        return;
    }

    resolve_type_vars_for_node(wbcx, p.span, p.id);
    debug!("Type for pattern binding %s (id %d) resolved to %s",
           pat_to_str(p, wbcx.fcx.ccx.tcx.sess.intr()), p.id,
           wbcx.fcx.infcx().ty_to_str(
               ty::node_id_to_type(wbcx.fcx.ccx.tcx,
                                   p.id)));
    visit::visit_pat(p, (wbcx, v));
}

fn visit_local(l: @ast::Local, (wbcx, v): (@mut WbCtxt, wb_vt)) {
    if !wbcx.success { return; }
    let var_ty = wbcx.fcx.local_ty(l.span, l.id);
    match resolve_type(wbcx.fcx.infcx(), var_ty, resolve_all | force_all) {
        Ok(lty) => {
            debug!("Type for local %s (id %d) resolved to %s",
                   pat_to_str(l.pat, wbcx.fcx.tcx().sess.intr()),
                   l.id,
                   wbcx.fcx.infcx().ty_to_str(lty));
            write_ty_to_tcx(wbcx.fcx.ccx.tcx, l.id, lty);
        }
        Err(e) => {
            wbcx.fcx.ccx.tcx.sess.span_err(
                l.span,
                fmt!("cannot determine a type \
                      for this local variable: %s",
                     infer::fixup_err_to_str(e)));
            wbcx.success = false;
        }
    }
    visit::visit_local(l, (wbcx, v));
}
fn visit_item(_item: @ast::item, (_wbcx, _v): (@mut WbCtxt, wb_vt)) {
    // Ignore items
}

fn mk_visitor() -> visit::vt<@mut WbCtxt> {
    visit::mk_vt(@visit::Visitor {visit_item: visit_item,
                                  visit_stmt: visit_stmt,
                                  visit_expr: visit_expr,
                                  visit_block: visit_block,
                                  visit_pat: visit_pat,
                                  visit_local: visit_local,
                                  .. *visit::default_visitor()})
}

pub fn resolve_type_vars_in_expr(fcx: @mut FnCtxt, e: @ast::expr) -> bool {
    let wbcx = @mut WbCtxt { fcx: fcx, success: true };
    let visit = mk_visitor();
    (visit.visit_expr)(e, (wbcx, visit));
    return wbcx.success;
}

pub fn resolve_type_vars_in_fn(fcx: @mut FnCtxt,
                               decl: &ast::fn_decl,
                               blk: &ast::Block,
                               self_info: Option<SelfInfo>) -> bool {
    let wbcx = @mut WbCtxt { fcx: fcx, success: true };
    let visit = mk_visitor();
    (visit.visit_block)(blk, (wbcx, visit));
    foreach self_info in self_info.iter() {
        resolve_type_vars_for_node(wbcx,
                                   self_info.span,
                                   self_info.self_id);
    }
    foreach arg in decl.inputs.iter() {
        (visit.visit_pat)(arg.pat, (wbcx, visit));
        // Privacy needs the type for the whole pattern, not just each binding
        if !pat_util::pat_is_binding(fcx.tcx().def_map, arg.pat) {
            resolve_type_vars_for_node(wbcx, arg.pat.span, arg.pat.id);
        }
    }
    return wbcx.success;
}
