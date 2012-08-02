// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.

import check::{fn_ctxt, lookup_local, methods};
import infer::{resolve_type, resolve_all, force_all};
export resolve_type_vars_in_fn;
export resolve_type_vars_in_expr;

fn resolve_type_vars_in_type(fcx: @fn_ctxt, sp: span, typ: ty::t) ->
    option<ty::t> {
    if !ty::type_needs_infer(typ) { return some(typ); }
    alt resolve_type(fcx.infcx, typ, resolve_all | force_all) {
      result::ok(new_type) { return some(new_type); }
      result::err(e) {
        if !fcx.ccx.tcx.sess.has_errors() {
            fcx.ccx.tcx.sess.span_err(
                sp,
                fmt!{"cannot determine a type \
                      for this expression: %s",
                     infer::fixup_err_to_str(e)})
        }
        return none;
      }
    }
}
fn resolve_type_vars_for_node(wbcx: wb_ctxt, sp: span, id: ast::node_id)
    -> option<ty::t> {
    let fcx = wbcx.fcx, tcx = fcx.ccx.tcx;
    let n_ty = fcx.node_ty(id);
    alt resolve_type_vars_in_type(fcx, sp, n_ty) {
      none {
        wbcx.success = false;
        return none;
      }

      some(t) {
        debug!{"resolve_type_vars_for_node(id=%d, n_ty=%s, t=%s)",
               id, ty_to_str(tcx, n_ty), ty_to_str(tcx, t)};
        write_ty_to_tcx(tcx, id, t);
        alt fcx.opt_node_ty_substs(id) {
          some(substs) {
            let mut new_tps = ~[];
            for substs.tps.each |subst| {
                alt resolve_type_vars_in_type(fcx, sp, subst) {
                  some(t) { vec::push(new_tps, t); }
                  none { wbcx.success = false; return none; }
                }
            }
            write_substs_to_tcx(tcx, id, new_tps);
          }
          none {}
        }
        return some(t);
      }
    }
}

fn maybe_resolve_type_vars_for_node(wbcx: wb_ctxt, sp: span,
                                    id: ast::node_id)
    -> option<ty::t> {
    if wbcx.fcx.node_types.contains_key(id) {
        resolve_type_vars_for_node(wbcx, sp, id)
    } else {
        none
    }
}

type wb_ctxt =
    // As soon as we hit an error we have to stop resolving
    // the entire function
    {fcx: @fn_ctxt, mut success: bool};
type wb_vt = visit::vt<wb_ctxt>;

fn visit_stmt(s: @ast::stmt, wbcx: wb_ctxt, v: wb_vt) {
    if !wbcx.success { return; }
    resolve_type_vars_for_node(wbcx, s.span, ty::stmt_node_id(s));
    visit::visit_stmt(s, wbcx, v);
}
fn visit_expr(e: @ast::expr, wbcx: wb_ctxt, v: wb_vt) {
    if !wbcx.success { return; }
    resolve_type_vars_for_node(wbcx, e.span, e.id);
    alt e.node {
      ast::expr_fn(_, decl, _, _) |
      ast::expr_fn_block(decl, _, _) {
        do vec::iter(decl.inputs) |input| {
            let r_ty = resolve_type_vars_for_node(wbcx, e.span, input.id);

            // Just in case we never constrained the mode to anything,
            // constrain it to the default for the type in question.
            alt (r_ty, input.mode) {
              (some(t), ast::infer(_)) {
                let tcx = wbcx.fcx.ccx.tcx;
                let m_def = ty::default_arg_mode_for_ty(t);
                ty::set_default_mode(tcx, input.mode, m_def);
              }
              _ {}
            }
        }
      }

      ast::expr_new(_, alloc_id, _) {
        resolve_type_vars_for_node(wbcx, e.span, alloc_id);
      }

      ast::expr_binary(*) | ast::expr_unary(*) | ast::expr_assign_op(*)
        | ast::expr_index(*) {
        maybe_resolve_type_vars_for_node(wbcx, e.span, e.callee_id);
      }

      _ { }
    }
    visit::visit_expr(e, wbcx, v);
}
fn visit_block(b: ast::blk, wbcx: wb_ctxt, v: wb_vt) {
    if !wbcx.success { return; }
    resolve_type_vars_for_node(wbcx, b.span, b.node.id);
    visit::visit_block(b, wbcx, v);
}
fn visit_pat(p: @ast::pat, wbcx: wb_ctxt, v: wb_vt) {
    if !wbcx.success { return; }
    resolve_type_vars_for_node(wbcx, p.span, p.id);
    debug!{"Type for pattern binding %s (id %d) resolved to %s",
           pat_to_str(p), p.id,
           wbcx.fcx.infcx.ty_to_str(
               ty::node_id_to_type(wbcx.fcx.ccx.tcx,
                                   p.id))};
    visit::visit_pat(p, wbcx, v);
}
fn visit_local(l: @ast::local, wbcx: wb_ctxt, v: wb_vt) {
    if !wbcx.success { return; }
    let var_id = lookup_local(wbcx.fcx, l.span, l.node.id);
    let var_ty = ty::mk_var(wbcx.fcx.tcx(), var_id);
    alt resolve_type(wbcx.fcx.infcx, var_ty, resolve_all | force_all) {
      result::ok(lty) {
        debug!{"Type for local %s (id %d) resolved to %s",
               pat_to_str(l.node.pat), l.node.id,
               wbcx.fcx.infcx.ty_to_str(lty)};
        write_ty_to_tcx(wbcx.fcx.ccx.tcx, l.node.id, lty);
      }
      result::err(e) {
        wbcx.fcx.ccx.tcx.sess.span_err(
            l.span,
            fmt!{"cannot determine a type \
                  for this local variable: %s",
                 infer::fixup_err_to_str(e)});
        wbcx.success = false;
      }
    }
    visit::visit_local(l, wbcx, v);
}
fn visit_item(_item: @ast::item, _wbcx: wb_ctxt, _v: wb_vt) {
    // Ignore items
}

fn mk_visitor() -> visit::vt<wb_ctxt> {
    visit::mk_vt(@{visit_item: visit_item,
                   visit_stmt: visit_stmt,
                   visit_expr: visit_expr,
                   visit_block: visit_block,
                   visit_pat: visit_pat,
                   visit_local: visit_local
                   with *visit::default_visitor()})
}

fn resolve_type_vars_in_expr(fcx: @fn_ctxt, e: @ast::expr) -> bool {
    let wbcx = {fcx: fcx, mut success: true};
    let visit = mk_visitor();
    visit.visit_expr(e, wbcx, visit);
    if wbcx.success {
        infer::resolve_borrowings(fcx.infcx);
    }
    return wbcx.success;
}

fn resolve_type_vars_in_fn(fcx: @fn_ctxt,
                           decl: ast::fn_decl,
                           blk: ast::blk) -> bool {
    let wbcx = {fcx: fcx, mut success: true};
    let visit = mk_visitor();
    visit.visit_block(blk, wbcx, visit);
    for decl.inputs.each |arg| {
        resolve_type_vars_for_node(wbcx, arg.ty.span, arg.id);
    }
    if wbcx.success {
        infer::resolve_borrowings(fcx.infcx);
    }
    return wbcx.success;
}
