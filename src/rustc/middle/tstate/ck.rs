
use syntax::ast;
use ast::{stmt, fn_ident, node_id, crate, return_val, noreturn, expr};
use syntax::{visit, print};
use syntax::codemap::span;
use middle::ty;
use tstate::ann::{precond, prestate,
                     implies, ann_precond, ann_prestate};
use aux::*;

use util::ppaux::ty_to_str;
use bitvectors::*;
use annotate::annotate_crate;
use collect_locals::mk_f_to_fn_info;
use pre_post_conditions::fn_pre_post;
use states::find_pre_post_state_fn;
use syntax::print::pprust::expr_to_str;
use driver::session::session;
use std::map::hashmap;

fn check_states_expr(e: @expr, fcx: fn_ctxt, v: visit::vt<fn_ctxt>) {
    visit::visit_expr(e, fcx, v);

    let prec: precond = expr_precond(fcx.ccx, e);
    let pres: prestate = expr_prestate(fcx.ccx, e);

    if !implies(pres, prec) {
        let mut s = ~"";
        let diff = first_difference_string(fcx, prec, pres);
        s +=
            ~"unsatisfied precondition constraint (for example, " + diff +
                ~") for expression:\n";
        s += syntax::print::pprust::expr_to_str(e);
        s += ~"\nprecondition:\n";
        s += tritv_to_str(fcx, prec);
        s += ~"\nprestate:\n";
        s += tritv_to_str(fcx, pres);
        fcx.ccx.tcx.sess.span_fatal(e.span, s);
    }
}

fn check_states_stmt(s: @stmt, fcx: fn_ctxt, v: visit::vt<fn_ctxt>) {
    visit::visit_stmt(s, fcx, v);

    let a = stmt_to_ann(fcx.ccx, *s);
    let prec: precond = ann_precond(a);
    let pres: prestate = ann_prestate(a);

    debug!("check_states_stmt:");
    log(debug, print::pprust::stmt_to_str(*s));
    debug!("prec = ");
    log_tritv(fcx, prec);
    debug!("pres = ");
    log_tritv(fcx, pres);

    if !implies(pres, prec) {
        let mut ss = ~"";
        let diff = first_difference_string(fcx, prec, pres);
        ss +=
            ~"unsatisfied precondition constraint (for example, " + diff +
                ~") for statement:\n";
        ss += syntax::print::pprust::stmt_to_str(*s);
        ss += ~"\nprecondition:\n";
        ss += tritv_to_str(fcx, prec);
        ss += ~"\nprestate: \n";
        ss += tritv_to_str(fcx, pres);
        fcx.ccx.tcx.sess.span_fatal(s.span, ss);
    }
}

fn check_states_against_conditions(fcx: fn_ctxt,
                                   fk: visit::fn_kind,
                                   f_decl: ast::fn_decl,
                                   f_body: ast::blk,
                                   sp: span,
                                   id: node_id) {
    /* Postorder traversal instead of pre is important
       because we want the smallest possible erroneous statement
       or expression. */
    let visitor = visit::mk_vt(
        @{visit_stmt: check_states_stmt,
          visit_expr: check_states_expr,
          visit_fn: |a,b,c,d,e,f,g| {
              do_nothing::<fn_ctxt>(a, b, c, d, e, f, g)
          }
          with *visit::default_visitor::<fn_ctxt>()});
    visit::visit_fn(fk, f_decl, f_body, sp, id, fcx, visitor);
}

fn check_fn_states(fcx: fn_ctxt,
                   fk: visit::fn_kind,
                   f_decl: ast::fn_decl,
                   f_body: ast::blk,
                   sp: span,
                   id: node_id) {
    /* Compute the pre- and post-states for this function */

    // Fixpoint iteration
    while find_pre_post_state_fn(fcx, f_decl, f_body) { }

    /* Now compare each expr's pre-state to its precondition
       and post-state to its postcondition */

    check_states_against_conditions(fcx, fk, f_decl, f_body, sp, id);
}

fn fn_states(fk: visit::fn_kind, f_decl: ast::fn_decl, f_body: ast::blk,
             sp: span, id: node_id,
             ccx: crate_ctxt, v: visit::vt<crate_ctxt>) {

    visit::visit_fn(fk, f_decl, f_body, sp, id, ccx, v);

    // We may not care about typestate for this function if it contains
    // no constrained calls
    if !ccx.fm.get(id).ignore {
        /* Look up the var-to-bit-num map for this function */

        let f_info = ccx.fm.get(id);
        let name = visit::name_of_fn(fk);
        let fcx = {enclosing: f_info, id: id, name: name, ccx: ccx};
        check_fn_states(fcx, fk, f_decl, f_body, sp, id)
    }
}

fn check_crate(cx: ty::ctxt, crate: @crate) {
    let ccx: crate_ctxt = new_crate_ctxt(cx);
    /* Build the global map from function id to var-to-bit-num-map */

    mk_f_to_fn_info(ccx, crate);
    /* Add a blank ts_ann for every statement (and expression) */

    annotate_crate(ccx, *crate);
    /* Compute the pre and postcondition for every subexpression */

    let vtor = visit::default_visitor::<crate_ctxt>();
    let vtor = @{visit_fn: fn_pre_post with *vtor};
    visit::visit_crate(*crate, ccx, visit::mk_vt(vtor));

    /* Check the pre- and postcondition against the pre- and poststate
       for every expression */
    let vtor = visit::default_visitor::<crate_ctxt>();
    let vtor = @{visit_fn: fn_states with *vtor};
    visit::visit_crate(*crate, ccx, visit::mk_vt(vtor));
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
