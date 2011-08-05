
import syntax::ast;
import ast::method;
import ast::item;
import ast::item_fn;
import ast::_fn;
import ast::obj_field;
import ast::_obj;
import ast::stmt;
import ast::ident;
import ast::fn_ident;
import ast::node_id;
import ast::def_id;
import ast::local_def;
import ast::ty_param;
import ast::crate;
import ast::return;
import ast::noreturn;
import ast::expr;
import syntax::visit;
import syntax::codemap::span;
import middle::ty::type_is_nil;
import middle::ty::ret_ty_of_fn;
import tstate::ann::ts_ann;
import tstate::ann::empty_poststate;
import tstate::ann::true_precond;
import tstate::ann::true_postcond;
import tstate::ann::false_postcond;
import tstate::ann::precond;
import tstate::ann::postcond;
import tstate::ann::poststate;
import tstate::ann::prestate;
import tstate::ann::implies;
import tstate::ann::ann_precond;
import tstate::ann::ann_prestate;
import std::option;
import std::option::t;
import std::option::some;
import std::option::none;
import aux::*;
import syntax::print::pprust::ty_to_str;
import util::common::log_stmt_err;
import bitvectors::*;
import annotate::annotate_crate;
import collect_locals::mk_f_to_fn_info;
import pre_post_conditions::fn_pre_post;
import states::find_pre_post_state_fn;

fn check_unused_vars(fcx: &fn_ctxt) {

    // FIXME: could be more efficient
    for c: norm_constraint  in constraints(fcx) {
        alt c.c.node {
          ninit(id, v) {
            if !vec_contains(fcx.enclosing.used_vars, id) {
                fcx.ccx.tcx.sess.span_warn(c.c.span, "Unused variable " + v);
            }
          }
          _ {/* ignore pred constraints */ }
        }
    }
}

fn check_states_expr(e: &@expr, fcx: &fn_ctxt, v: &visit::vt[fn_ctxt]) {
    visit::visit_expr(e, fcx, v);

    let prec: precond = expr_precond(fcx.ccx, e);
    let pres: prestate = expr_prestate(fcx.ccx, e);


    /*
    log_err("check_states_expr:");
      util::common::log_expr_err(*e);
      log_err("prec = ");
      log_tritv_err(fcx, prec);
      log_err("pres = ");
      log_tritv_err(fcx, pres);
    */

    if !implies(pres, prec) {
        let s = "";
        let diff = first_difference_string(fcx, prec, pres);
        s +=
            "Unsatisfied precondition constraint (for example, " + diff +
                ") for expression:\n";
        s += syntax::print::pprust::expr_to_str(e);
        s += "\nPrecondition:\n";
        s += tritv_to_str(fcx, prec);
        s += "\nPrestate:\n";
        s += tritv_to_str(fcx, pres);
        fcx.ccx.tcx.sess.span_fatal(e.span, s);
    }
}

fn check_states_stmt(s: &@stmt, fcx: &fn_ctxt, v: &visit::vt[fn_ctxt]) {
    visit::visit_stmt(s, fcx, v);

    let a = stmt_to_ann(fcx.ccx, *s);
    let prec: precond = ann_precond(a);
    let pres: prestate = ann_prestate(a);


    /*
      log_err("check_states_stmt:");
      log_stmt_err(*s);
      log_err("prec = ");
      log_tritv_err(fcx, prec);
      log_err("pres = ");
      log_tritv_err(fcx, pres);
    */

    if !implies(pres, prec) {
        let ss = "";
        let diff = first_difference_string(fcx, prec, pres);
        ss +=
            "Unsatisfied precondition constraint (for example, " + diff +
                ") for statement:\n";
        ss += syntax::print::pprust::stmt_to_str(*s);
        ss += "\nPrecondition:\n";
        ss += tritv_to_str(fcx, prec);
        ss += "\nPrestate: \n";
        ss += tritv_to_str(fcx, pres);
        fcx.ccx.tcx.sess.span_fatal(s.span, ss);
    }
}

fn check_states_against_conditions(fcx: &fn_ctxt, f: &_fn,
                                   tps: &ast::ty_param[], id: node_id,
                                   sp: &span, i: &fn_ident) {
    /* Postorder traversal instead of pre is important
       because we want the smallest possible erroneous statement
       or expression. */

    let visitor = visit::default_visitor[fn_ctxt]();

    visitor =
        @{visit_stmt: check_states_stmt,
          visit_expr: check_states_expr,
          visit_fn: do_nothing with *visitor};
    visit::visit_fn(f, tps, sp, i, id, fcx, visit::mk_vt(visitor));

    /* Check that the return value is initialized */
    let post = aux::block_poststate(fcx.ccx, f.body);
    if f.proto == ast::proto_fn &&
        !promises(fcx, post, fcx.enclosing.i_return) &&
        !type_is_nil(fcx.ccx.tcx, ret_ty_of_fn(fcx.ccx.tcx, id)) &&
        f.decl.cf == return {
        fcx.ccx.tcx.sess.span_err(f.body.span,
                                   "In function " + fcx.name +
                                       ", not all control paths \
                                        return a value");
        fcx.ccx.tcx.sess.span_fatal(f.decl.output.span,
                                    "see declared return type of '" +
                                        ty_to_str(*f.decl.output) + "'");
    } else if (f.decl.cf == noreturn) {
        // check that this really always fails
        // Note that it's ok for i_diverge and i_return to both be true.
        // In fact, i_diverge implies i_return. (But not vice versa!)

        if !promises(fcx, post, fcx.enclosing.i_diverge) {
            fcx.ccx.tcx.sess.span_fatal(f.body.span,
                                        "In non-returning function " +
                                            fcx.name +
                                            ", some control paths may \
                                           return to the caller");
        }
    }

    /* Finally, check for unused variables */
    check_unused_vars(fcx);
}

fn check_fn_states(fcx: &fn_ctxt, f: &_fn, tps: &ast::ty_param[], id: node_id,
                   sp: &span, i: &fn_ident) {
    /* Compute the pre- and post-states for this function */

    // Fixpoint iteration
    while find_pre_post_state_fn(fcx, f) { }

    /* Now compare each expr's pre-state to its precondition
       and post-state to its postcondition */

    check_states_against_conditions(fcx, f, tps, id, sp, i);
}

fn fn_states(f: &_fn, tps: &ast::ty_param[], sp: &span, i: &fn_ident,
             id: node_id, ccx: &crate_ctxt, v: &visit::vt[crate_ctxt]) {
    visit::visit_fn(f, tps, sp, i, id, ccx, v);
    /* Look up the var-to-bit-num map for this function */

    assert (ccx.fm.contains_key(id));
    let f_info = ccx.fm.get(id);
    let name = option::from_maybe("anon", i);
    let fcx = {enclosing: f_info, id: id, name: name, ccx: ccx};
    check_fn_states(fcx, f, tps, id, sp, i);
}

fn check_crate(cx: ty::ctxt, crate: @crate) {
    let ccx: crate_ctxt = new_crate_ctxt(cx);
    /* Build the global map from function id to var-to-bit-num-map */

    mk_f_to_fn_info(ccx, crate);
    /* Add a blank ts_ann for every statement (and expression) */

    annotate_crate(ccx, *crate);
    /* Compute the pre and postcondition for every subexpression */

    let vtor = visit::default_visitor[crate_ctxt]();
    vtor = @{visit_fn: fn_pre_post with *vtor};
    visit::visit_crate(*crate, ccx, visit::mk_vt(vtor));

    /* Check the pre- and postcondition against the pre- and poststate
       for every expression */
    vtor = @{visit_fn: fn_states with *vtor};
    visit::visit_crate(*crate, ccx, visit::mk_vt(vtor));
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
