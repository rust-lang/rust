
import front::ast;
import front::ast::method;
import front::ast::item;
import front::ast::item_fn;
import front::ast::_fn;
import front::ast::obj_field;
import front::ast::_obj;
import front::ast::stmt;
import front::ast::ident;
import front::ast::fn_ident;
import front::ast::node_id;
import front::ast::def_id;
import front::ast::local_def;
import front::ast::ty_param;
import front::ast::crate;
import front::ast::return;
import front::ast::noreturn;
import front::ast::expr;
import middle::ty::type_is_nil;
import middle::ty::ret_ty_of_fn;
import util::common::span;
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
import std::vec::map;
import std::vec;
import std::vec::slice;
import std::vec::unzip;
import std::vec::plus_option;
import std::vec::cat_options;
import std::option;
import std::option::t;
import std::option::some;
import std::option::none;
import aux::fn_ctxt;
import aux::crate_ctxt;
import aux::new_crate_ctxt;
import aux::expr_precond;
import aux::expr_prestate;
import aux::expr_poststate;
import aux::stmt_poststate;
import aux::stmt_to_ann;
import aux::num_constraints;
import aux::fixed_point_states;
import aux::tritv_to_str;
import aux::first_difference_string;
import pretty::pprust::ty_to_str;
import util::common::log_stmt_err;
import aux::log_tritv_err;
import bitvectors::promises;
import annotate::annotate_crate;
import collect_locals::mk_f_to_fn_info;
import pre_post_conditions::fn_pre_post;
import states::find_pre_post_state_fn;

fn check_states_expr(&fn_ctxt fcx, &@expr e) {
    let precond prec = expr_precond(fcx.ccx, e);
    let prestate pres = expr_prestate(fcx.ccx, e);

    /*
    log_err("check_states_expr:");
      util::common::log_expr_err(*e);
      log_err("prec = ");
      log_bitv_err(fcx, prec);
      log_err("pres = ");
      log_bitv_err(fcx, pres);
    */

    if (!implies(pres, prec)) {
        auto s = "";
        auto diff = first_difference_string(fcx, prec, pres);
        s +=
            "Unsatisfied precondition constraint (for example, " + diff +
                ") for expression:\n";
        s += pretty::pprust::expr_to_str(e);
        s += "\nPrecondition:\n";
        s += tritv_to_str(fcx, prec);
        s += "\nPrestate:\n";
        s += tritv_to_str(fcx, pres);
        fcx.ccx.tcx.sess.span_fatal(e.span, s);
    }
}

fn check_states_stmt(&fn_ctxt fcx, &@stmt s) {
    auto a = stmt_to_ann(fcx.ccx, *s);
    let precond prec = ann_precond(a);
    let prestate pres = ann_prestate(a);

    /*
      log_err("check_states_stmt:");
      log_stmt_err(*s);
      log_err("prec = ");
      log_bitv_err(fcx, prec);
      log_err("pres = ");
      log_bitv_err(fcx, pres);
    */

    if (!implies(pres, prec)) {
        auto ss = "";
        auto diff = first_difference_string(fcx, prec, pres);
        ss +=
            "Unsatisfied precondition constraint (for example, " + diff +
                ") for statement:\n";
        ss += pretty::pprust::stmt_to_str(*s);
        ss += "\nPrecondition:\n";
        ss += tritv_to_str(fcx, prec);
        ss += "\nPrestate: \n";
        ss += tritv_to_str(fcx, pres);
        fcx.ccx.tcx.sess.span_fatal(s.span, ss);
    }
}

fn check_states_against_conditions(&fn_ctxt fcx, &_fn f, node_id id,
                                   &span sp, &fn_ident i) {
    /* Postorder traversal instead of pre is important
       because we want the smallest possible erroneous statement
       or expression. */

    let @mutable bool keepgoing = @mutable true;
    
    /* TODO probably should use visit instead */

    fn quit(@mutable bool keepgoing, &@ast::item i) {
        *keepgoing = false;
    }
    fn kg(@mutable bool keepgoing) -> bool { 
        ret *keepgoing;
    }

    auto v = rec (visit_stmt_post=bind check_states_stmt(fcx, _),
                  visit_expr_post=bind check_states_expr(fcx, _),
                  visit_item_pre=bind quit(keepgoing, _),
                  keep_going=bind kg(keepgoing)
                  with walk::default_visitor());

    walk::walk_fn(v, f, sp, i, id);

    /* Finally, check that the return value is initialized */
    auto post = aux::block_poststate(fcx.ccx, f.body);
    let aux::constr_ ret_c = rec(id=fcx.id, c=aux::ninit(fcx.name));
    if (f.proto == ast::proto_fn && !promises(fcx, post, ret_c) &&
            !type_is_nil(fcx.ccx.tcx, ret_ty_of_fn(fcx.ccx.tcx, id)) &&
            f.decl.cf == return) {
        fcx.ccx.tcx.sess.span_note(f.body.span,
                                   "In function " + fcx.name +
                                       ", not all control paths \
                                        return a value");
        fcx.ccx.tcx.sess.span_fatal(f.decl.output.span,
                                  "see declared return type of '" +
                                      ty_to_str(*f.decl.output) + "'");
    } else if (f.decl.cf == noreturn) {

        // check that this really always fails
        // the fcx.id bit means "returns" for a returning fn,
        // "diverges" for a non-returning fn
        if (!promises(fcx, post, ret_c)) {
            fcx.ccx.tcx.sess.span_fatal(f.body.span,
                                      "In non-returning function " + fcx.name
                                          +
                                          ", some control paths may \
                                           return to the caller");
        }
    }
}

fn check_fn_states(&fn_ctxt fcx, &_fn f, node_id id, &span sp, &fn_ident i) {
    /* Compute the pre- and post-states for this function */

    auto g = find_pre_post_state_fn;
    fixed_point_states(fcx, g, f);
    /* Now compare each expr's pre-state to its precondition
       and post-state to its postcondition */

    check_states_against_conditions(fcx, f, id, sp, i);
}

fn fn_states(&crate_ctxt ccx, &_fn f, &span sp, &fn_ident i, node_id id) {
    /* Look up the var-to-bit-num map for this function */

    assert (ccx.fm.contains_key(id));
    auto f_info = ccx.fm.get(id);
    auto name = option::from_maybe("anon", i);
    auto fcx = rec(enclosing=f_info, id=id, name=name, ccx=ccx);
    check_fn_states(fcx, f, id, sp, i);
}

fn check_crate(ty::ctxt cx, @crate crate) {
    let crate_ctxt ccx = new_crate_ctxt(cx);
    /* Build the global map from function id to var-to-bit-num-map */

    mk_f_to_fn_info(ccx, crate);
    /* Add a blank ts_ann for every statement (and expression) */

    annotate_crate(ccx, *crate);
    /* Compute the pre and postcondition for every subexpression */

    auto do_pre_post = walk::default_visitor();
    do_pre_post =
        rec(visit_fn_post=bind fn_pre_post(ccx, _, _, _, _)
            with do_pre_post);
    walk::walk_crate(do_pre_post, *crate);
    /* Check the pre- and postcondition against the pre- and poststate
       for every expression */

    auto do_states = walk::default_visitor();
    do_states =
        rec(visit_fn_post=bind fn_states(ccx, _, _, _, _) with do_states);
    walk::walk_crate(do_states, *crate);
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
