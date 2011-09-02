import std::vec;
import std::int::str;
import std::str;
import std::option;
import std::option::*;
import std::int;
import std::uint;
import syntax::ast::*;
import syntax::ast_util::*;
import syntax::codemap::span;
import syntax::visit;
import util::common;
import util::common::log_block;
import std::map::new_int_hash;
import std::map::new_uint_hash;
import util::common::log_expr_err;
import util::common::lit_eq;
import syntax::print::pprust::path_to_str;
import tstate::ann::pre_and_post;
import tstate::ann::pre_and_post_state;
import tstate::ann::empty_ann;
import tstate::ann::prestate;
import tstate::ann::poststate;
import tstate::ann::precond;
import tstate::ann::postcond;
import tstate::ann::empty_states;
import tstate::ann::pps_len;
import tstate::ann::set_prestate;
import tstate::ann::set_poststate;
import tstate::ann::set_in_poststate_;
import tstate::ann::extend_prestate;
import tstate::ann::extend_poststate;
import tstate::ann::set_precondition;
import tstate::ann::set_postcondition;
import tstate::ann::set_in_postcond_;
import tstate::ann::ts_ann;
import tstate::ann::clear_in_postcond;
import tstate::ann::clear_in_poststate;
import tstate::ann::clear_in_poststate_;
import tritv::*;
import bitvectors::promises_;

import syntax::print::pprust::constr_args_to_str;
import syntax::print::pprust::constr_arg_to_str;
import syntax::print::pprust::lit_to_str;

// Used to communicate which operands should be invalidated
// to helper functions
tag oper_type {
    oper_move;
    oper_swap;
    oper_assign;
    oper_assign_op;
    oper_pure;
}

/* logging funs */
fn def_id_to_str(d: def_id) -> istr {
    ret int::str(d.crate) + ~"," + int::str(d.node);
}

fn comma_str(args: &[@constr_arg_use]) -> istr {
    let rslt = ~"";
    let comma = false;
    for a: @constr_arg_use in args {
        if comma { rslt += ~", "; } else { comma = true; }
        alt a.node {
          carg_base. { rslt += ~"*"; }
          carg_ident(i) { rslt += i.ident; }
          carg_lit(l) { rslt += lit_to_str(l); }
        }
    }
    ret rslt;
}

fn constraint_to_str(tcx: &ty::ctxt, c: &sp_constr) -> istr {
    alt c.node {
      ninit(_, i) {
        ret ~"init(" + i + ~" [" +
            tcx.sess.span_str(c.span) + ~"])";
      }
      npred(p, _, args) {
        ret path_to_str(p) + ~"(" +
            comma_str(args) + ~")" + ~"[" +
                tcx.sess.span_str(c.span) + ~"]";
      }
    }
}

fn tritv_to_str(fcx: fn_ctxt, v: &tritv::t) -> istr {
    let s = ~"";
    let comma = false;
    for p: norm_constraint in constraints(fcx) {
        alt tritv_get(v, p.bit_num) {
          dont_care. { }
          t {
            s +=
                if comma { ~", " } else { comma = true; ~"" } +
                    if t == tfalse { ~"!" } else { ~"" } +
                    constraint_to_str(fcx.ccx.tcx, p.c);
          }
        }
    }
    ret s;
}

fn log_tritv(fcx: &fn_ctxt, v: &tritv::t) { log tritv_to_str(fcx, v); }

fn first_difference_string(fcx: &fn_ctxt, expected: &tritv::t,
                           actual: &tritv::t) -> istr {
    let s: istr = ~"";
    for c: norm_constraint in constraints(fcx) {
        if tritv_get(expected, c.bit_num) == ttrue &&
               tritv_get(actual, c.bit_num) != ttrue {
            ret constraint_to_str(fcx.ccx.tcx, c.c);
        }
    }
    ret s;
}

fn log_tritv_err(fcx: fn_ctxt, v: tritv::t) { log_err tritv_to_str(fcx, v); }

fn tos(v: &[uint]) -> istr {
    let rslt = ~"";
    for i: uint in v {
        if i == 0u {
            rslt += ~"0";
        } else if i == 1u { rslt += ~"1"; } else { rslt += ~"?"; }
    }
    ret rslt;
}

fn log_cond(v: &[uint]) { log tos(v); }

fn log_cond_err(v: &[uint]) { log_err tos(v); }

fn log_pp(pp: &pre_and_post) {
    let p1 = tritv::to_vec(pp.precondition);
    let p2 = tritv::to_vec(pp.postcondition);
    log "pre:";
    log_cond(p1);
    log "post:";
    log_cond(p2);
}

fn log_pp_err(pp: &pre_and_post) {
    let p1 = tritv::to_vec(pp.precondition);
    let p2 = tritv::to_vec(pp.postcondition);
    log_err "pre:";
    log_cond_err(p1);
    log_err "post:";
    log_cond_err(p2);
}

fn log_states(pp: &pre_and_post_state) {
    let p1 = tritv::to_vec(pp.prestate);
    let p2 = tritv::to_vec(pp.poststate);
    log "prestate:";
    log_cond(p1);
    log "poststate:";
    log_cond(p2);
}

fn log_states_err(pp: &pre_and_post_state) {
    let p1 = tritv::to_vec(pp.prestate);
    let p2 = tritv::to_vec(pp.poststate);
    log_err "prestate:";
    log_cond_err(p1);
    log_err "poststate:";
    log_cond_err(p2);
}

fn print_ident(i: &ident) { log ~" " + i + ~" "; }

fn print_idents(idents: &mutable [ident]) {
    if vec::len::<ident>(idents) == 0u { ret; }
    log ~"an ident: " + vec::pop::<ident>(idents);
    print_idents(idents);
}


/* data structures */

/**********************************************************************/

/* Two different data structures represent constraints in different
 contexts: constraint and norm_constraint.

constraint gets used to record constraints in a table keyed by def_ids.
cinit constraints represent a single constraint, for the initialization
state of a variable; a cpred constraint, with a single operator and a
list of possible argument lists, could represent several constraints at
once.

norm_constraint, in contrast, gets used when handling an instance
of a constraint rather than a definition of a constraint. It can
also be init or pred (ninit or npred), but the npred case just has
a single argument list.

The representation of constraints, where multiple instances of the
same predicate are collapsed into one entry in the table, makes it
easier to look up a specific instance.

Both types are in constrast with the constraint type defined in
syntax::ast, which is for predicate constraints only, and is what
gets generated by the parser. aux and ast share the same type
to represent predicate *arguments* however. This type
(constr_arg_general) is parameterized (see comments in syntax::ast).

Both types store an ident and span, for error-logging purposes.
*/
type pred_args_ = {args: [@constr_arg_use], bit_num: uint};

type pred_args = spanned<pred_args_>;

// The attached node ID is the *defining* node ID
// for this local.
type constr_arg_use = spanned<constr_arg_general_<inst>>;

tag constraint {
    cinit(uint, span, ident);

    // FIXME: really only want it to be mutable during collect_locals.
    // freeze it after that.
    cpred(path, @mutable [pred_args]);
}

// An ninit variant has a node_id because it refers to a local var.
// An npred has a def_id since the definition of the typestate
// predicate need not be local.
// FIXME: would be nice to give both a def_id field,
// and give ninit a constraint saying it's local.
tag tsconstr {
    ninit(node_id, ident);
    npred(path, def_id, [@constr_arg_use]);
}

type sp_constr = spanned<tsconstr>;

type norm_constraint = {bit_num: uint, c: sp_constr};

type constr_map = @std::map::hashmap<def_id, constraint>;

/* Contains stuff that has to be computed up front */
/* For easy access, the fn_info stores two special constraints for each
function.  i_return holds if all control paths in this function terminate
in either a return expression, or an appropriate tail expression.
i_diverge holds if all control paths in this function terminate in a fail
or diverging call.

It might be tempting to use a single constraint C for both properties,
where C represents i_return and !C represents i_diverge. This is
inadvisable, because then the sense of the bit depends on context. If we're
inside a ! function, that reverses the sense of the bit: C would be
i_diverge and !C would be i_return.  That's awkward, because we have to
pass extra context around to functions that shouldn't care.

Okay, suppose C represents i_return and !C represents i_diverge, regardless
of context. Consider this code:

if (foo) { ret; } else { fail; }

C is true in the consequent and false in the alternative. What's T `join`
F, then?  ? doesn't work, because this code should definitely-return if the
context is a returning function (and be definitely-rejected if the context
is a ! function).  F doesn't work, because then the code gets incorrectly
rejected if the context is a returning function. T would work, but it
doesn't make sense for T `join` F to be T (consider init constraints, for
example).;

So we need context. And so it seems clearer to just have separate
constraints.
*/
type fn_info =
    {constrs: constr_map,
     num_constraints: uint,
     cf: controlflow,
     i_return: tsconstr,
     i_diverge: tsconstr,
     /* list, accumulated during pre/postcondition
     computation, of all local variables that may be
     used */
     // Doesn't seem to work without the @ -- bug
     used_vars: @mutable [node_id]};

fn tsconstr_to_def_id(t: &tsconstr) -> def_id {
    alt t { ninit(id, _) { local_def(id) } npred(_, id, _) { id } }
}

fn tsconstr_to_node_id(t: &tsconstr) -> node_id {
    alt t {
      ninit(id, _) { id }
      npred(_, id, _) { fail "tsconstr_to_node_id called on pred constraint" }
    }
}

/* mapping from node ID to typestate annotation */
type node_ann_table = @mutable [mutable ts_ann];


/* mapping from function name to fn_info map */
type fn_info_map = @std::map::hashmap<node_id, fn_info>;

type fn_ctxt =
    {enclosing: fn_info, id: node_id, name: ident, ccx: crate_ctxt};

type crate_ctxt = {tcx: ty::ctxt, node_anns: node_ann_table, fm: fn_info_map};

fn get_fn_info(ccx: &crate_ctxt, id: node_id) -> fn_info {
    assert (ccx.fm.contains_key(id));
    ret ccx.fm.get(id);
}

fn add_node(ccx: &crate_ctxt, i: node_id, a: &ts_ann) {
    let sz = vec::len(*ccx.node_anns);
    if sz <= i as uint {
        vec::grow_mut(*ccx.node_anns, (i as uint) - sz + 1u, empty_ann(0u));
    }
    ccx.node_anns[i] = a;
}

fn get_ts_ann(ccx: &crate_ctxt, i: node_id) -> option::t<ts_ann> {
    if i as uint < vec::len(*ccx.node_anns) {
        ret some::<ts_ann>(ccx.node_anns[i]);
    } else { ret none::<ts_ann>; }
}


/********* utils ********/
fn node_id_to_ts_ann(ccx: &crate_ctxt, id: node_id) -> ts_ann {
    alt get_ts_ann(ccx, id) {
      none. {
        log_err ~"node_id_to_ts_ann: no ts_ann for node_id "
            + int::str(id);
        fail;
      }
      some(t) { ret t; }
    }
}

fn node_id_to_poststate(ccx: &crate_ctxt, id: node_id) -> poststate {
    log "node_id_to_poststate";
    ret node_id_to_ts_ann(ccx, id).states.poststate;
}

fn stmt_to_ann(ccx: &crate_ctxt, s: &stmt) -> ts_ann {
    log "stmt_to_ann";
    alt s.node {
      stmt_decl(_, id) { ret node_id_to_ts_ann(ccx, id); }
      stmt_expr(_, id) { ret node_id_to_ts_ann(ccx, id); }
      stmt_crate_directive(_) {
        log_err "expecting an annotated statement here";
        fail;
      }
    }
}


/* fails if e has no annotation */
fn expr_states(ccx: &crate_ctxt, e: @expr) -> pre_and_post_state {
    log "expr_states";
    ret node_id_to_ts_ann(ccx, e.id).states;
}


/* fails if e has no annotation */
fn expr_pp(ccx: &crate_ctxt, e: @expr) -> pre_and_post {
    log "expr_pp";
    ret node_id_to_ts_ann(ccx, e.id).conditions;
}

fn stmt_pp(ccx: &crate_ctxt, s: &stmt) -> pre_and_post {
    ret stmt_to_ann(ccx, s).conditions;
}


/* fails if b has no annotation */
fn block_pp(ccx: &crate_ctxt, b: &blk) -> pre_and_post {
    log "block_pp";
    ret node_id_to_ts_ann(ccx, b.node.id).conditions;
}

fn clear_pp(pp: pre_and_post) {
    ann::clear(pp.precondition);
    ann::clear(pp.postcondition);
}

fn clear_precond(ccx: &crate_ctxt, id: node_id) {
    let pp = node_id_to_ts_ann(ccx, id);
    ann::clear(pp.conditions.precondition);
}

fn block_states(ccx: &crate_ctxt, b: &blk) -> pre_and_post_state {
    log "block_states";
    ret node_id_to_ts_ann(ccx, b.node.id).states;
}

fn stmt_states(ccx: &crate_ctxt, s: &stmt) -> pre_and_post_state {
    ret stmt_to_ann(ccx, s).states;
}

fn expr_precond(ccx: &crate_ctxt, e: @expr) -> precond {
    ret expr_pp(ccx, e).precondition;
}

fn expr_postcond(ccx: &crate_ctxt, e: @expr) -> postcond {
    ret expr_pp(ccx, e).postcondition;
}

fn expr_prestate(ccx: &crate_ctxt, e: @expr) -> prestate {
    ret expr_states(ccx, e).prestate;
}

fn expr_poststate(ccx: &crate_ctxt, e: @expr) -> poststate {
    ret expr_states(ccx, e).poststate;
}

fn stmt_precond(ccx: &crate_ctxt, s: &stmt) -> precond {
    ret stmt_pp(ccx, s).precondition;
}

fn stmt_postcond(ccx: &crate_ctxt, s: &stmt) -> postcond {
    ret stmt_pp(ccx, s).postcondition;
}

fn states_to_poststate(ss: &pre_and_post_state) -> poststate {
    ret ss.poststate;
}

fn stmt_prestate(ccx: &crate_ctxt, s: &stmt) -> prestate {
    ret stmt_states(ccx, s).prestate;
}

fn stmt_poststate(ccx: &crate_ctxt, s: &stmt) -> poststate {
    ret stmt_states(ccx, s).poststate;
}

fn block_precond(ccx: &crate_ctxt, b: &blk) -> precond {
    ret block_pp(ccx, b).precondition;
}

fn block_postcond(ccx: &crate_ctxt, b: &blk) -> postcond {
    ret block_pp(ccx, b).postcondition;
}

fn block_prestate(ccx: &crate_ctxt, b: &blk) -> prestate {
    ret block_states(ccx, b).prestate;
}

fn block_poststate(ccx: &crate_ctxt, b: &blk) -> poststate {
    ret block_states(ccx, b).poststate;
}

fn set_prestate_ann(ccx: &crate_ctxt, id: node_id, pre: &prestate) -> bool {
    log "set_prestate_ann";
    ret set_prestate(node_id_to_ts_ann(ccx, id), pre);
}

fn extend_prestate_ann(ccx: &crate_ctxt, id: node_id, pre: &prestate) ->
   bool {
    log "extend_prestate_ann";
    ret extend_prestate(node_id_to_ts_ann(ccx, id).states.prestate, pre);
}

fn set_poststate_ann(ccx: &crate_ctxt, id: node_id, post: &poststate) ->
   bool {
    log "set_poststate_ann";
    ret set_poststate(node_id_to_ts_ann(ccx, id), post);
}

fn extend_poststate_ann(ccx: &crate_ctxt, id: node_id, post: &poststate) ->
   bool {
    log "extend_poststate_ann";
    ret extend_poststate(node_id_to_ts_ann(ccx, id).states.poststate, post);
}

fn set_pre_and_post(ccx: &crate_ctxt, id: node_id, pre: &precond,
                    post: &postcond) {
    log "set_pre_and_post";
    let t = node_id_to_ts_ann(ccx, id);
    set_precondition(t, pre);
    set_postcondition(t, post);
}

fn copy_pre_post(ccx: &crate_ctxt, id: node_id, sub: &@expr) {
    log "set_pre_and_post";
    let p = expr_pp(ccx, sub);
    copy_pre_post_(ccx, id, p.precondition, p.postcondition);
}

fn copy_pre_post_(ccx: &crate_ctxt, id: node_id, pre: &prestate,
                  post: &poststate) {
    log "set_pre_and_post";
    let t = node_id_to_ts_ann(ccx, id);
    set_precondition(t, pre);
    set_postcondition(t, post);
}

/* sets all bits to *1* */
fn set_postcond_false(ccx: &crate_ctxt, id: node_id) {
    let p = node_id_to_ts_ann(ccx, id);
    ann::set(p.conditions.postcondition);
}

fn pure_exp(ccx: &crate_ctxt, id: node_id, p: &prestate) -> bool {
    ret set_prestate_ann(ccx, id, p) | set_poststate_ann(ccx, id, p);
}

fn num_constraints(m: fn_info) -> uint { ret m.num_constraints; }

fn new_crate_ctxt(cx: ty::ctxt) -> crate_ctxt {
    let na: [mutable ts_ann] = [mutable];
    ret {tcx: cx, node_anns: @mutable na, fm: @new_int_hash::<fn_info>()};
}

/* Use e's type to determine whether it returns.
 If it has a function type with a ! annotation,
the answer is noreturn. */
fn controlflow_expr(ccx: &crate_ctxt, e: @expr) -> controlflow {
    alt ty::struct(ccx.tcx, ty::node_id_to_type(ccx.tcx, e.id)) {
      ty::ty_fn(_, _, _, cf, _) { ret cf; }
      _ { ret return; }
    }
}

fn constraints_expr(cx: &ty::ctxt, e: @expr) -> [@ty::constr] {
    alt ty::struct(cx, ty::node_id_to_type(cx, e.id)) {
      ty::ty_fn(_, _, _, _, cs) { ret cs; }
      _ { ret []; }
    }
}

fn node_id_to_def_strict(cx: &ty::ctxt, id: node_id) -> def {
    alt cx.def_map.find(id) {
      none. {
        log_err ~"node_id_to_def: node_id "
            + int::str(id) + ~" has no def";
        fail;
      }
      some(d) { ret d; }
    }
}

fn node_id_to_def(ccx: &crate_ctxt, id: node_id) -> option::t<def> {
    ret ccx.tcx.def_map.find(id);
}

fn norm_a_constraint(id: def_id, c: &constraint) -> [norm_constraint] {
    alt c {
      cinit(n, sp, i) {
        ret [{bit_num: n, c: respan(sp, ninit(id.node, i))}];
      }
      cpred(p, descs) {
        let rslt: [norm_constraint] = [];
        for pd: pred_args in *descs {
            rslt +=
                [{bit_num: pd.node.bit_num,
                  c: respan(pd.span, npred(p, id, pd.node.args))}];
        }
        ret rslt;
      }
    }
}


// Tried to write this as an iterator, but I got a
// non-exhaustive match in trans.
fn constraints(fcx: &fn_ctxt) -> [norm_constraint] {
    let rslt: [norm_constraint] = [];
    for each p: @{key: def_id, val: constraint} in
             fcx.enclosing.constrs.items() {
        rslt += norm_a_constraint(p.key, p.val);
    }
    ret rslt;
}

// FIXME
// Would rather take an immutable vec as an argument,
// should freeze it at some earlier point.
fn match_args(fcx: &fn_ctxt, occs: &@mutable [pred_args],
              occ: &[@constr_arg_use]) -> uint {
    log ~"match_args: looking at " +
            constr_args_to_str(fn (i: &inst) -> istr {
                ret i.ident;
            }, occ);
    for pd: pred_args in *occs {
        log ~"match_args: candidate " + pred_args_to_str(pd);
        fn eq(p: &inst, q: &inst) -> bool { ret p.node == q.node; }
        if ty::args_eq(eq, pd.node.args, occ) { ret pd.node.bit_num; }
    }
    fcx.ccx.tcx.sess.bug(~"match_args: no match for occurring args");
}

fn def_id_for_constr(tcx: ty::ctxt, t: node_id) -> def_id {
    alt tcx.def_map.find(t) {
      none. {
        tcx.sess.bug(~"node_id_for_constr: bad node_id "
                     + int::str(t));
      }
      some(def_fn(i, _)) { ret i; }
      _ { tcx.sess.bug(~"node_id_for_constr: pred is not a function"); }
    }
}

fn expr_to_constr_arg(tcx: ty::ctxt, e: &@expr) -> @constr_arg_use {
    alt e.node {
      expr_path(p) {
        alt tcx.def_map.find(e.id) {
          some(def_local(id)) | some(def_arg(id, _)) | some(def_binding(id)) |
          some(def_upvar(id, _, _)) {
            ret @respan(p.span,
                        carg_ident({ident: p.node.idents[0], node: id.node}));
          }
          some(_) {
            tcx.sess.bug(~"exprs_to_constr_args: non-local variable " +
                             ~"as pred arg");
          }
          none {
            tcx.sess.bug(~"exprs_to_constr_args: NONE " +
                             ~"as pred arg");

          }
        }
      }
      expr_lit(l) { ret @respan(e.span, carg_lit(l)); }
      _ {
        tcx.sess.span_fatal(e.span,
                            ~"Arguments to constrained functions must be " +
                                ~"literals or local variables");
      }
    }
}

fn exprs_to_constr_args(tcx: ty::ctxt, args: &[@expr]) -> [@constr_arg_use] {
    let f = bind expr_to_constr_arg(tcx, _);
    let rslt: [@constr_arg_use] = [];
    for e: @expr in args { rslt += [f(e)]; }
    rslt
}

fn expr_to_constr(tcx: ty::ctxt, e: &@expr) -> sp_constr {
    alt e.node {
      expr_call(operator, args) {
        alt operator.node {
          expr_path(p) {
            ret respan(e.span,
                       npred(p, def_id_for_constr(tcx, operator.id),
                             exprs_to_constr_args(tcx, args)));
          }
          _ {
            tcx.sess.span_fatal(operator.span,
                                ~"Internal error: " +
                                    ~" ill-formed operator \
                                            in predicate");
          }
        }
      }
      _ {
        tcx.sess.span_fatal(e.span,
                            ~"Internal error: " + ~" ill-formed predicate");
      }
    }
}

fn pred_args_to_str(p: &pred_args) -> istr {
    ~"<" + uint::str(p.node.bit_num) + ~", " +
        constr_args_to_str(fn (i: &inst) -> istr {
            ret i.ident;
        }, p.node.args)
        + ~">"
}

fn substitute_constr_args(cx: &ty::ctxt, actuals: &[@expr], c: &@ty::constr)
   -> tsconstr {
    let rslt: [@constr_arg_use] = [];
    for a: @constr_arg in c.node.args {
        rslt += [substitute_arg(cx, actuals, a)];
    }
    ret npred(c.node.path, c.node.id, rslt);
}

fn substitute_arg(cx: &ty::ctxt, actuals: &[@expr], a: @constr_arg) ->
   @constr_arg_use {
    let num_actuals = vec::len(actuals);
    alt a.node {
      carg_ident(i) {
        if i < num_actuals {
            ret expr_to_constr_arg(cx, actuals[i]);
        } else {
            cx.sess.span_fatal(a.span, ~"Constraint argument out of bounds");
        }
      }
      carg_base. { ret @respan(a.span, carg_base); }
      carg_lit(l) { ret @respan(a.span, carg_lit(l)); }
    }
}

fn pred_args_matches(pattern: &[constr_arg_general_<inst>], desc: &pred_args)
   -> bool {
    let i = 0u;
    for c: @constr_arg_use in desc.node.args {
        let n = pattern[i];
        alt c.node {
          carg_ident(p) {
            alt n {
              carg_ident(q) { if p.node != q.node { ret false; } }
              _ { ret false; }
            }
          }
          carg_base. { if n != carg_base { ret false; } }
          carg_lit(l) {
            alt n {
              carg_lit(m) { if !lit_eq(l, m) { ret false; } }
              _ { ret false; }
            }
          }
        }
        i += 1u;
    }
    ret true;
}

fn find_instance_(pattern: &[constr_arg_general_<inst>], descs: &[pred_args])
   -> option::t<uint> {
    for d: pred_args in descs {
        if pred_args_matches(pattern, d) { ret some(d.node.bit_num); }
    }
    ret none;
}

type inst = {ident: ident, node: node_id};
type subst = [{from: inst, to: inst}];

fn find_instances(_fcx: &fn_ctxt, subst: &subst, c: &constraint) ->
   [{from: uint, to: uint}] {

    let rslt = [];
    if vec::len(subst) == 0u { ret rslt; }

    alt c {
      cinit(_, _, _) {/* this is dealt with separately */ }
      cpred(p, descs) {
        for d: pred_args in *descs {
            if args_mention(d.node.args, find_in_subst_bool, subst) {
                let old_bit_num = d.node.bit_num;
                let new = replace(subst, d);
                alt find_instance_(new, *descs) {
                  some(d1) { rslt += [{from: old_bit_num, to: d1}]; }
                  _ { }
                }
            }
        }
      }
    }
    rslt
}

fn find_in_subst(id: node_id, s: &subst) -> option::t<inst> {
    for p: {from: inst, to: inst} in s {
        if id == p.from.node { ret some(p.to); }
    }
    ret none;
}

fn find_in_subst_bool(s: &subst, id: node_id) -> bool {
    is_some(find_in_subst(id, s))
}

fn insts_to_str(stuff: &[constr_arg_general_<inst>]) -> istr {
    let rslt = ~"<";
    for i: constr_arg_general_<inst> in stuff {
        rslt +=
            ~" " +
                alt i {
                  carg_ident(p) { p.ident }
                  carg_base. { ~"*" }
                  carg_lit(_) { ~"[lit]" }
                } + ~" ";
    }
    rslt += ~">";
    rslt
}

fn replace(subst: subst, d: pred_args) -> [constr_arg_general_<inst>] {
    let rslt: [constr_arg_general_<inst>] = [];
    for c: @constr_arg_use in d.node.args {
        alt c.node {
          carg_ident(p) {
            alt find_in_subst(p.node, subst) {
              some(new) { rslt += [carg_ident(new)]; }
              _ { rslt += [c.node]; }
            }
          }
          _ {
            //  log_err "##";
            rslt += [c.node];
          }
        }
    }

    /*
    for (constr_arg_general_<tup(ident, def_id)> p in rslt) {
        alt (p) {
            case (carg_ident(?p)) {
                log_err p._0;
            }
            case (_) {}
        }
    }
    */

    ret rslt;
}

fn path_to_ident(cx: &ty::ctxt, p: &path) -> ident {
    alt vec::last(p.node.idents) {
      none. { cx.sess.span_fatal(p.span, ~"Malformed path"); }
      some(i) { ret i; }
    }
}

tag if_ty { if_check; plain_if; }

fn local_node_id_to_def_id_strict(fcx: &fn_ctxt, sp: &span, i: &node_id) ->
   def_id {
    alt local_node_id_to_def(fcx, i) {
      some(def_local(id)) | some(def_arg(id, _)) | some(def_upvar(id, _, _)) {
        ret id;
      }
      some(_) {
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    ~"local_node_id_to_def_id: id \
               isn't a local");
      }
      none. {
        // should really be bug. span_bug()?
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    ~"local_node_id_to_def_id: id \
               is unbound");
      }
    }
}

fn local_node_id_to_def(fcx: &fn_ctxt, i: &node_id) -> option::t<def> {
    fcx.ccx.tcx.def_map.find(i)
}

fn local_node_id_to_def_id(fcx: &fn_ctxt, i: &node_id) -> option::t<def_id> {
    alt local_node_id_to_def(fcx, i) {
      some(def_local(id)) | some(def_arg(id, _)) | some(def_binding(id)) |
      some(def_upvar(id, _, _)) { some(id) }
      _ { none }
    }
}

fn local_node_id_to_local_def_id(fcx: &fn_ctxt, i: &node_id) ->
   option::t<node_id> {
    alt local_node_id_to_def_id(fcx, i) {
      some(did) { some(did.node) }
      _ { none }
    }
}

fn copy_in_postcond(fcx: &fn_ctxt, parent_exp: node_id, dest: inst, src: inst,
                    ty: oper_type) {
    let post =
        node_id_to_ts_ann(fcx.ccx, parent_exp).conditions.postcondition;
    copy_in_poststate_two(fcx, post, post, dest, src, ty);
}

// FIXME refactor
fn copy_in_poststate(fcx: &fn_ctxt, post: &poststate, dest: inst, src: inst,
                     ty: oper_type) {
    copy_in_poststate_two(fcx, post, post, dest, src, ty);
}

// In target_post, set the bits corresponding to copies of any
// constraints mentioning src that are set in src_post, with
// dest substituted for src.
// (This doesn't create any new constraints. If a new, substituted
// constraint isn't already in the bit vector, it's ignored.)
fn copy_in_poststate_two(fcx: &fn_ctxt, src_post: &poststate,
                         target_post: &poststate, dest: inst, src: inst,
                         ty: oper_type) {
    let subst;
    alt ty {
      oper_swap. { subst = [{from: dest, to: src}, {from: src, to: dest}]; }
      oper_assign_op. {
        ret; // Don't do any propagation
      }
      _ { subst = [{from: src, to: dest}]; }
    }


    for each p: @{key: def_id, val: constraint} in
             fcx.enclosing.constrs.items() {
        // replace any occurrences of the src def_id with the
        // dest def_id
        let insts = find_instances(fcx, subst, p.val);
        for p: {from: uint, to: uint} in insts {
            if promises_(p.from, src_post) {
                set_in_poststate_(p.to, target_post);
            }
        }
    }
}


/* FIXME should refactor this better */
fn forget_in_postcond(fcx: &fn_ctxt, parent_exp: node_id, dead_v: node_id) {
    // In the postcondition given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    let d = local_node_id_to_local_def_id(fcx, dead_v);
    alt d {
      some(d_id) {
        for c: norm_constraint in constraints(fcx) {
            if constraint_mentions(fcx, c, d_id) {
                clear_in_postcond(c.bit_num,
                                  node_id_to_ts_ann(fcx.ccx,
                                                    parent_exp).conditions);
            }
        }
      }
      _ { }
    }
}

fn forget_in_postcond_still_init(fcx: &fn_ctxt, parent_exp: node_id,
                                 dead_v: node_id) {
    // In the postcondition given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    let d = local_node_id_to_local_def_id(fcx, dead_v);
    alt d {
      some(d_id) {
        for c: norm_constraint in constraints(fcx) {
            if non_init_constraint_mentions(fcx, c, d_id) {
                clear_in_postcond(c.bit_num,
                                  node_id_to_ts_ann(fcx.ccx,
                                                    parent_exp).conditions);
            }
        }
      }
      _ { }
    }
}

fn forget_in_poststate(fcx: &fn_ctxt, p: &poststate, dead_v: node_id) ->
   bool {
    // In the poststate given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    let d = local_node_id_to_local_def_id(fcx, dead_v);
    let changed = false;
    alt d {
      some(d_id) {
        for c: norm_constraint in constraints(fcx) {
            if constraint_mentions(fcx, c, d_id) {
                changed |= clear_in_poststate_(c.bit_num, p);
            }
        }
      }
      _ { }
    }
    ret changed;
}

fn forget_in_poststate_still_init(fcx: &fn_ctxt, p: &poststate,
                                  dead_v: node_id) -> bool {
    // In the poststate given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    let d = local_node_id_to_local_def_id(fcx, dead_v);
    let changed = false;
    alt d {
      some(d_id) {
        for c: norm_constraint in constraints(fcx) {
            if non_init_constraint_mentions(fcx, c, d_id) {
                changed |= clear_in_poststate_(c.bit_num, p);
            }
        }
      }
      _ { }
    }
    ret changed;
}

fn any_eq(v: &[node_id], d: node_id) -> bool {
    for i: node_id in v { if i == d { ret true; } }
    false
}

fn constraint_mentions(_fcx: &fn_ctxt, c: &norm_constraint, v: node_id) ->
   bool {
    ret alt c.c.node {
          ninit(id, _) { v == id }
          npred(_, _, args) { args_mention(args, any_eq, [v]) }
        };
}

fn non_init_constraint_mentions(_fcx: &fn_ctxt, c: &norm_constraint,
                                v: &node_id) -> bool {
    ret alt c.c.node {
          ninit(_, _) { false }
          npred(_, _, args) { args_mention(args, any_eq, [v]) }
        };
}

fn args_mention<T>(args: &[@constr_arg_use], q: fn(&[T], node_id) -> bool,
                   s: &[T]) -> bool {
    /*
      FIXME
      The following version causes an assertion in trans to fail
      (something about type_is_tup_like)
    fn mentions<T>(&[T] s, &fn(&[T], def_id) -> bool q,
                            &@constr_arg_use a) -> bool {
        alt (a.node) {
            case (carg_ident(?p1)) {
                auto res = q(s, p1._1);
                log_err (res);
                res
                    }
            case (_)               { false }
        }
    }
    ret vec::any(bind mentions(s,q,_), args);
    */

    for a: @constr_arg_use in args {
        alt a.node { carg_ident(p1) { if q(s, p1.node) { ret true; } } _ { } }
    }
    ret false;
}

fn use_var(fcx: &fn_ctxt, v: &node_id) { *fcx.enclosing.used_vars += [v]; }

// FIXME: This should be a function in std::vec::.
fn vec_contains(v: &@mutable [node_id], i: &node_id) -> bool {
    for d: node_id in *v { if d == i { ret true; } }
    ret false;
}

fn op_to_oper_ty(io: init_op) -> oper_type {
    alt io { init_move. { oper_move } _ { oper_assign } }
}

// default function visitor
fn do_nothing<T>(_f: &_fn, _tp: &[ty_param], _sp: &span, _i: &fn_ident,
                 _iid: node_id, _cx: &T, _v: &visit::vt<T>) {
}


fn args_to_constr_args(tcx: &ty::ctxt, args: &[arg],
                       indices:&[@sp_constr_arg<uint>]) -> [@constr_arg_use] {
    let actuals: [@constr_arg_use] = [];
    let num_args = vec::len(args);
    for a:@sp_constr_arg<uint> in indices {
        actuals += [@respan(a.span, alt a.node {
          carg_base. { carg_base }
          carg_ident(i) {
            if i < num_args {
                carg_ident({ident: args[i].ident, node:args[i].id})
            }
            else {
                tcx.sess.span_bug(a.span, ~"Index out of bounds in \
                  constraint arg");
            }
          }
          carg_lit(l) { carg_lit(l) }
        })];
    }
    ret actuals;
}

fn ast_constr_to_ts_constr(tcx: &ty::ctxt, args: &[arg], c: &@constr) ->
   tsconstr {
    let tconstr = ty::ast_constr_to_constr(tcx, c);
    ret npred(tconstr.node.path, tconstr.node.id,
         args_to_constr_args(tcx, args, tconstr.node.args));
}

fn ast_constr_to_sp_constr(tcx: &ty::ctxt, args: &[arg], c: &@constr) ->
   sp_constr {
    let tconstr = ast_constr_to_ts_constr(tcx, args, c);
    ret respan(c.span, tconstr);
}

type binding = {lhs: [inst], rhs: option::t<initializer>};

fn local_to_bindings(loc: &@local) -> binding {
    let lhs = [];
    for each p: @pat in pat_bindings(loc.node.pat) {
        let ident = alt p.node { pat_bind(name) { name } };
        lhs += [{ident: ident, node: p.id}];
    }
    {lhs: lhs, rhs: loc.node.init}
}

fn locals_to_bindings(locals: &[@local]) -> [binding] {
    vec::map(local_to_bindings, locals)
}

fn callee_modes(fcx: &fn_ctxt, callee: node_id) -> [ty::mode] {
    let ty =
        ty::type_autoderef(fcx.ccx.tcx,
                           ty::node_id_to_type(fcx.ccx.tcx, callee));
    alt ty::struct(fcx.ccx.tcx, ty) {
      ty::ty_fn(_, args, _, _, _) | ty::ty_native_fn(_, args, _) {
        let modes = [];
        for arg: ty::arg in args { modes += [arg.mode]; }
        ret modes;
      }
      _ {
        // Shouldn't happen; callee should be ty_fn.
        fcx.ccx.tcx.sess.bug(
            ~"non-fn callee type in callee_modes: " +
            util::ppaux::ty_to_str(fcx.ccx.tcx, ty));
      }
    }
}

fn callee_arg_init_ops(fcx: &fn_ctxt, callee: node_id) -> [init_op] {
    fn mode_to_op(m: &ty::mode) -> init_op {
        alt m { ty::mo_move. { init_move } _ { init_assign } }
    }
    vec::map(mode_to_op, callee_modes(fcx, callee))
}

fn anon_bindings(ops: &[init_op], es: &[@expr]) -> [binding] {
    let bindings: [binding] = [];
    let i = 0;
    for op: init_op in ops {
        bindings += [{lhs: [], rhs: some({op: op, expr: es[i]})}];
        i += 1;
    }
    ret bindings;
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
