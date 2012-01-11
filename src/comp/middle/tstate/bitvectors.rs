import syntax::ast::*;
import syntax::visit;
import vec;
import option::*;
import aux::*;
import tstate::ann::{pre_and_post, precond, postcond, prestate, poststate,
                     relax_prestate, relax_precond, relax_poststate,
                     pps_len, true_precond,
                     difference, union, clone,
                     set_in_postcond, set_in_poststate, set_in_poststate_,
                     clear_in_poststate, clear_in_prestate,
                     clear_in_poststate_};
import tritv::*;
import util::common::*;
import driver::session::session;

fn bit_num(fcx: fn_ctxt, c: tsconstr) -> uint {
    let d = tsconstr_to_def_id(c);
    assert (fcx.enclosing.constrs.contains_key(d));
    let rslt = fcx.enclosing.constrs.get(d);
    alt c {
      ninit(_, _) {
        alt rslt {
          cinit(n, _, _) { ret n; }
          _ {
            fcx.ccx.tcx.sess.bug("bit_num: asked for init constraint," +
                                     " found a pred constraint");
          }
        }
      }
      npred(_, _, args) {
        alt rslt {
          cpred(_, descs) { ret match_args(fcx, descs, args); }
          _ {
            fcx.ccx.tcx.sess.bug("bit_num: asked for pred constraint," +
                                     " found an init constraint");
          }
        }
      }
    }
}

fn promises(fcx: fn_ctxt, p: poststate, c: tsconstr) -> bool {
    ret promises_(bit_num(fcx, c), p);
}

fn promises_(n: uint, p: poststate) -> bool { ret tritv_get(p, n) == ttrue; }

// v "happens after" u
fn seq_trit(u: trit, v: trit) -> trit {
    alt v { ttrue. { ttrue } tfalse. { tfalse } dont_care. { u } }
}

// idea: q "happens after" p -- so if something is
// 1 in q and 0 in p, it's 1 in the result; however,
// if it's 0 in q and 1 in p, it's 0 in the result
fn seq_tritv(p: postcond, q: postcond) {
    let i = 0u;
    assert (p.nbits == q.nbits);
    while i < p.nbits {
        tritv_set(i, p, seq_trit(tritv_get(p, i), tritv_get(q, i)));
        i += 1u;
    }
}

fn seq_postconds(fcx: fn_ctxt, ps: [postcond]) -> postcond {
    let sz = vec::len(ps);
    if sz >= 1u {
        let prev = tritv_clone(ps[0]);
        for p: postcond in vec::slice(ps, 1u, sz) { seq_tritv(prev, p); }
        ret prev;
    } else { ret ann::empty_poststate(num_constraints(fcx.enclosing)); }
}

// Given a list of pres and posts for exprs e0 ... en,
// return the precondition for evaluating each expr in order.
// So, if e0's post is {x} and e1's pre is {x, y, z}, the entire
// precondition shouldn't include x.
fn seq_preconds(fcx: fn_ctxt, pps: [pre_and_post]) -> precond {
    let sz: uint = vec::len(pps);
    let num_vars: uint = num_constraints(fcx.enclosing);

    fn seq_preconds_go(fcx: fn_ctxt, pps: [pre_and_post], first: pre_and_post)
       -> precond {
        let sz: uint = vec::len(pps);
        if sz >= 1u {
            let second = pps[0];
            assert (pps_len(second) == num_constraints(fcx.enclosing));
            let second_pre = clone(second.precondition);
            difference(second_pre, first.postcondition);
            let next_first = clone(first.precondition);
            union(next_first, second_pre);
            let next_first_post = clone(first.postcondition);
            seq_tritv(next_first_post, second.postcondition);
            ret seq_preconds_go(fcx, vec::slice(pps, 1u, sz),
                                @{precondition: next_first,
                                  postcondition: next_first_post});
        } else { ret first.precondition; }
    }


    if sz >= 1u {
        let first = pps[0];
        assert (pps_len(first) == num_vars);
        ret seq_preconds_go(fcx, vec::slice(pps, 1u, sz), first);
    } else { ret true_precond(num_vars); }
}

fn intersect_states(p: prestate, q: prestate) -> prestate {
    let rslt = tritv_clone(p);
    tritv_intersect(rslt, q);
    ret rslt;
}

fn gen(fcx: fn_ctxt, id: node_id, c: tsconstr) -> bool {
    ret set_in_postcond(bit_num(fcx, c),
                        node_id_to_ts_ann(fcx.ccx, id).conditions);
}

fn declare_var(fcx: fn_ctxt, c: tsconstr, pre: prestate) -> prestate {
    let rslt = clone(pre);
    relax_prestate(bit_num(fcx, c), rslt);
    // idea is this is scoped
    relax_poststate(bit_num(fcx, c), rslt);
    ret rslt;
}

fn relax_precond_expr(e: @expr, cx: relax_ctxt, vt: visit::vt<relax_ctxt>) {
    relax_precond(cx.i as uint, expr_precond(cx.fcx.ccx, e));
    visit::visit_expr(e, cx, vt);
}

fn relax_precond_stmt(s: @stmt, cx: relax_ctxt, vt: visit::vt<relax_ctxt>) {
    relax_precond(cx.i as uint, stmt_precond(cx.fcx.ccx, *s));
    visit::visit_stmt(s, cx, vt);
}

type relax_ctxt = {fcx: fn_ctxt, i: node_id};

fn relax_precond_block_inner(b: blk, cx: relax_ctxt,
                             vt: visit::vt<relax_ctxt>) {
    relax_precond(cx.i as uint, block_precond(cx.fcx.ccx, b));
    visit::visit_block(b, cx, vt);
}

fn relax_precond_block(fcx: fn_ctxt, i: node_id, b: blk) {
    let cx = {fcx: fcx, i: i};
    let visitor = visit::default_visitor::<relax_ctxt>();
    visitor =
        @{visit_block: relax_precond_block_inner,
          visit_expr: relax_precond_expr,
          visit_stmt: relax_precond_stmt,
          visit_item:
              fn@(_i: @item, _cx: relax_ctxt, _vt: visit::vt<relax_ctxt>) { },
          visit_fn: bind do_nothing(_, _, _, _, _, _, _)
             with *visitor};
    let v1 = visit::mk_vt(visitor);
    v1.visit_block(b, cx, v1);
}

fn gen_poststate(fcx: fn_ctxt, id: node_id, c: tsconstr) -> bool {
    #debug("gen_poststate");
    ret set_in_poststate(bit_num(fcx, c),
                         node_id_to_ts_ann(fcx.ccx, id).states);
}

fn kill_prestate(fcx: fn_ctxt, id: node_id, c: tsconstr) -> bool {
    ret clear_in_prestate(bit_num(fcx, c),
                          node_id_to_ts_ann(fcx.ccx, id).states);
}

fn kill_all_prestate(fcx: fn_ctxt, id: node_id) {
    tritv::tritv_kill(node_id_to_ts_ann(fcx.ccx, id).states.prestate);
}


fn kill_poststate(fcx: fn_ctxt, id: node_id, c: tsconstr) -> bool {
    #debug("kill_poststate");
    ret clear_in_poststate(bit_num(fcx, c),
                           node_id_to_ts_ann(fcx.ccx, id).states);
}

fn clear_in_poststate_expr(fcx: fn_ctxt, e: @expr, t: poststate) {
    alt e.node {
      expr_path(p) {
        alt vec::last(p.node.idents) {
          some(i) {
            alt local_node_id_to_def(fcx, e.id) {
              some(def_local(d_id, _)) {
                clear_in_poststate_(bit_num(fcx, ninit(d_id.node, i)), t);
              }
              some(_) {/* ignore args (for now...) */ }
              _ {
                fcx.ccx.tcx.sess.bug("clear_in_poststate_expr: \
                                   unbound var");
              }
            }
          }
          _ { fcx.ccx.tcx.sess.bug("clear_in_poststate_expr"); }
        }
      }
      _ {/* do nothing */ }
    }
}

fn kill_poststate_(fcx: fn_ctxt, c: tsconstr, post: poststate) -> bool {
    #debug("kill_poststate_");
    ret clear_in_poststate_(bit_num(fcx, c), post);
}

fn set_in_poststate_ident(fcx: fn_ctxt, id: node_id, ident: ident,
                          t: poststate) -> bool {
    ret set_in_poststate_(bit_num(fcx, ninit(id, ident)), t);
}

fn set_in_prestate_constr(fcx: fn_ctxt, c: tsconstr, t: prestate) -> bool {
    ret set_in_poststate_(bit_num(fcx, c), t);
}

fn clear_in_poststate_ident(fcx: fn_ctxt, id: node_id, ident: ident,
                            parent: node_id) -> bool {
    ret kill_poststate(fcx, parent, ninit(id, ident));
}

fn clear_in_prestate_ident(fcx: fn_ctxt, id: node_id, ident: ident,
                           parent: node_id) -> bool {
    ret kill_prestate(fcx, parent, ninit(id, ident));
}

fn clear_in_poststate_ident_(fcx: fn_ctxt, id: node_id, ident: ident,
                             post: poststate) -> bool {
    ret kill_poststate_(fcx, ninit(id, ident), post);
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
