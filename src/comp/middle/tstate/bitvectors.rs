
import std::bitv;
import std::vec;
import std::vec::len;
import std::vec::slice;
import front::ast::*;
import aux::fn_ctxt;
import aux::fn_info;
import aux::log_bitv;
import aux::num_constraints;
import aux::cinit;
import aux::cpred;
import aux::ninit;
import aux::npred;
import aux::pred_desc;
import aux::match_args;
import aux::constr_;
import aux::block_precond;
import aux::stmt_precond;
import aux::expr_precond;
import aux::block_prestate;
import aux::expr_prestate;
import aux::stmt_prestate;
import tstate::aux::node_id_to_ts_ann;
import tstate::ann::pre_and_post;
import tstate::ann::precond;
import tstate::ann::postcond;
import tstate::ann::prestate;
import tstate::ann::poststate;
import tstate::ann::relax_prestate;
import tstate::ann::relax_precond;
import tstate::ann::relax_poststate;
import tstate::ann::pps_len;
import tstate::ann::true_precond;
import tstate::ann::empty_prestate;
import tstate::ann::difference;
import tstate::ann::union;
import tstate::ann::intersect;
import tstate::ann::clone;
import tstate::ann::set_in_postcond;
import tstate::ann::set_in_poststate;
import tstate::ann::clear_in_poststate;

fn bit_num(&fn_ctxt fcx, &constr_ c) -> uint {
    assert (fcx.enclosing.constrs.contains_key(c.id));
    auto res = fcx.enclosing.constrs.get(c.id);
    alt (c.c) {
        case (ninit(_)) {
            alt (res) {
                case (cinit(?n, _, _)) { ret n; }
                case (_) {
                    fcx.ccx.tcx.sess.bug("bit_num: asked for init constraint,"
                                             + " found a pred constraint");
                }
            }
        }
        case (npred(_, ?args)) {
            alt (res) {
                case (cpred(_, ?descs)) { ret match_args(fcx, *descs, args); }
                case (_) {
                    fcx.ccx.tcx.sess.bug("bit_num: asked for pred constraint,"
                                             + " found an init constraint");
                }
            }
        }
    }
}

fn promises(&fn_ctxt fcx, &poststate p, &constr_ c) -> bool {
    ret bitv::get(p, bit_num(fcx, c));
}


// Given a list of pres and posts for exprs e0 ... en,
// return the precondition for evaluating each expr in order.
// So, if e0's post is {x} and e1's pre is {x, y, z}, the entire
// precondition shouldn't include x.
fn seq_preconds(fn_ctxt fcx, vec[pre_and_post] pps) -> precond {
    let uint sz = len[pre_and_post](pps);
    let uint num_vars = num_constraints(fcx.enclosing);
    if (sz >= 1u) {
        auto first = pps.(0);
        assert (pps_len(first) == num_vars);
        let precond rest =
            seq_preconds(fcx, slice[pre_and_post](pps, 1u, sz));
        difference(rest, first.postcondition);
        auto res = clone(first.precondition);
        union(res, rest);
        log "seq_preconds:";
        log "first.postcondition =";
        log_bitv(fcx, first.postcondition);
        log "rest =";
        log_bitv(fcx, rest);
        log "returning";
        log_bitv(fcx, res);
        ret res;
    } else { ret true_precond(num_vars); }
}


/* works on either postconds or preconds
 should probably rethink the whole type synonym situation */
fn union_postconds_go(&postcond first, &vec[postcond] rest) -> postcond {
    auto sz = vec::len[postcond](rest);
    if (sz > 0u) {
        auto other = rest.(0);
        union(first, other);
        union_postconds_go(first,
                           slice[postcond](rest, 1u, len[postcond](rest)));
    }
    ret first;
}

fn union_postconds(uint nv, &vec[postcond] pcs) -> postcond {
    if (len[postcond](pcs) > 0u) {
        ret union_postconds_go(bitv::clone(pcs.(0)), pcs);
    } else { ret empty_prestate(nv); }
}


/* Gee, maybe we could use foldl or something */
fn intersect_postconds_go(&postcond first, &vec[postcond] rest) -> postcond {
    auto sz = vec::len[postcond](rest);
    if (sz > 0u) {
        auto other = rest.(0);
        intersect(first, other);
        intersect_postconds_go(first,
                               slice[postcond](rest, 1u,
                                               len[postcond](rest)));
    }
    ret first;
}

fn intersect_postconds(&vec[postcond] pcs) -> postcond {
    assert (len[postcond](pcs) > 0u);
    ret intersect_postconds_go(bitv::clone(pcs.(0)), pcs);
}

fn gen(&fn_ctxt fcx, node_id id, &constr_ c) -> bool {
    ret set_in_postcond(bit_num(fcx, c),
                        node_id_to_ts_ann(fcx.ccx, id).conditions);
}

fn declare_var(&fn_ctxt fcx, &constr_ c, prestate pre) -> prestate {
    auto res = clone(pre);
    relax_prestate(bit_num(fcx, c), res);
    // idea is this is scoped
    relax_poststate(bit_num(fcx, c), res);
    ret res;
}

fn relax_precond_block_non_recursive(&fn_ctxt fcx, node_id i, &block b) {
    relax_precond(i as uint, block_precond(fcx.ccx, b));
}

fn relax_precond_expr(&fn_ctxt fcx, node_id i, &@expr e) {
    relax_precond(i as uint, expr_precond(fcx.ccx, e));
}

fn relax_precond_stmt(&fn_ctxt fcx, node_id i, &@stmt s) {
    relax_precond(i as uint, stmt_precond(fcx.ccx, *s));
}

fn relax_precond_block(&fn_ctxt fcx, node_id i, &block b) {
    relax_precond_block_non_recursive(fcx, i, b);
    // FIXME: should use visit instead
    // could at least generalize this pattern 
    // (also seen in ck::check_states_against_conditions)
    let @mutable bool keepgoing = @mutable true;

    fn quit(@mutable bool keepgoing, &@item i) {
        *keepgoing = false;
    }
    fn kg(@mutable bool keepgoing) -> bool { ret *keepgoing; }

    auto v = rec(visit_block_pre = bind
                    relax_precond_block_non_recursive(fcx, i, _),
                 visit_expr_pre  = bind relax_precond_expr(fcx, i, _),
                 visit_stmt_pre  = bind relax_precond_stmt(fcx, i, _),
                  visit_item_pre=bind quit(keepgoing, _),
                  keep_going=bind kg(keepgoing)

                   with walk::default_visitor());
    walk::walk_block(v, b);
}

fn gen_poststate(&fn_ctxt fcx, node_id id, &constr_ c) -> bool {
    log "gen_poststate";
    ret set_in_poststate(bit_num(fcx, c),
                         node_id_to_ts_ann(fcx.ccx, id).states);
}

fn kill_poststate(&fn_ctxt fcx, node_id id, &constr_ c) -> bool {
    log "kill_poststate";
    ret clear_in_poststate(bit_num(fcx, c),
                           node_id_to_ts_ann(fcx.ccx, id).states);
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
