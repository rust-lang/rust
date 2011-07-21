import syntax::ast::*;
import syntax::walk;
import std::ivec;
import std::option::*;
import aux::*;
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
import tstate::ann::set_in_poststate_;
import tstate::ann::clear_in_poststate;
import tstate::ann::clear_in_prestate;
import tstate::ann::clear_in_poststate_;
import tritv::*;

fn bit_num(&fn_ctxt fcx, &tsconstr c) -> uint {
    auto d = tsconstr_to_def_id(c);
    assert (fcx.enclosing.constrs.contains_key(d));
    auto rslt = fcx.enclosing.constrs.get(d);
    alt (c) {
        case (ninit(_,_)) {
            alt (rslt) {
                case (cinit(?n, _, _)) { ret n; }
                case (_) {
                    fcx.ccx.tcx.sess.bug("bit_num: asked for init constraint,"
                                             + " found a pred constraint");
                }
            }
        }
        case (npred(_, _, ?args)) {
            alt (rslt) {
                case (cpred(_, ?descs)) {
                    ret match_args(fcx, descs, args);
                }
                case (_) {
                    fcx.ccx.tcx.sess.bug("bit_num: asked for pred constraint,"
                                             + " found an init constraint");
                }
            }
        }
    }
}

fn promises(&fn_ctxt fcx, &poststate p, &tsconstr c) -> bool {
    ret promises_(bit_num(fcx, c), p);
}

fn promises_(uint n, &poststate p) -> bool {
    ret tritv_get(p, n) == ttrue;
}

// v "happens after" u
fn seq_trit(trit u, trit v) -> trit {
    alt (v) {
        case (ttrue)     { ttrue }
        case (tfalse)    { tfalse }
        case (dont_care) { u }
    }
}

// idea: q "happens after" p -- so if something is
// 1 in q and 0 in p, it's 1 in the result; however,
// if it's 0 in q and 1 in p, it's 0 in the result
fn seq_tritv(&postcond p, &postcond q) {
    auto i = 0u;
    assert (p.nbits == q.nbits);
    while (i < p.nbits) {
        tritv_set(i, p, seq_trit(tritv_get(p, i), tritv_get(q, i)));
        i += 1u;
    }
}

fn seq_postconds(&fn_ctxt fcx, &postcond[] ps) -> postcond {
    auto sz = ivec::len(ps);
    if (sz >= 1u) {
        auto prev = tritv_clone(ps.(0));
        for (postcond p in ivec::slice(ps, 1u, sz)) {
            seq_tritv(prev, p);
        }
        ret prev;
    }
    else {
        ret ann::empty_poststate(num_constraints(fcx.enclosing));
    }
}

// Given a list of pres and posts for exprs e0 ... en,
// return the precondition for evaluating each expr in order.
// So, if e0's post is {x} and e1's pre is {x, y, z}, the entire
// precondition shouldn't include x.
fn seq_preconds(&fn_ctxt fcx, &pre_and_post[] pps) -> precond {
    let uint sz = ivec::len(pps);
    let uint num_vars = num_constraints(fcx.enclosing);

    fn seq_preconds_go(&fn_ctxt fcx, &pre_and_post[] pps,
                       &pre_and_post first)
        -> precond {
        let uint sz = ivec::len(pps);
        if (sz >= 1u) {
            auto second = pps.(0);
            assert (pps_len(second) == num_constraints(fcx.enclosing));
            auto second_pre = clone(second.precondition);
            difference(second_pre, first.postcondition);
            auto next_first = clone(first.precondition);
            union(next_first, second_pre);
            auto next_first_post = clone(first.postcondition);
            seq_tritv(next_first_post, second.postcondition);
            ret seq_preconds_go(fcx, ivec::slice(pps, 1u, sz),
                                @rec(precondition=next_first,
                                     postcondition=next_first_post));
        }
        else {
            ret first.precondition;
        }
    }

    if (sz >= 1u) {
        auto first = pps.(0);
        assert (pps_len(first) == num_vars);
        ret seq_preconds_go(fcx, ivec::slice(pps, 1u, sz), first);
    } else { ret true_precond(num_vars); }
}

fn intersect_states(&prestate p, &prestate q) -> prestate {
    auto rslt = tritv_clone(p);
    tritv_intersect(rslt, q);
    ret rslt;
}

fn gen(&fn_ctxt fcx, node_id id, &tsconstr c) -> bool {
    ret set_in_postcond(bit_num(fcx, c),
                        node_id_to_ts_ann(fcx.ccx, id).conditions);
}

fn declare_var(&fn_ctxt fcx, &tsconstr c, prestate pre) -> prestate {
    auto rslt = clone(pre);
    relax_prestate(bit_num(fcx, c), rslt);
    // idea is this is scoped
    relax_poststate(bit_num(fcx, c), rslt);
    ret rslt;
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

fn gen_poststate(&fn_ctxt fcx, node_id id, &tsconstr c) -> bool {
    log "gen_poststate";
    ret set_in_poststate(bit_num(fcx, c),
                         node_id_to_ts_ann(fcx.ccx, id).states);
}

fn kill_prestate(&fn_ctxt fcx, node_id id, &tsconstr c) -> bool {
    ret clear_in_prestate(bit_num(fcx, c),
                           node_id_to_ts_ann(fcx.ccx, id).states);
}

fn kill_poststate(&fn_ctxt fcx, node_id id, &tsconstr c) -> bool {
    log "kill_poststate";
    ret clear_in_poststate(bit_num(fcx, c),
                           node_id_to_ts_ann(fcx.ccx, id).states);
}

fn clear_in_poststate_expr(&fn_ctxt fcx, &@expr e, &poststate t) {
    alt (e.node) {
        case (expr_path(?p)) {
            alt (ivec::last(p.node.idents)) {
                case (some(?i)) {
                    alt (local_node_id_to_def(fcx, e.id)) {
                        case (some(def_local(?d_id))) {
                            clear_in_poststate_(
                               bit_num(fcx,ninit(d_id._1, i)), t);
                        }
                        case (some(_)) { /* ignore args (for now...) */ }
                        case (_) {
                            fcx.ccx.tcx.sess.bug("clear_in_poststate_expr: \
                                   unbound var"); }
                        }
                }
                case (_) { fcx.ccx.tcx.sess.bug("clear_in_poststate_expr"); }
            }
        }
        case (_) { /* do nothing */ }
    }
}

fn set_in_poststate_ident(&fn_ctxt fcx, &node_id id, &ident ident,
                          &poststate t) -> bool {
    ret set_in_poststate_(bit_num(fcx, ninit(id, ident)), t);
}

fn set_in_prestate_constr(&fn_ctxt fcx, &tsconstr c, &prestate t) -> bool {
    ret set_in_poststate_(bit_num(fcx, c), t);
}

fn clear_in_poststate_ident(&fn_ctxt fcx, &node_id id, &ident ident,
                            &node_id parent) -> bool {
    ret kill_poststate(fcx, parent, ninit(id, ident));
}

fn clear_in_prestate_ident(&fn_ctxt fcx, &node_id id, &ident ident,
                            &node_id parent) -> bool {
    ret kill_prestate(fcx, parent, ninit(id, ident));
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
