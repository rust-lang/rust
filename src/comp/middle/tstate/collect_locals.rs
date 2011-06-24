
import std::vec;
import std::vec::plus_option;
import front::ast;
import front::ast::*;
import pretty::ppaux::fn_ident_to_string;
import std::option::*;
import middle::walk::walk_crate;
import middle::walk::walk_fn;
import middle::walk::ast_visitor;
import aux::cinit;
import aux::ninit;
import aux::npred;
import aux::cpred;
import aux::constraint;
import aux::fn_info;
import aux::crate_ctxt;
import aux::num_constraints;
import aux::constr_map;
import aux::expr_to_constr;
import aux::constraints_expr;
import aux::node_id_to_def_strict;
import util::common::new_int_hash;
import util::common::new_def_hash;
import util::common::uistr;
import util::common::span;
import util::common::respan;

type ctxt = rec(@mutable vec[aux::constr] cs, ty::ctxt tcx);

fn collect_local(&ctxt cx, &@local loc) {
    log "collect_local: pushing " + loc.node.ident;
    vec::push(*cx.cs,
              respan(loc.span, rec(id=loc.node.id, c=ninit(loc.node.ident))));
}

fn collect_pred(&ctxt cx, &@expr e) {
    alt (e.node) {
        case (expr_check(?ch)) {
            vec::push(*cx.cs, expr_to_constr(cx.tcx, ch));
        }
        case (expr_if_check(?ex, _, _)) {
            vec::push(*cx.cs, expr_to_constr(cx.tcx, ex));
        }
        // If it's a call, generate appropriate instances of the
        // call's constraints.
        case (expr_call(?operator, ?operands)) {
            for (@ty::constr_def c in constraints_expr(cx.tcx, operator)) {
                let aux::constr ct = respan(c.span,
                      rec(id=c.node.id._1,
                          c=aux::substitute_constr_args(cx.tcx,
                                                        operands, c)));
                vec::push(*cx.cs, ct);
            }
        }
        case (_) { }
    }
}

fn find_locals(&ty::ctxt tcx, &_fn f, &span sp, &fn_ident i, node_id id)
    -> ctxt {
    let ctxt cx = rec(cs=@mutable vec::alloc(0u), tcx=tcx);
    auto visitor = walk::default_visitor();
    visitor =
        rec(visit_local_pre=bind collect_local(cx, _),
            visit_expr_pre=bind collect_pred(cx, _) with visitor);
    walk_fn(visitor, f, sp, i, id);
    ret cx;
}

fn add_constraint(&ty::ctxt tcx, aux::constr c, uint next, constr_map tbl) ->
   uint {
    log aux::constraint_to_str(tcx, c) + " |-> " + util::common::uistr(next);
    alt (c.node.c) {
        case (ninit(?i)) { tbl.insert(c.node.id, cinit(next, c.span, i)); }
        case (npred(?p, ?args)) {
            alt (tbl.find(c.node.id)) {
                case (some(?ct)) {
                    alt (ct) {
                        case (cinit(_, _, _)) {
                            tcx.sess.bug("add_constraint: same def_id used" +
                                             " as a variable and a pred");
                        }
                        case (cpred(_, ?pds)) {
                            vec::push(*pds,
                                      respan(c.span,
                                             rec(args=args, bit_num=next)));
                        }
                    }
                }
                case (none) {
                    tbl.insert(c.node.id,
                               cpred(p,
                                     @mutable [respan(c.span,
                                                      rec(args=args,
                                                          bit_num=next))]));
                }
            }
        }
    }
    ret next + 1u;
}


/* builds a table mapping each local var defined in f
   to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(&crate_ctxt ccx, &_fn f, &span f_sp, &fn_ident f_name,
              node_id id) {
    auto res_map = @new_int_hash[constraint]();
    let uint next = 0u;
    let vec[arg] f_args = f.decl.inputs;
    /* ignore args, which we know are initialized;
       just collect locally declared vars */

    let ctxt cx = find_locals(ccx.tcx, f, f_sp, f_name, id);
    /* now we have to add bit nums for both the constraints
       and the variables... */

    for (aux::constr c in { *cx.cs }) {
        next = add_constraint(cx.tcx, c, next, res_map);
    }
    /* add a pseudo-entry for the function's return value
       we can safely use the function's name itself for this purpose */

    auto name = fn_ident_to_string(id, f_name);
    add_constraint(cx.tcx, respan(f_sp, rec(id=id, c=ninit(name))), next,
                   res_map);
    auto rslt =
        rec(constrs=res_map,
            num_constraints=vec::len(*cx.cs) + 1u,
            cf=f.decl.cf);
    ccx.fm.insert(id, rslt);
    log name + " has " + uistr(num_constraints(rslt)) + " constraints";
}


/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(&crate_ctxt ccx, @crate c) {
    let ast_visitor vars_visitor = walk::default_visitor();
    vars_visitor =
        rec(visit_fn_pre=bind mk_fn_info(ccx, _, _, _, _)
            with vars_visitor);
    walk_crate(vars_visitor, *c);
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
