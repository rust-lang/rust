import std::uint;
import std::int;
import std::ivec;
import syntax::ast::*;
import util::ppaux::fn_ident_to_string;
import std::option::*;
import syntax::visit;
import aux::*;
import std::map::new_int_hash;
import util::common::new_def_hash;
import syntax::codemap::span;
import syntax::ast::respan;

type ctxt = rec(@mutable (sp_constr[]) cs, ty::ctxt tcx);

fn collect_local(&@local loc, &ctxt cx, &visit::vt[ctxt] v) {
    log "collect_local: pushing " + loc.node.ident;
    *cx.cs += ~[respan(loc.span, ninit(loc.node.id, loc.node.ident))];
    visit::visit_local(loc, cx, v);
}

fn collect_pred(&@expr e, &ctxt cx, &visit::vt[ctxt] v) {
    alt (e.node) {
        case (expr_check(_, ?ch)) {
            *cx.cs += ~[expr_to_constr(cx.tcx, ch)];
        }
        case (expr_if_check(?ex, _, _)) {
            *cx.cs += ~[expr_to_constr(cx.tcx, ex)];
        }
        // If it's a call, generate appropriate instances of the
        // call's constraints.
        case (expr_call(?operator, ?operands)) {
            for (@ty::constr c in constraints_expr(cx.tcx, operator)) {
                let sp_constr ct = respan(c.span,
                           aux::substitute_constr_args(cx.tcx, operands,
                                                       c));
                *cx.cs += ~[ct];
            }
        }
        case (_) { }
    }
    // visit subexpressions
    visit::visit_expr(e, cx, v);
}

fn find_locals(&ty::ctxt tcx, &_fn f, &ty_param[] tps, &span sp, &fn_ident i,
               node_id id) -> ctxt {
    let ctxt cx = rec(cs=@mutable ~[], tcx=tcx);
    auto visitor = visit::default_visitor[ctxt]();

    visitor = @rec(visit_local=collect_local,
                   visit_expr=collect_pred,
                   visit_fn=do_nothing
                   with *visitor);
    visit::visit_fn(f, tps, sp, i, id, cx, visit::mk_vt(visitor));
    ret cx;
}

fn add_constraint(&ty::ctxt tcx, sp_constr c, uint next, constr_map tbl) ->
   uint {
    log constraint_to_str(tcx, c) + " |-> " + std::uint::str(next);
    alt (c.node) {
        case (ninit(?id, ?i)) { tbl.insert(local_def(id),
                                           cinit(next, c.span, i)); }
        case (npred(?p, ?d_id, ?args)) {
            alt (tbl.find(d_id)) {
                case (some(?ct)) {
                    alt (ct) {
                        case (cinit(_, _, _)) {
                            tcx.sess.bug("add_constraint: same def_id used" +
                                             " as a variable and a pred");
                        }
                        case (cpred(_, ?pds)) {
                            *pds += ~[respan(c.span,
                                            rec(args=args, bit_num=next))];
                        }
                    }
                }
                case (none) {
                    let @mutable(pred_args[]) rslt = @mutable(~[respan(c.span,
                                                         rec(args=args,
                                                             bit_num=next))]);
                    tbl.insert(d_id, cpred(p, rslt));
                }
            }
        }
    }
    ret next + 1u;
}


/* builds a table mapping each local var defined in f
   to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(&crate_ctxt ccx, &_fn f, &ty_param[] tp,
              &span f_sp, &fn_ident f_name, node_id id) {
    auto name = fn_ident_to_string(id, f_name);
    auto res_map = @new_def_hash[constraint]();
    let uint next = 0u;

    let ctxt cx = find_locals(ccx.tcx, f, tp, f_sp, f_name, id);
    /* now we have to add bit nums for both the constraints
       and the variables... */

    for (sp_constr c in { *cx.cs }) {
        next = add_constraint(cx.tcx, c, next, res_map);
    }
    /* if this function has any constraints, instantiate them to the
       argument names and add them */
    auto sc;
    for (@constr c in f.decl.constraints) {
        sc = ast_constr_to_sp_constr(cx.tcx, f.decl.inputs, c);
        next = add_constraint(cx.tcx, sc, next, res_map);
    }

    /* add a pseudo-entry for the function's return value
       we can safely use the function's name itself for this purpose */

    add_constraint(cx.tcx, respan(f_sp, ninit(id, name)), next, res_map);
    let @mutable node_id[] v = @mutable ~[];
    auto rslt =
        rec(constrs=res_map,
            num_constraints=ivec::len(*cx.cs) + ivec::len(f.decl.constraints)
                            + 1u,
            cf=f.decl.cf,
            used_vars=v);
    ccx.fm.insert(id, rslt);
    log name + " has " + std::uint::str(num_constraints(rslt)) +
        " constraints";
}


/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(&crate_ctxt ccx, @crate c) {
    auto visitor = visit::mk_simple_visitor
        (@rec(visit_fn=bind mk_fn_info(ccx, _, _, _, _, _)
              with *visit::default_simple_visitor()));
    visit::visit_crate(*c, (), visitor);
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
