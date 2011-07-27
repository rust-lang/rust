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

type ctxt = {cs: @mutable sp_constr[], tcx: ty::ctxt};

fn collect_local(loc: &@local, cx: &ctxt, v: &visit::vt[ctxt]) {
    log "collect_local: pushing " + loc.node.ident;
    *cx.cs += ~[respan(loc.span, ninit(loc.node.id, loc.node.ident))];
    visit::visit_local(loc, cx, v);
}

fn collect_pred(e: &@expr, cx: &ctxt, v: &visit::vt[ctxt]) {
    alt e.node {
      expr_check(_, ch) { *cx.cs += ~[expr_to_constr(cx.tcx, ch)]; }
      expr_if_check(ex, _, _) { *cx.cs += ~[expr_to_constr(cx.tcx, ex)]; }

      // If it's a call, generate appropriate instances of the
      // call's constraints.
      expr_call(operator, operands) {
        for c: @ty::constr  in constraints_expr(cx.tcx, operator) {
            let ct: sp_constr =
                respan(c.span,
                       aux::substitute_constr_args(cx.tcx, operands, c));
            *cx.cs += ~[ct];
        }
      }
      _ { }
    }
    // visit subexpressions
    visit::visit_expr(e, cx, v);
}

fn find_locals(tcx: &ty::ctxt, f: &_fn, tps: &ty_param[], sp: &span,
               i: &fn_ident, id: node_id) -> ctxt {
    let cx: ctxt = {cs: @mutable ~[], tcx: tcx};
    let visitor = visit::default_visitor[ctxt]();

    visitor =
        @{visit_local: collect_local,
          visit_expr: collect_pred,
          visit_fn: do_nothing with *visitor};
    visit::visit_fn(f, tps, sp, i, id, cx, visit::mk_vt(visitor));
    ret cx;
}

fn add_constraint(tcx: &ty::ctxt, c: sp_constr, next: uint, tbl: constr_map)
   -> uint {
    log constraint_to_str(tcx, c) + " |-> " + std::uint::str(next);
    alt c.node {
      ninit(id, i) { tbl.insert(local_def(id), cinit(next, c.span, i)); }
      npred(p, d_id, args) {
        alt tbl.find(d_id) {
          some(ct) {
            alt ct {
              cinit(_, _, _) {
                tcx.sess.bug("add_constraint: same def_id used" +
                                 " as a variable and a pred");
              }
              cpred(_, pds) {
                *pds += ~[respan(c.span, {args: args, bit_num: next})];
              }
            }
          }
          none. {
            let rslt: @mutable pred_args[] =
                @mutable ~[respan(c.span, {args: args, bit_num: next})];
            tbl.insert(d_id, cpred(p, rslt));
          }
        }
      }
    }
    ret next + 1u;
}


/* builds a table mapping each local var defined in f
   to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(ccx: &crate_ctxt, f: &_fn, tp: &ty_param[], f_sp: &span,
              f_name: &fn_ident, id: node_id) {
    let name = fn_ident_to_string(id, f_name);
    let res_map = @new_def_hash[constraint]();
    let next: uint = 0u;

    let cx: ctxt = find_locals(ccx.tcx, f, tp, f_sp, f_name, id);
    /* now we have to add bit nums for both the constraints
       and the variables... */

    for c: sp_constr  in { *cx.cs } {
        next = add_constraint(cx.tcx, c, next, res_map);
    }
    /* if this function has any constraints, instantiate them to the
       argument names and add them */
    let sc;
    for c: @constr  in f.decl.constraints {
        sc = ast_constr_to_sp_constr(cx.tcx, f.decl.inputs, c);
        next = add_constraint(cx.tcx, sc, next, res_map);
    }

    /* add a pseudo-entry for the function's return value
       we can safely use the function's name itself for this purpose */

    add_constraint(cx.tcx, respan(f_sp, ninit(id, name)), next, res_map);
    let v: @mutable node_id[] = @mutable ~[];
    let rslt =
        {constrs: res_map,
         num_constraints:
             ivec::len(*cx.cs) + ivec::len(f.decl.constraints) + 1u,
         cf: f.decl.cf,
         used_vars: v};
    ccx.fm.insert(id, rslt);
    log name + " has " + std::uint::str(num_constraints(rslt)) +
            " constraints";
}


/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(ccx: &crate_ctxt, c: @crate) {
    let visitor =
        visit::mk_simple_visitor(@{visit_fn:
                                       bind mk_fn_info(ccx, _, _, _, _, _)
                                      with *visit::default_simple_visitor()});
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
