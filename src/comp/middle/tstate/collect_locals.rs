import std::istr;
import std::uint;
import std::int;
import std::vec;
import syntax::ast::*;
import syntax::ast_util::*;
import util::ppaux::fn_ident_to_string;
import std::option::*;
import syntax::visit;
import aux::*;
import std::map::new_int_hash;
import util::common::new_def_hash;
import syntax::codemap::span;
import syntax::ast_util::respan;

type ctxt = {cs: @mutable [sp_constr], tcx: ty::ctxt};

fn collect_local(loc: &@local, cx: &ctxt, v: &visit::vt<ctxt>) {
    for each p: @pat in pat_bindings(loc.node.pat) {
        let ident = alt p.node { pat_bind(id) { id } };
        log ~"collect_local: pushing " + ident;;
        *cx.cs += [respan(loc.span, ninit(p.id, ident))];
    }
    visit::visit_local(loc, cx, v);
}

fn collect_pred(e: &@expr, cx: &ctxt, v: &visit::vt<ctxt>) {
    alt e.node {
      expr_check(_, ch) { *cx.cs += [expr_to_constr(cx.tcx, ch)]; }
      expr_if_check(ex, _, _) { *cx.cs += [expr_to_constr(cx.tcx, ex)]; }


      // If it's a call, generate appropriate instances of the
      // call's constraints.
      expr_call(operator, operands) {
        for c: @ty::constr in constraints_expr(cx.tcx, operator) {
            let ct: sp_constr =
                respan(c.span,
                       aux::substitute_constr_args(cx.tcx, operands, c));
            *cx.cs += [ct];
        }
      }
      _ { }
    }
    // visit subexpressions
    visit::visit_expr(e, cx, v);
}

fn find_locals(tcx: &ty::ctxt, f: &_fn, tps: &[ty_param], sp: &span,
               i: &fn_ident, id: node_id) -> ctxt {
    let cx: ctxt = {cs: @mutable [], tcx: tcx};
    let visitor = visit::default_visitor::<ctxt>();

    visitor =
        @{visit_local: collect_local,
          visit_expr: collect_pred,
          visit_fn: do_nothing with *visitor};
    visit::visit_fn(f, tps, sp, i, id, cx, visit::mk_vt(visitor));
    ret cx;
}

fn add_constraint(tcx: &ty::ctxt, c: sp_constr, next: uint, tbl: constr_map)
   -> uint {
    log constraint_to_str(tcx, c) + ~" |-> "
        + std::uint::str(next);
    alt c.node {
      ninit(id, i) { tbl.insert(local_def(id), cinit(next, c.span, i)); }
      npred(p, d_id, args) {
        alt tbl.find(d_id) {
          some(ct) {
            alt ct {
              cinit(_, _, _) {
                tcx.sess.bug(~"add_constraint: same def_id used" +
                                 ~" as a variable and a pred");
              }
              cpred(_, pds) {
                *pds += [respan(c.span, {args: args, bit_num: next})];
              }
            }
          }
          none. {
            let rslt: @mutable [pred_args] =
                @mutable [respan(c.span, {args: args, bit_num: next})];
            tbl.insert(d_id, cpred(p, rslt));
          }
        }
      }
    }
    ret next + 1u;
}


/* builds a table mapping each local var defined in f
   to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(ccx: &crate_ctxt, f: &_fn, tp: &[ty_param], f_sp: &span,
              f_name: &fn_ident, id: node_id) {
    let name = fn_ident_to_string(id, f_name);
    let res_map = @new_def_hash::<constraint>();
    let next: uint = 0u;

    let cx: ctxt = find_locals(ccx.tcx, f, tp, f_sp, f_name, id);
    /* now we have to add bit nums for both the constraints
       and the variables... */

    for c: sp_constr in { *cx.cs } {
        next = add_constraint(cx.tcx, c, next, res_map);
    }
    /* if this function has any constraints, instantiate them to the
       argument names and add them */
    let sc;
    for c: @constr in f.decl.constraints {
        sc = ast_constr_to_sp_constr(cx.tcx, f.decl.inputs, c);
        next = add_constraint(cx.tcx, sc, next, res_map);
    }

    /* Need to add constraints for args too, b/c they
    can be deinitialized */
    for a: arg in f.decl.inputs {
        next =
            add_constraint(cx.tcx, respan(f_sp, ninit(a.id, a.ident)), next,
                           res_map);
    }

    /* add the special i_diverge and i_return constraints
    (see the type definition for auxiliary::fn_info for an explanation) */

    // use the name of the function for the "return" constraint
    next =
        add_constraint(cx.tcx, respan(f_sp, ninit(id, name)), next, res_map);
    // and the name of the function, with a '!' appended to it, for the
    // "diverges" constraint
    let diverges_id = ccx.tcx.sess.next_node_id();
    let diverges_name = name + ~"!";
    add_constraint(cx.tcx, respan(f_sp, ninit(diverges_id, diverges_name)),
                   next, res_map);

    let v: @mutable [node_id] = @mutable [];
    let rslt =
        {constrs: res_map,

         // add 2 to account for the i_return and i_diverge constraints
         num_constraints:
             vec::len(*cx.cs) + vec::len(f.decl.constraints) +
                 vec::len(f.decl.inputs) + 2u,
         cf: f.decl.cf,
         i_return: ninit(id, name),
         i_diverge: ninit(diverges_id, diverges_name),
         used_vars: v};
    ccx.fm.insert(id, rslt);
    log istr::to_estr(name + ~" has "
                      + std::uint::str(num_constraints(rslt))
                      + ~" constraints");
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
