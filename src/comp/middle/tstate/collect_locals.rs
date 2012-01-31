import option::*;
import pat_util::*;
import syntax::ast::*;
import syntax::ast_util::*;
import syntax::visit;
import util::common::new_def_hash;
import syntax::codemap::span;
import syntax::ast_util::respan;
import driver::session::session;
import aux::*;

type ctxt = {cs: @mutable [sp_constr], tcx: ty::ctxt};

fn collect_local(loc: @local, cx: ctxt, v: visit::vt<ctxt>) {
    pat_bindings(pat_util::normalize_pat(cx.tcx, loc.node.pat))
     {|p_id, _s, id|
       *cx.cs += [respan(loc.span, ninit(p_id, path_to_ident(id)))];
    };
    visit::visit_local(loc, cx, v);
}

fn collect_pred(e: @expr, cx: ctxt, v: visit::vt<ctxt>) {
    alt e.node {
      expr_check(_, ch) { *cx.cs += [expr_to_constr(cx.tcx, ch)]; }
      expr_if_check(ex, _, _) { *cx.cs += [expr_to_constr(cx.tcx, ex)]; }

      // If it's a call, generate appropriate instances of the
      // call's constraints.
      expr_call(operator, operands, _) {
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

fn find_locals(tcx: ty::ctxt,
               fk: visit::fn_kind,
               f_decl: fn_decl,
               f_body: blk,
               sp: span,
               id: node_id) -> ctxt {
    let cx: ctxt = {cs: @mutable [], tcx: tcx};
    let visitor = visit::default_visitor::<ctxt>();

    visitor =
        @{visit_local: collect_local,
          visit_expr: collect_pred,
          visit_fn: bind do_nothing(_, _, _, _, _, _, _)
          with *visitor};
    visit::visit_fn(fk, f_decl, f_body, sp,
                    id, cx, visit::mk_vt(visitor));
    ret cx;
}

fn add_constraint(tcx: ty::ctxt, c: sp_constr, next: uint, tbl: constr_map) ->
   uint {
    log(debug,
             constraint_to_str(tcx, c) + " |-> " + uint::str(next));
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
                *pds += [respan(c.span, {args: args, bit_num: next})];
              }
            }
          }
          none {
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
fn mk_fn_info(ccx: crate_ctxt,
              fk: visit::fn_kind,
              f_decl: fn_decl,
              f_body: blk,
              f_sp: span,
              id: node_id) {
    let name = visit::name_of_fn(fk);
    let res_map = new_def_hash::<constraint>();
    let next: uint = 0u;

    let cx: ctxt = find_locals(ccx.tcx, fk, f_decl, f_body, f_sp, id);
    /* now we have to add bit nums for both the constraints
       and the variables... */

    for c: sp_constr in { *cx.cs } {
        next = add_constraint(cx.tcx, c, next, res_map);
    }
    /* if this function has any constraints, instantiate them to the
       argument names and add them */
    let sc;
    for c: @constr in f_decl.constraints {
        sc = ast_constr_to_sp_constr(cx.tcx, f_decl.inputs, c);
        next = add_constraint(cx.tcx, sc, next, res_map);
    }

    /* Need to add constraints for args too, b/c they
    can be deinitialized */
    for a: arg in f_decl.inputs {
        next = add_constraint(
            cx.tcx,
            respan(f_sp, ninit(a.id, a.ident)),
            next,
            res_map);
    }

    /* add the special i_diverge and i_return constraints
    (see the type definition for auxiliary::fn_info for an explanation) */

    // use the function name for the "returns" constraint"
    let returns_id = ccx.tcx.sess.next_node_id();
    let returns_constr = ninit(returns_id, name);
    next =
        add_constraint(cx.tcx, respan(f_sp, returns_constr), next, res_map);
    // and the name of the function, with a '!' appended to it, for the
    // "diverges" constraint
    let diverges_id = ccx.tcx.sess.next_node_id();
    let diverges_constr = ninit(diverges_id, name + "!");
    next = add_constraint(cx.tcx, respan(f_sp, diverges_constr), next,
                          res_map);

    let v: @mutable [node_id] = @mutable [];
    let rslt =
        {constrs: res_map,
         num_constraints: next,
         cf: f_decl.cf,
         i_return: returns_constr,
         i_diverge: diverges_constr,
         used_vars: v};
    ccx.fm.insert(id, rslt);
    #debug("%s has %u constraints", name, num_constraints(rslt));
}


/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(ccx: crate_ctxt, c: @crate) {
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
// End:
//
