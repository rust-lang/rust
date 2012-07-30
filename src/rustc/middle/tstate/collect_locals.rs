import option::*;
import pat_util::*;
import syntax::ast::*;
import syntax::ast_util::*;
import syntax::visit;
import syntax::codemap::span;
import syntax::ast_util::respan;
import driver::session::session;
import aux::*;
import std::map::hashmap;
import dvec::{dvec, extensions};

type ctxt = {cs: @mut ~[sp_constr], tcx: ty::ctxt};

fn collect_pred(e: @expr, cx: ctxt, v: visit::vt<ctxt>) {
    alt e.node {
      expr_check(_, ch) { vec::push(*cx.cs, expr_to_constr(cx.tcx, ch)); }
      expr_if_check(ex, _, _) {
        vec::push(*cx.cs, expr_to_constr(cx.tcx, ex));
      }

      // If it's a call, generate appropriate instances of the
      // call's constraints.
      expr_call(operator, operands, _) {
        for constraints_expr(cx.tcx, operator).each |c| {
            let ct: sp_constr =
                respan(c.span,
                       aux::substitute_constr_args(cx.tcx, operands, c));
            vec::push(*cx.cs, ct);
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
    let cx: ctxt = {cs: @mut ~[], tcx: tcx};
    let visitor = visit::default_visitor::<ctxt>();
    let visitor =
        @{visit_expr: collect_pred,
          visit_fn: do_nothing
          with *visitor};
    visit::visit_fn(fk, f_decl, f_body, sp,
                    id, cx, visit::mk_vt(visitor));
    ret cx;
}

fn add_constraint(tcx: ty::ctxt, c: sp_constr, next: uint, tbl: constr_map) ->
   uint {
    log(debug,
             constraint_to_str(tcx, c) + ~" |-> " + uint::str(next));

    let {path: p, def_id: d_id, args: args} = c.node;
    alt tbl.find(d_id) {
      some(ct) {
        (*ct.descs).push(respan(c.span, {args: args, bit_num: next}));
      }
      none {
        let rslt = @dvec();
        (*rslt).push(respan(c.span, {args: args, bit_num: next}));
        tbl.insert(d_id, {path:p, descs:rslt});
      }
    }
    ret next + 1u;
}

fn contains_constrained_calls(tcx: ty::ctxt, body: blk) -> bool {
    type cx = @{
        tcx: ty::ctxt,
        mut has: bool
    };
    let cx = @{
        tcx: tcx,
        mut has: false
    };
    let vtor = visit::default_visitor::<cx>();
    let vtor = @{visit_expr: visit_expr with *vtor};
    visit::visit_block(body, cx, visit::mk_vt(vtor));
    ret cx.has;

    fn visit_expr(e: @expr, &&cx: cx, v: visit::vt<cx>) {
        import syntax::print::pprust;
        debug!{"visiting %?", pprust::expr_to_str(e)};

        visit::visit_expr(e, cx, v);

        if constraints_expr(cx.tcx, e).is_not_empty() {
            debug!{"has constraints"};
            cx.has = true;
        } else {
            debug!{"has not constraints"};
        }
    }
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
    let mut next: uint = 0u;

    let cx: ctxt = find_locals(ccx.tcx, fk, f_decl, f_body, f_sp, id);
    /* now we have to add bit nums for both the constraints
       and the variables... */

    let ignore = !contains_constrained_calls(ccx.tcx, f_body);

    if !ignore {
        let mut i = 0u, l = vec::len(*cx.cs);
        while i < l {
            next = add_constraint(cx.tcx, copy cx.cs[i], next, res_map);
            i += 1u;
        }
        /* if this function has any constraints, instantiate them to the
        argument names and add them */
        for f_decl.constraints.each |c| {
            let sc = ast_constr_to_sp_constr(cx.tcx, f_decl.inputs, c);
            next = add_constraint(cx.tcx, sc, next, res_map);
        }
    }

    let v: @mut ~[node_id] = @mut ~[];
    let rslt =
        {constrs: res_map,
         num_constraints: next,
         cf: f_decl.cf,
         used_vars: v,
         ignore: ignore};
    ccx.fm.insert(id, rslt);
    debug!{"%s has %u constraints", *name, num_constraints(rslt)};
}


/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(ccx: crate_ctxt, c: @crate) {
    let visitor =
        visit::mk_simple_visitor(@{
            visit_fn: |a,b,c,d,e| mk_fn_info(ccx, a, b, c, d, e)
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
