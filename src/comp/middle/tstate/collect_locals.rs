import std::vec;
import std::vec::plus_option;

import front::ast::*;
import option::*;

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

import util::common::new_def_hash;
import util::common::uistr;
import util::common::span;
import util::common::respan;

type ctxt = rec(@vec[constraint_info] cs,
                ty::ctxt tcx);

type constraint_info = rec(def_id id, aux::constr c);

fn collect_local(&ctxt cx, &@decl d) -> () {
    alt (d.node) {
      case (decl_local(?loc)) {
        log("collect_local: pushing " + loc.ident);
        vec::push[constraint_info](*cx.cs,
                                   rec(id=loc.id,
                                       c=respan(d.span,
                                                ninit(loc.ident))));
      }
      case (_) { ret; }
    }
}

fn exprs_to_constr_args(ty::ctxt tcx, vec[@expr] args) -> vec[@constr_arg] {
    fn one(ty::ctxt tcx, &@expr e) -> @constr_arg {
        alt (e.node) {
            case (expr_path(?p, _)) {
                if (vec::len(p.node.idents) == 1u) {
                    ret @respan(p.span, carg_ident(p.node.idents.(0)));
                }
                else {
                    tcx.sess.bug("exprs_to_constr_args: non-local variable "
                                 + "as pred arg");
                }
            }
            case (expr_lit(?l, _)) {
                ret @respan(e.span, carg_lit(l));
            }
            case (_) {
                tcx.sess.bug("exprs_to_constr_args: ill-formed pred arg");
            }
        }
    }
    auto f = bind one(tcx, _);
    ret vec::map(f, args); 
}

fn def_id_for_constr(ty::ctxt tcx, uint t) -> def_id {
    alt (tcx.def_map.find(t)) {
        case (none) {
            tcx.sess.bug("def_id_for_constr: bad node_id " + uistr(t));
         }
        case (some(def_fn(?i))) {
            ret i;
        }
        case (_) {
            tcx.sess.bug("def_id_for_constr: pred is not a function");
        }
    }
}

fn expr_to_constr(ty::ctxt tcx, &@expr e) -> constraint_info {
    alt (e.node) {
        // change the first pattern to expr_path to test a typechecker bug
        case (expr_call(?operator, ?args, _)) {
            alt (operator.node) {
                case (expr_path(?p, ?a)) {
                    ret rec(id=def_id_for_constr(tcx, a.id),
                            c=respan(e.span,
                                npred(p, exprs_to_constr_args(tcx, args)))); 
                }
                case (_) {
                    tcx.sess.span_err(operator.span, "Internal error: " +
                       " ill-formed operator in predicate");
                }
            }
        }
        case (_) {
            tcx.sess.span_err(e.span, "Internal error: " +
                              " ill-formed predicate");
        }
    }
}

fn collect_pred(&ctxt cx, &@expr e) -> () {
    alt (e.node) {
        case (expr_check(?e, _)) {
            vec::push[constraint_info](*cx.cs, expr_to_constr(cx.tcx, e));
        }
        case (_) { }
    }
}

fn find_locals(&ty::ctxt tcx, &_fn f, &span sp, &ident i, &def_id d, &ann a)
    -> ctxt {
    let ctxt cx = rec(cs=@vec::alloc[constraint_info](0u), tcx=tcx);
    auto visitor = walk::default_visitor();
    visitor = rec(visit_decl_pre=bind collect_local(cx,_),
                  visit_expr_pre=bind collect_pred(cx,_)
                  with visitor);
    walk_fn(visitor, f, sp, i, d, a);
    ret cx;
}

fn add_constraint(&ty::ctxt tcx, constraint_info c, uint next, constr_map tbl)
    -> uint {
    log(aux::constraint_to_str(tcx, c.c) + " |-> "
        + util::common::uistr(next));
    let aux::constr cn = c.c;
    alt (cn.node) {
        case (ninit(?i)) {
            tbl.insert(c.id, cinit(next, cn.span, i));
        }
        case (npred(?p, ?args)) {
            alt (tbl.find(c.id)) {
                case (some[constraint](?ct)) {
                    alt (ct) {
                        case (cinit(_,_,_)) {
                            tcx.sess.bug("add_constraint: same def_id used"
                                         + " as a variable and a pred");
                        }
                        case (cpred(_, ?pds)) {
                            vec::push(pds, respan(cn.span,
                              rec(args=args, bit_num=next)));
                        }
                    }
                }
                case (none[constraint]) {
                    tbl.insert(c.id, cpred(p,
                      [respan(cn.span, rec(args=args, bit_num=next))]));
                }
            }
        }
    }
    ret (next + 1u);
}

/* builds a table mapping each local var defined in f
   to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(&crate_ctxt ccx, &_fn f, &span f_sp,
              &ident f_name, &def_id f_id, &ann a)
    -> () {
    auto res_map = @new_def_hash[constraint]();
    let uint next = 0u;
    let vec[arg] f_args = f.decl.inputs;

    /* ignore args, which we know are initialized;
       just collect locally declared vars */

    let ctxt cx = find_locals(ccx.tcx, f, f_sp, f_name, f_id, a);
    /* now we have to add bit nums for both the constraints
       and the variables... */

    for (constraint_info c in *cx.cs) {
        next = add_constraint(cx.tcx, c, next, res_map);
    }
    /* add a pseudo-entry for the function's return value
       we can safely use the function's name itself for this purpose */
    add_constraint(cx.tcx, rec(id=f_id,
                               c=respan(f_sp, ninit(f_name))), next, res_map);
    
    auto res = rec(constrs=res_map,
                            num_constraints=vec::len(*cx.cs) + 1u,
                   cf=f.decl.cf);

    ccx.fm.insert(f_id, res);
    
    log(f_name + " has " + uistr(num_constraints(res)) + " constraints");

}

/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(&crate_ctxt ccx, @crate c) -> () {
  let ast_visitor vars_visitor = walk::default_visitor();
  vars_visitor = rec(visit_fn_pre=bind mk_fn_info(ccx,_,_,_,_,_)
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

