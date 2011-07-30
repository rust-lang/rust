/*
* Kinds are types of type.
*
* Every type has a kind. Every type parameter has a set of kind-capabilities
* saying which kind of type may be passed as the parameter.
*
* The kinds are based on two capabilities: move and send. These may each be
* present or absent, though only three of the four combinations can actually
* occur:
*
*
*
*    MOVE +   SEND  =  "Unique": no shared substructures or pins, only
*                                interiors and ~ boxes.
*
*    MOVE + NOSEND  =  "Shared": structures containing @, fixed to the local
*                                task heap/pool; or ~ structures pointing to
*                                pinned values.
*
*  NOMOVE + NOSEND  =  "Pinned": structures directly containing resources, or
*                                by-alias closures as interior or
*                                uniquely-boxed members.
*
*  NOMOVE +   SEND  =  --      : no types are like this.
*
*
* Since this forms a lattice, we denote the capabilites in terms of a
* worst-case requirement.  That is, if your function needs to move-and-send
* (or copy) your T, you write fn<~T>(...). If you need to move but not send,
* you write fn<@T>(...). And if you need neither -- can work with any sort of
* pinned data at all -- then you write fn<T>(...).
*
*
* Most types are unique or shared. Other possible name combinations for these
* two: (tree, graph; pruned, pooled; message, local; owned, common) are
* plausible but nothing stands out as completely pithy-and-obvious.
*
* Resources cannot be copied or sent; they're pinned. They can't be copied
* because it would interfere with destruction (multiple destruction?) They
* cannot be sent because we don't want to oblige the communication system to
* run destructors in some weird limbo context of messages-in-transit. It
* should always be ok to just free messages it's dropping.
*
* Note that obj~ and fn~ -- those that capture a unique environment -- can be
* sent, so satisfy ~T. So can plain obj and fn.
*
*
* Further notes on copying and moving; sending is accomplished by calling a
* move-in operator on something constrained to a unique type ~T.
*
*
* COPYING:
* --------
*
*   A copy is made any time you pass-by-value or execute the = operator in a
*   non-init expression.
*
*   @ copies shallow, is always legal
*   ~ copies deep, is only legal if pointee is unique.
*     pinned values (pinned resources, alias-closures) can't be copied
*     all other unique (eg. interior) values copy shallow
*
*   Note this means that only type parameters constrained to ~T can be copied.
*
* MOVING:
* -------
*
*  A move is made any time you pass-by-move (that is, with 'move' mode) or
*  execute the <- operator.
*
*/


import syntax::ast;
import syntax::visit;

import std::ivec;

import ast::kind;
import ast::kind_unique;
import ast::kind_shared;
import ast::kind_pinned;

fn kind_lteq(a: kind, b: kind) -> bool {
    alt a {
      kind_pinned. { true }
      kind_shared. { b != kind_pinned }
      kind_unique. { b == kind_unique }
    }
}

fn lower_kind(a: kind, b: kind) -> kind {
    if kind_lteq(a, b) { a } else { b }
}

fn kind_to_str(k: kind) -> str {
    alt k {
      ast::kind_pinned. { "pinned" }
      ast::kind_unique. { "unique" }
      ast::kind_shared. { "shared" }
    }
}

fn type_and_kind(tcx: &ty::ctxt, e: &@ast::expr)
    -> {ty: ty::t, kind: ast::kind} {
    let t = ty::expr_ty(tcx, e);
    let k = ty::type_kind(tcx, t);
    {ty: t, kind: k}
}

fn need_expr_kind(tcx: &ty::ctxt, e: &@ast::expr,
                  k_need: ast::kind, descr: &str) {
    let tk = type_and_kind(tcx, e);
    log #fmt("for %s: want %s type, got %s type %s",
             descr,
             kind_to_str(k_need),
             kind_to_str(tk.kind),
             util::ppaux::ty_to_str(tcx, tk.ty));

    if ! kind_lteq(k_need, tk.kind) {
        let s =
            #fmt("mismatched kinds for %s: needed %s type, got %s type %s",
                 descr,
                 kind_to_str(k_need),
                 kind_to_str(tk.kind),
                 util::ppaux::ty_to_str(tcx, tk.ty));
        tcx.sess.span_err(e.span, s);
    }
}

fn need_shared_lhs_rhs(tcx: &ty::ctxt,
                       a: &@ast::expr, b: &@ast::expr,
                       op: &str) {
    need_expr_kind(tcx, a, ast::kind_shared, op + " lhs");
    need_expr_kind(tcx, b, ast::kind_shared, op + " rhs");
}

fn check_expr(tcx: &ty::ctxt, e: &@ast::expr) {
    alt e.node {
      ast::expr_move(a, b) { need_shared_lhs_rhs(tcx, a, b, "<-"); }
      ast::expr_assign(a, b) { need_shared_lhs_rhs(tcx, a, b, "="); }
      ast::expr_swap(a, b) { need_shared_lhs_rhs(tcx, a, b, "<->"); }
      ast::expr_call(callee, _) {
        let tpt = ty::expr_ty_params_and_ty(tcx, callee);
        // If we have typarams, we're calling an item; we need to check
        // that all the types we're supplying as typarams conform to the
        // typaram kind constraints on that item.
        if ivec::len(tpt.params) != 0u {
            let callee_def = ast::def_id_of_def(tcx.def_map.get(callee.id));
            let item_tk = ty::lookup_item_type(tcx, callee_def);
            let i = 0;
            assert ivec::len(item_tk.kinds) == ivec::len(tpt.params);
            for k_need: ast::kind in item_tk.kinds {
                let t = tpt.params.(i);
                let k = ty::type_kind(tcx, t);
                if ! kind_lteq(k_need, k) {
                    let s = #fmt("mismatched kinds for typaram %d: \
                                  needed %s type, got %s type %s",
                                 i,
                                 kind_to_str(k_need),
                                 kind_to_str(k),
                                 util::ppaux::ty_to_str(tcx, t));
                    tcx.sess.span_err(e.span, s);
                }
                i += 1;
            }
        }
      }
      _ { }
    }
}

fn check_crate(tcx: &ty::ctxt, crate: &@ast::crate) {
    let visit = visit::mk_simple_visitor
        (@{visit_expr: bind check_expr(tcx, _)
           with *visit::default_simple_visitor()});
    visit::visit_crate(*crate, (), visit);
    tcx.sess.abort_if_errors();
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
