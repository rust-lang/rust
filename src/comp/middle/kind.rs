/*
* Kinds are types of type.
*
* Every type has a kind. Every type parameter has a set of kind-capabilities
* saying which kind of type may be passed as the parameter.
*
* The kinds are based on two capabilities: copy and send. These may each be
* present or absent, though only three of the four combinations can actually
* occur:
*
*
*
*    COPY +   SEND  =  "Unique": no shared substructures or pins, only
*                                interiors and ~ boxes.
*
*    COPY + NOSEND  =  "Shared": structures containing @, fixed to the local
*                                task heap/pool.
*
*  NOCOPY + NOSEND  =  "Pinned": structures containing resources or
*                                by-alias closures as interior or
*                                uniquely-boxed members.
*
*  NOCOPY +   SEND  =  --      : no types are like this.
*
*
* Since this forms a lattice, we denote the capabilites in terms of a
* worst-case requirement.  That is, if your function needs to copy-and-send
* your T, you write fn<~T>(...). If you need to copy but not send, you write
* fn<@T>(...). And if you need neither -- can work with any sort of pinned
* data at all -- then you write fn<T>(...).
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
*   ~ copies deep
*   @ copies shallow
*     pinned values (pinned resources, alias-closures) can't be copied
*     all other interiors copy shallow
*
* MOVING:
* -------
*
*  A move is made any time you pass-by-move (that is, with 'move' mode) or
*  execute the <- operator.
*
*  Anything you can copy, you can move. Move is (semantically) just
*  shallow-copy + deinit.  Note that: ~ moves shallow even though it copies
*  deep. Move is the operator that lets ~ copy shallow: by pairing it with a
*  deinit.
*
*/


import syntax::ast;
import syntax::walk;

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

fn check_expr(tcx: &ty::ctxt, e: &@ast::expr) {
    let t = ty::expr_ty(tcx, e);
    let k = ty::type_kind(tcx, t);
    log #fmt("%s type: %s", kind_to_str(k),
             util::ppaux::ty_to_str(tcx, t));
}

fn check_crate(tcx: &ty::ctxt, crate: &@ast::crate) {
    let visit =
        {visit_expr_pre: bind check_expr(tcx, _)
         with walk::default_visitor()};
    walk::walk_crate(visit, *crate);
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
