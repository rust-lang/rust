import option::{some, none};
import syntax::{visit, ast_util};
import syntax::ast::*;
import syntax::codemap::span;
import ty::{kind, kind_copyable, kind_sendable, kind_noncopyable};

// Kind analysis pass. There are three kinds:
//
//  sendable: scalar types, and unique types containing only sendable types
//  copyable: boxes, objects, closures, and uniques containing copyable types
//  noncopyable: resources, or unique types containing resources
//
// This pass ensures that type parameters are only instantiated with types
// whose kinds are equal or less general than the way the type parameter was
// annotated (with the `send` or `copy` keyword).
//
// It also verifies that noncopyable kinds are not copied. Sendability is not
// applied, since none of our language primitives send. Instead, the sending
// primitives in the stdlib are explicitly annotated to only take sendable
// types.

fn kind_to_str(k: kind) -> str {
    alt k {
      kind_sendable. { "sendable" }
      kind_copyable. { "copyable" }
      kind_noncopyable. { "noncopyable" }
    }
}

type rval_map = std::map::hashmap<node_id, ()>;

type ctx = {tcx: ty::ctxt,
            rval_map: rval_map,
            method_map: typeck::method_map,
            last_uses: last_use::last_uses};

fn check_crate(tcx: ty::ctxt, method_map: typeck::method_map,
               last_uses: last_use::last_uses, crate: @crate)
    -> rval_map {
    let ctx = {tcx: tcx,
               rval_map: std::map::new_int_hash(),
               method_map: method_map,
               last_uses: last_uses};
    let visit = visit::mk_vt(@{
        visit_expr: check_expr,
        visit_stmt: check_stmt,
        visit_fn: check_fn
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, ctx, visit);
    tcx.sess.abort_if_errors();
    ret ctx.rval_map;
}

// Yields the appropriate function to check the kind of closed over
// variables. `id` is the node_id for some expression that creates the
// closure.
fn with_closure_check_fn(cx: ctx, id: node_id,
                         b: block(fn(ctx, ty::t, sp: span))) {
    let fty = ty::node_id_to_monotype(cx.tcx, id);
    alt ty::ty_fn_proto(cx.tcx, fty) {
      proto_send. { b(check_send); }
      proto_shared(_) { b(check_copy); }
      proto_block. | proto_bare. { /* no check needed */ }
    }
}

// Check that the free variables used in a shared/sendable closure conform
// to the copy/move kind bounds. Then recursively check the function body.
fn check_fn(fk: visit::fn_kind, decl: fn_decl, body: blk, sp: span,
            id: node_id, cx: ctx, v: visit::vt<ctx>) {

    // n.b.: This could be the body of either a fn decl or a fn expr.  In the
    // former case, the prototype will be proto_bare and no check occurs.  In
    // the latter case, we do not check the variables that in the capture
    // clause (as we don't have access to that here) but just those that
    // appear free.  The capture clauses are checked below, in check_expr().
    //
    // We could do this check also in check_expr(), but it seems more
    // "future-proof" to do it this way, as check_fn_body() is supposed to be
    // the common flow point for all functions that appear in the AST.

    with_closure_check_fn(cx, id) { |check_fn|
        for @{def, span} in *freevars::get_freevars(cx.tcx, id) {
            let id = ast_util::def_id_of_def(def).node;
            let ty = ty::node_id_to_type(cx.tcx, id);
            check_fn(cx, ty, span);
        }
    }

    visit::visit_fn(fk, decl, body, sp, id, cx, v);
}

fn check_fn_cap_clause(cx: ctx,
                       id: node_id,
                       cap_clause: capture_clause) {
    // Check that the variables named in the clause which are not free vars
    // (if any) are also legal.  freevars are checked above in check_fn_body.
    // This is kind of a degenerate case, as captured variables will generally
    // appear free in the body.
    let freevars = freevars::get_freevars(cx.tcx, id);
    let freevar_ids = vec::map(*freevars, { |freevar|
        ast_util::def_id_of_def(freevar.def).node
    });
    //log("freevar_ids", freevar_ids);
    with_closure_check_fn(cx, id) { |check_fn|
        let check_var = lambda(&&cap_item: @capture_item) {
            let cap_def = cx.tcx.def_map.get(cap_item.id);
            let cap_def_id = ast_util::def_id_of_def(cap_def).node;
            if !vec::member(cap_def_id, freevar_ids) {
                let ty = ty::node_id_to_type(cx.tcx, cap_def_id);
                check_fn(cx, ty, cap_item.span);
            }
        };
        vec::iter(cap_clause.copies, check_var);
        vec::iter(cap_clause.moves, check_var);
    }
}

fn check_expr(e: @expr, cx: ctx, v: visit::vt<ctx>) {

    alt e.node {
      expr_assign(_, ex) | expr_assign_op(_, _, ex) |
      expr_block({node: {expr: some(ex), _}, _}) |
      expr_unary(box(_), ex) | expr_unary(uniq(_), ex) { maybe_copy(cx, ex); }
      expr_ret(some(ex)) { maybe_copy(cx, ex); }
      expr_copy(expr) { check_copy_ex(cx, expr, false); }
      // Vector add copies.
      expr_binary(add., ls, rs) { maybe_copy(cx, ls); maybe_copy(cx, rs); }
      expr_rec(fields, def) {
        for field in fields { maybe_copy(cx, field.node.expr); }
        alt def {
          some(ex) {
            // All noncopyable fields must be overridden
            let t = ty::expr_ty(cx.tcx, ex);
            let ty_fields = alt ty::struct(cx.tcx, t) { ty::ty_rec(f) { f } };
            for tf in ty_fields {
                if !vec::any(fields, {|f| f.node.ident == tf.ident}) &&
                    !ty::kind_can_be_copied(ty::type_kind(cx.tcx, tf.mt.ty)) {
                    cx.tcx.sess.span_err(ex.span,
                                         "copying a noncopyable value");
                }
            }
          }
          _ {}
        }
      }
      expr_tup(exprs) | expr_vec(exprs, _) {
        for expr in exprs { maybe_copy(cx, expr); }
      }
      expr_bind(_, args) {
        for a in args { alt a { some(ex) { maybe_copy(cx, ex); } _ {} } }
      }
      expr_call(f, args, _) {
        let i = 0u;
        for arg_t in ty::ty_fn_args(cx.tcx, ty::expr_ty(cx.tcx, f)) {
            alt arg_t.mode { by_copy. { maybe_copy(cx, args[i]); } _ {} }
            i += 1u;
        }
      }
      expr_path(_) {
        let substs = ty::node_id_to_ty_param_substs_opt_and_ty(cx.tcx, e.id);
        alt substs.substs {
          some(ts) {
            let did = ast_util::def_id_of_def(cx.tcx.def_map.get(e.id));
            let kinds = vec::map(ty::lookup_item_type(cx.tcx, did).bounds,
                                 {|bs| ty::param_bounds_to_kind(bs)});
            let i = 0u;
            for ty in ts {
                let kind = ty::type_kind(cx.tcx, ty);
                if !ty::kind_lteq(kinds[i], kind) {
                    cx.tcx.sess.span_err(e.span, "instantiating a " +
                                         kind_to_str(kinds[i]) +
                                         " type parameter with a "
                                         + kind_to_str(kind) + " type");
                }
                i += 1u;
            }
          }
          none. {}
        }
      }
      expr_ternary(_, a, b) { maybe_copy(cx, a); maybe_copy(cx, b); }
      expr_fn(_, _, _, cap_clause) {
        check_fn_cap_clause(cx, e.id, *cap_clause);
      }
      _ { }
    }
    visit::visit_expr(e, cx, v);
}

fn check_stmt(stmt: @stmt, cx: ctx, v: visit::vt<ctx>) {
    alt stmt.node {
      stmt_decl(@{node: decl_local(locals), _}, _) {
        for (_, local) in locals {
            alt local.node.init {
              some({op: init_assign., expr}) { maybe_copy(cx, expr); }
              _ {}
            }
        }
      }
      _ {}
    }
    visit::visit_stmt(stmt, cx, v);
}

fn maybe_copy(cx: ctx, ex: @expr) {
    check_copy_ex(cx, ex, true);
}

fn check_copy_ex(cx: ctx, ex: @expr, _warn: bool) {
    if ty::expr_is_lval(cx.method_map, cx.tcx, ex) &&
       !cx.last_uses.contains_key(ex.id) {
        let ty = ty::expr_ty(cx.tcx, ex);
        check_copy(cx, ty, ex.span);
        // FIXME turn this on again once vector types are no longer unique.
        // Right now, it is too annoying to be useful.
        /* if warn && ty::type_is_unique(cx.tcx, ty) {
            cx.tcx.sess.span_warn(ex.span, "copying a unique value");
        }*/
    }
}

fn check_copy(cx: ctx, ty: ty::t, sp: span) {
    if !ty::kind_can_be_copied(ty::type_kind(cx.tcx, ty)) {
        cx.tcx.sess.span_err(sp, "copying a noncopyable value");
    }
}

fn check_send(cx: ctx, ty: ty::t, sp: span) {
    if !ty::kind_can_be_sent(ty::type_kind(cx.tcx, ty)) {
        cx.tcx.sess.span_err(sp, "not a sendable value");
    }
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
