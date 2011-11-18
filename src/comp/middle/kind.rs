import std::option::{some, none};
import syntax::{visit, ast_util};
import syntax::ast::*;
import syntax::codemap::span;

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
            mutable ret_by_ref: bool};

fn check_crate(tcx: ty::ctxt, crate: @crate) -> rval_map {
    let ctx = {tcx: tcx,
               rval_map: std::map::new_int_hash(),
               mutable ret_by_ref: false};
    let visit = visit::mk_vt(@{
        visit_expr: check_expr,
        visit_stmt: check_stmt,
        visit_fn: visit_fn
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, ctx, visit);
    tcx.sess.abort_if_errors();
    ret ctx.rval_map;
}

fn check_expr(e: @expr, cx: ctx, v: visit::vt<ctx>) {
    alt e.node {
      expr_assign(_, ex) | expr_assign_op(_, _, ex) |
      expr_block({node: {expr: some(ex), _}, _}) |
      expr_unary(box(_), ex) | expr_unary(uniq(_), ex) { maybe_copy(cx, ex); }
      expr_ret(some(ex)) { if !cx.ret_by_ref { maybe_copy(cx, ex); } }
      expr_copy(expr) { check_copy_ex(cx, expr, false); }
      // Vector add copies.
      expr_binary(add., ls, rs) { maybe_copy(cx, ls); maybe_copy(cx, rs); }
      expr_rec(fields, _) {
        for field in fields { maybe_copy(cx, field.node.expr); }
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
            let kinds = ty::lookup_item_type(cx.tcx, did).kinds, i = 0u;
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
      expr_fn({proto: proto_shared(_), _}) {
        for free in *freevars::get_freevars(cx.tcx, e.id) {
            let id = ast_util::def_id_of_def(free).node;
            let ty = ty::node_id_to_type(cx.tcx, id);
            check_copy(cx, ty, e.span);
        }
      }
      expr_ternary(_, a, b) { maybe_copy(cx, a); maybe_copy(cx, b); }
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

fn visit_fn(f: _fn, tps: [ty_param], sp: span, ident: fn_ident,
            id: node_id, cx: ctx, v: visit::vt<ctx>) {
    let old_ret = cx.ret_by_ref;
    cx.ret_by_ref = ast_util::ret_by_ref(f.decl.cf);
    visit::visit_fn(f, tps, sp, ident, id, cx, v);
    cx.ret_by_ref = old_ret;
}

fn maybe_copy(cx: ctx, ex: @expr) {
    check_copy_ex(cx, ex, true);
}

fn check_copy_ex(cx: ctx, ex: @expr, _warn: bool) {
    if ty::expr_is_lval(cx.tcx, ex) {
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
    if ty::type_kind(cx.tcx, ty) == kind_noncopyable {
        cx.tcx.sess.span_err(sp, "copying a noncopyable value");
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
