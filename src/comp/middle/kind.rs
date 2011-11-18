import std::option::some;
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
      // FIXME check for by-copy args
      expr_call(_f, _args, _) {

      }
      // FIXME: generic instantiation
      expr_path(_) {}
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
* worst-case requirement. That is, if your function needs to move-and-send (or
* copy) your T, you write fn<uniq T>(...). If you need to move but not send,
* you write fn<T>(...). And if you need neither -- can work with any sort of
* pinned data at all -- then you write fn<pin T>(...).
*
* Most types are unique or shared. Other possible name combinations for these
* two: (tree, graph; pruned, pooled; message, local; owned, common) are
* plausible but nothing stands out as completely pithy-and-obvious.
*
* Pinned values arise in 2 contexts: resources and &-closures (blocks). The
* latter absolutely must not be moved, since they could escape to the heap;
* the former must not be copied, since they'd then be multiply-destructed.
* We achieve the no-copy restriction by recycling the no-move restriction
* in place on pinned kinds for &-closures; and as a benefit we can guarantee
* that a resource passed by reference to C will never move during its life,
* occasionally useful for FFI-code.
*
* Resources cannot be sent because we don't want to oblige the communication
* system to run destructors in some weird limbo context of
* messages-in-transit. It should always be ok to just free messages it's
* dropping. Even if you wanted to send them, you'd need a new sigil for the
* NOMOVE + SEND combination, and you couldn't use the move-mode library
* interface to chan.send in that case (NOMOVE after all), so the whole thing
* wouldn't really work as minimally as the encoding we have here.
*
* Note that obj~ and fn~ -- those that capture a unique environment -- can be
* sent, so satisfy ~T. So can plain obj and fn. They can all also be copied.
*
* Further notes on copying and moving; sending is accomplished by calling a
* move-in operator on something constrained to a unique type ~T.
*
*
* COPYING:
* --------
*
*   A copy is made any time you pass-by-value or execute the = operator in a
*   non-init expression. Copying requires discriminating on type constructor.
*
*   @-boxes copy shallow, copying is always legal.
*
*   ~-boxes copy deep, copying is only legal if pointee is unique-kind.
*
*     Pinned-kind values (resources, &-closures) can't be copied. All other
*     unique-kind (eg. interior) values can be copied, and copy shallow.
*
*   Note: If you have no type constructor -- only an opaque typaram -- then
*   you can only copy if the typaram is constrained to ~T; this is because @T
*   might be a "~resource" box, and making a copy would cause a deep
*   resource-copy.
*
*
* MOVING:
* -------
*
*  A move is made any time you pass-by-move (that is, with move mode '-') or
*  execute the move ('<-') or swap ('<->') operators.
*
*/

/*
fn type_and_kind(tcx: ty::ctxt, e: @ast::expr) ->
   {ty: ty::t, kind: ast::kind} {
    let t = ty::expr_ty(tcx, e);
    let k = ty::type_kind(tcx, t);
    {ty: t, kind: k}
}

fn need_expr_kind(tcx: ty::ctxt, e: @ast::expr, k_need: ast::kind,
                  descr: str) {
    let tk = type_and_kind(tcx, e);
    log #fmt["for %s: want %s type, got %s type %s", descr,
             kind_to_str(k_need), kind_to_str(tk.kind),
             util::ppaux::ty_to_str(tcx, tk.ty)];

    demand_kind(tcx, e.span, tk.ty, k_need, descr);
}

fn demand_kind(tcx: ty::ctxt, sp: codemap::span, t: ty::t,
               k_need: ast::kind, descr: str) {
    let k = ty::type_kind(tcx, t);
    if !kind_lteq(k_need, k) {
        let s =
            #fmt["mismatched kinds for %s: needed %s type, got %s type %s",
                 descr, kind_to_str(k_need), kind_to_str(k),
                 util::ppaux::ty_to_str(tcx, t)];
        tcx.sess.span_err(sp, s);
    }
}

fn need_shared_lhs_rhs(tcx: ty::ctxt, a: @ast::expr, b: @ast::expr, op: str) {
    need_expr_kind(tcx, a, ast::kind_copyable, op + " lhs");
    need_expr_kind(tcx, b, ast::kind_copyable, op + " rhs");
}

/*
This ... is a hack (I find myself writing that too often *sadface*).

We need to be able to put pinned kinds into other types but such operations
are conceptually copies, and pinned kinds can't do that, e.g.

let a = my_resource(x);
let b = @a; // no-go

So this function attempts to make a loophole where resources can be put into
other types as long as it's done in a safe way, specifically like

let b = @my_resource(x);
*/
fn need_shared_or_pinned_ctor(tcx: ty::ctxt, a: @ast::expr, descr: str) {
    let tk = type_and_kind(tcx, a);
    if tk.kind == ast::kind_pinned && !pinned_ctor(a) {
        let err =
            #fmt["mismatched kinds for %s: cannot copy pinned type %s",
                 descr, util::ppaux::ty_to_str(tcx, tk.ty)];
        tcx.sess.span_err(a.span, err);
        let note =
            #fmt["try constructing %s directly into %s",
                 util::ppaux::ty_to_str(tcx, tk.ty), descr];
        tcx.sess.span_note(a.span, note);
    } else if tk.kind != ast::kind_pinned {
        need_expr_kind(tcx, a, ast::kind_shared, descr);
    }

    fn pinned_ctor(a: @ast::expr) -> bool {
        // FIXME: Technically a lambda block is also a pinned ctor
        alt a.node {
          ast::expr_call(cexpr, _, _) {
            // Assuming that if it's a call that it's safe to move in, mostly
            // because I don't know offhand how to ensure that it's a call
            // specifically to a resource constructor
            true
          }
          ast::expr_rec(_, _) {
            true
          }
          ast::expr_unary(ast::uniq(_), _) {
            true
          }
          ast::expr_tup(_) {
            true
          }
          ast::expr_vec(exprs, _) {
            true
          }
          _ { false }
        }
    }
}

fn check_expr(tcx: ty::ctxt, e: @ast::expr) {
    alt e.node {

      // FIXME: These rules do not fully implement the copy type-constructor
      // discrimination described by the block comment at the top of this
      // file. This code is wrong; it lets you copy anything shared-kind.

      ast::expr_move(a, b) { need_shared_lhs_rhs(tcx, a, b, "<-"); }
      ast::expr_assign(a, b) {
        need_shared_lhs_rhs(tcx, a, b, "=");
      }
      ast::expr_assign_op(_, a, b) {
        need_shared_lhs_rhs(tcx, a, b, "op=");
      }
      ast::expr_swap(a, b) { need_shared_lhs_rhs(tcx, a, b, "<->"); }
      ast::expr_copy(a) {
        need_expr_kind(tcx, a, ast::kind_shared, "'copy' operand");
      }
      ast::expr_ret(option::some(a)) {
        need_expr_kind(tcx, a, ast::kind_shared, "'ret' operand");
      }
      ast::expr_be(a) {
        need_expr_kind(tcx, a, ast::kind_shared, "'be' operand");
      }
      ast::expr_fail(option::some(a)) {
        need_expr_kind(tcx, a, ast::kind_shared, "'fail' operand");
      }
      ast::expr_call(callee, _, _) {
        let tpt = ty::expr_ty_params_and_ty(tcx, callee);

        // If we have typarams, we're calling an item; we need to check
        // that all the types we're supplying as typarams conform to the
        // typaram kind constraints on that item.
        if vec::len(tpt.params) != 0u {
            let callee_def =
                ast_util::def_id_of_def(tcx.def_map.get(callee.id));
            let item_tk = ty::lookup_item_type(tcx, callee_def);
            let i = 0;
            assert (vec::len(item_tk.kinds) == vec::len(tpt.params));
            for k_need: ast::kind in item_tk.kinds {
                let t = tpt.params[i];
                demand_kind(tcx, e.span, t, k_need,
                            #fmt("typaram %d", i));
                i += 1;
            }
        }
      }
      ast::expr_unary(op, a) {
        alt op {
          ast::box(_) {
            need_shared_or_pinned_ctor(tcx, a, "'@' operand");
          }
          ast::uniq(_) {
            need_shared_or_pinned_ctor(tcx, a, "'~' operand");
          }
          _ { /* fall through */ }
        }
      }
      ast::expr_rec(fields, _) {
        for field in fields {
            need_shared_or_pinned_ctor(tcx, field.node.expr, "record field");
        }
      }
      ast::expr_tup(exprs) {
        for expr in exprs {
            need_shared_or_pinned_ctor(tcx, expr, "tuple parameter");
        }
      }
      ast::expr_vec(exprs, _) {
        // Putting pinned things into vectors is pretty useless since vector
        // addition can't work (it's a copy)
        for expr in exprs {
            need_expr_kind(tcx, expr, ast::kind_shared, "vector element");
        }
      }
      _ { }
    }
}

fn check_stmt(tcx: ty::ctxt, stmt: @ast::stmt) {
    alt stmt.node {
      ast::stmt_decl(@{node: ast::decl_local(locals), _}, _) {
        for (let_style, local) in locals {
            alt local.node.init {
              option::some({op: ast::init_assign., expr}) {
                need_shared_or_pinned_ctor(tcx, expr,
                                           "local initializer");
              }
              option::some({op: ast::init_move., expr}) {
                need_shared_or_pinned_ctor(tcx, expr,
                                           "local initializer");
              }
              option::none. { /* fall through */ }
            }
        }
      }
      _ { /* fall through */ }
    }
}
*/

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
