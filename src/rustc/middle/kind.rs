import syntax::{visit, ast_util};
import syntax::ast::*;
import syntax::codemap::span;
import ty::{kind, kind_sendable, kind_copyable, kind_noncopyable, kind_const,
           operators};
import driver::session::session;
import std::map::hashmap;
import util::ppaux::{ty_to_str, tys_to_str};
import syntax::print::pprust::expr_to_str;
import freevars::freevar_entry;
import dvec::extensions;

// Kind analysis pass.
//
// There are several kinds defined by various operations. The most restrictive
// kind is noncopyable. The noncopyable kind can be extended with any number
// of the following attributes.
//
//  send: Things that can be sent on channels or included in spawned closures.
//  copy: Things that can be copied.
//  const: Things thare are deeply immutable. They are guaranteed never to
//    change, and can be safely shared without copying between tasks.
//
// Send includes scalar types, resources and unique types containing only
// sendable types.
//
// Copy includes boxes, closure and unique types containing copyable types.
//
// Const include scalar types, things without non-const fields, and pointers
// to const things.
//
// This pass ensures that type parameters are only instantiated with types
// whose kinds are equal or less general than the way the type parameter was
// annotated (with the `send`, `copy` or `const` keyword).
//
// It also verifies that noncopyable kinds are not copied. Sendability is not
// applied, since none of our language primitives send. Instead, the sending
// primitives in the stdlib are explicitly annotated to only take sendable
// types.

fn kind_to_str(k: kind) -> str {
    let mut kinds = [];
    if ty::kind_lteq(kind_const(), k) {
        kinds += ["const"];
    }
    if ty::kind_can_be_copied(k) {
        kinds += ["copy"];
    }
    if ty::kind_can_be_sent(k) {
        kinds += ["send"];
    }
    str::connect(kinds, " ")
}

type rval_map = std::map::hashmap<node_id, ()>;

type ctx = {tcx: ty::ctxt,
            rval_map: rval_map,
            method_map: typeck::method_map,
            last_use_map: liveness::last_use_map};

fn check_crate(tcx: ty::ctxt, method_map: typeck::method_map,
               last_use_map: liveness::last_use_map, crate: @crate)
    -> rval_map {
    let ctx = {tcx: tcx,
               rval_map: std::map::int_hash(),
               method_map: method_map,
               last_use_map: last_use_map};
    let visit = visit::mk_vt(@{
        visit_expr: check_expr,
        visit_stmt: check_stmt,
        visit_block: check_block,
        visit_fn: check_fn,
        visit_ty: check_ty
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, ctx, visit);
    tcx.sess.abort_if_errors();
    ret ctx.rval_map;
}

type check_fn = fn@(ctx, option<@freevar_entry>, bool, ty::t, sp: span);

// Yields the appropriate function to check the kind of closed over
// variables. `id` is the node_id for some expression that creates the
// closure.
fn with_appropriate_checker(cx: ctx, id: node_id, b: fn(check_fn)) {
    fn check_for_uniq(cx: ctx, fv: option<@freevar_entry>, is_move: bool,
                      var_t: ty::t, sp: span) {
        // all captured data must be sendable, regardless of whether it is
        // moved in or copied in
        check_send(cx, var_t, sp);

        // copied in data must be copyable, but moved in data can be anything
        let is_implicit = fv.is_some();
        if !is_move { check_copy(cx, var_t, sp, is_implicit); }

        // check that only immutable variables are implicitly copied in
        for fv.each { |fv|
            check_imm_free_var(cx, fv.def, fv.span);
        }
    }

    fn check_for_box(cx: ctx, fv: option<@freevar_entry>, is_move: bool,
                     var_t: ty::t, sp: span) {
        // copied in data must be copyable, but moved in data can be anything
        let is_implicit = fv.is_some();
        if !is_move { check_copy(cx, var_t, sp, is_implicit); }

        // check that only immutable variables are implicitly copied in
        for fv.each { |fv|
            check_imm_free_var(cx, fv.def, fv.span);
        }
    }

    fn check_for_block(cx: ctx, fv: option<@freevar_entry>, _is_move: bool,
                       _var_t: ty::t, sp: span) {
        // only restriction: no capture clauses (we would have to take
        // ownership of the moved/copied in data).
        if fv.is_none() {
            cx.tcx.sess.span_err(
                sp,
                "cannot capture values explicitly with a block closure");
        }
    }

    fn check_for_bare(cx: ctx, _fv: option<@freevar_entry>, _is_move: bool,
                      _var_t: ty::t, sp: span) {
        cx.tcx.sess.span_err(sp, "attempted dynamic environment capture");
    }

    let fty = ty::node_id_to_type(cx.tcx, id);
    alt ty::ty_fn_proto(fty) {
      proto_uniq { b(check_for_uniq) }
      proto_box { b(check_for_box) }
      proto_bare { b(check_for_bare) }
      proto_any | proto_block { b(check_for_block) }
    }
}

// Check that the free variables used in a shared/sendable closure conform
// to the copy/move kind bounds. Then recursively check the function body.
fn check_fn(fk: visit::fn_kind, decl: fn_decl, body: blk, sp: span,
            fn_id: node_id, cx: ctx, v: visit::vt<ctx>) {

    // Find the check function that enforces the appropriate bounds for this
    // kind of function:
    with_appropriate_checker(cx, fn_id) { |chk|

        // Begin by checking the variables in the capture clause, if any.
        // Here we slightly abuse the map function to both check and report
        // errors and produce a list of the def id's for all capture
        // variables.  This list is used below to avoid checking and reporting
        // on a given variable twice.
        let cap_clause = alt fk {
          visit::fk_anon(_, cc) | visit::fk_fn_block(cc) { cc }
          visit::fk_item_fn(*) | visit::fk_method(*) |
          visit::fk_res(*) | visit::fk_ctor(*) | visit::fk_dtor(*) { @[] }
        };
        let captured_vars = (*cap_clause).map { |cap_item|
            let cap_def = cx.tcx.def_map.get(cap_item.id);
            let cap_def_id = ast_util::def_id_of_def(cap_def).node;
            let ty = ty::node_id_to_type(cx.tcx, cap_def_id);
            chk(cx, none, cap_item.is_move, ty, cap_item.span);
            cap_def_id
        };

        // Iterate over any free variables that may not have appeared in the
        // capture list.  Ensure that they too are of the appropriate kind.
        for vec::each(*freevars::get_freevars(cx.tcx, fn_id)) {|fv|
            let id = ast_util::def_id_of_def(fv.def).node;

            // skip over free variables that appear in the cap clause
            if captured_vars.contains(id) { cont; }

            // if this is the last use of the variable, then it will be
            // a move and not a copy
            let is_move = {
                alt check cx.last_use_map.find(fn_id) {
                  some(vars) {(*vars).contains(id)}
                  none {false}
                }
            };

            let ty = ty::node_id_to_type(cx.tcx, id);
            chk(cx, some(fv), is_move, ty, fv.span);
        }
    }

    visit::visit_fn(fk, decl, body, sp, fn_id, cx, v);
}

fn check_block(b: blk, cx: ctx, v: visit::vt<ctx>) {
    alt b.node.expr {
      some(ex) { maybe_copy(cx, ex); }
      _ {}
    }
    visit::visit_block(b, cx, v);
}

fn check_expr(e: @expr, cx: ctx, v: visit::vt<ctx>) {
    #debug["kind::check_expr(%s)", expr_to_str(e)];
    alt e.node {
      expr_assign(_, ex) |
      expr_unary(box(_), ex) | expr_unary(uniq(_), ex) |
      expr_ret(some(ex)) | expr_cast(ex, _) { maybe_copy(cx, ex); }
      expr_copy(expr) { check_copy_ex(cx, expr, false); }
      // Vector add copies, but not "implicitly"
      expr_assign_op(_, _, ex) { check_copy_ex(cx, ex, false) }
      expr_binary(add, ls, rs) {
        check_copy_ex(cx, ls, false);
        check_copy_ex(cx, rs, false);
      }
      expr_rec(fields, def) {
        for fields.each {|field| maybe_copy(cx, field.node.expr); }
        alt def {
          some(ex) {
            // All noncopyable fields must be overridden
            let t = ty::expr_ty(cx.tcx, ex);
            let ty_fields = alt ty::get(t).struct {
              ty::ty_rec(f) { f }
              _ { cx.tcx.sess.span_bug(ex.span, "bad expr type in record"); }
            };
            for ty_fields.each {|tf|
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
        for exprs.each {|expr| maybe_copy(cx, expr); }
      }
      expr_bind(_, args) {
        for args.each {|a| alt a { some(ex) { maybe_copy(cx, ex); } _ {} } }
      }
      expr_call(f, args, _) {
        let mut i = 0u;
        for ty::ty_fn_args(ty::expr_ty(cx.tcx, f)).each {|arg_t|
            alt ty::arg_mode(cx.tcx, arg_t) {
              by_copy { maybe_copy(cx, args[i]); }
              by_ref | by_val | by_mutbl_ref | by_move { }
            }
            i += 1u;
        }
      }
      expr_path(_) | expr_field(_, _, _) {
        option::iter(cx.tcx.node_type_substs.find(e.id)) {|ts|
            let bounds = alt check e.node {
              expr_path(_) {
                let did = ast_util::def_id_of_def(cx.tcx.def_map.get(e.id));
                ty::lookup_item_type(cx.tcx, did).bounds
              }
              expr_field(base, _, _) {
                alt cx.method_map.get(e.id) {
                  typeck::method_static(did) {
                   /*
                        If this is a class method, we want to use the
                        class bounds plus the method bounds -- otherwise the
                        indices come out wrong. So we check base's type...
                   */
                   let mut bounds = ty::lookup_item_type(cx.tcx, did).bounds;
                   alt ty::get(ty::node_id_to_type(cx.tcx, base.id)).struct {
                        ty::ty_class(parent_id, ts) {
                            /* ...and if it has a class type, prepend the
                               class bounds onto the method bounds */
                            /* n.b. this code is very likely sketchy --
                             currently, class-impl-very-parameterized-iface
                             fails here and is thus xfailed */
                            bounds =
                             @(*ty::lookup_item_type(cx.tcx, parent_id).bounds
                               + *bounds);
                        }
                        _ { }
                      }
                      bounds
                  }
                  typeck::method_param(ifce_id, n_mth, _, _) |
                  typeck::method_iface(ifce_id, n_mth) {
                    let ifce_bounds =
                        ty::lookup_item_type(cx.tcx, ifce_id).bounds;
                    let mth = ty::iface_methods(cx.tcx, ifce_id)[n_mth];
                    @(*ifce_bounds + *mth.tps)
                  }
                }
              }
            };
            if vec::len(ts) != vec::len(*bounds) {
              // Fail earlier to make debugging easier
              fail #fmt("Internal error: in kind::check_expr, length \
                  mismatch between actual and declared bounds: actual = \
                  %s (%u tys), declared = %? (%u tys)",
                  tys_to_str(cx.tcx, ts), ts.len(), *bounds, (*bounds).len());
            }
            vec::iter2(ts, *bounds) {|ty, bound|
                check_bounds(cx, e.span, ty, bound)
            }
        }
      }
      _ { }
    }
    visit::visit_expr(e, cx, v);
}

fn check_stmt(stmt: @stmt, cx: ctx, v: visit::vt<ctx>) {
    alt stmt.node {
      stmt_decl(@{node: decl_local(locals), _}, _) {
        for locals.each {|local|
            alt local.node.init {
              some({op: init_assign, expr}) { maybe_copy(cx, expr); }
              _ {}
            }
        }
      }
      _ {}
    }
    visit::visit_stmt(stmt, cx, v);
}

fn check_ty(aty: @ty, cx: ctx, v: visit::vt<ctx>) {
    alt aty.node {
      ty_path(_, id) {
        option::iter(cx.tcx.node_type_substs.find(id)) {|ts|
            let did = ast_util::def_id_of_def(cx.tcx.def_map.get(id));
            let bounds = ty::lookup_item_type(cx.tcx, did).bounds;
            vec::iter2(ts, *bounds) {|ty, bound|
                check_bounds(cx, aty.span, ty, bound)
            }
        }
      }
      _ {}
    }
    visit::visit_ty(aty, cx, v);
}

fn check_bounds(cx: ctx, sp: span, ty: ty::t, bounds: ty::param_bounds) {
    let kind = ty::type_kind(cx.tcx, ty);
    let p_kind = ty::param_bounds_to_kind(bounds);
    if !ty::kind_lteq(p_kind, kind) {
        cx.tcx.sess.span_err(
            sp, "instantiating a type parameter with an incompatible type " +
            "(needs `" + kind_to_str(p_kind) +
            "`, got `" + kind_to_str(kind) +
            "`, missing `" + kind_to_str(p_kind - kind) + "`)");
    }
}

fn maybe_copy(cx: ctx, ex: @expr) {
    check_copy_ex(cx, ex, true);
}

fn is_nullary_variant(cx: ctx, ex: @expr) -> bool {
    alt ex.node {
      expr_path(_) {
        alt cx.tcx.def_map.get(ex.id) {
          def_variant(edid, vdid) {
            vec::len(ty::enum_variant_with_id(cx.tcx, edid, vdid).args) == 0u
          }
          _ { false }
        }
      }
      _ { false }
    }
}

fn check_copy_ex(cx: ctx, ex: @expr, implicit_copy: bool) {
    if ty::expr_is_lval(cx.method_map, ex) &&
       !cx.last_use_map.contains_key(ex.id) &&
       !is_nullary_variant(cx, ex) {
        let ty = ty::expr_ty(cx.tcx, ex);
        check_copy(cx, ty, ex.span, implicit_copy);
    }
}

fn check_imm_free_var(cx: ctx, def: def, sp: span) {
    let msg = "mutable variables cannot be implicitly captured; \
               use a capture clause";
    alt def {
      def_local(_, is_mutbl) {
        if is_mutbl {
            cx.tcx.sess.span_err(sp, msg);
        }
      }
      def_arg(_, mode) {
        alt ty::resolved_mode(cx.tcx, mode) {
          by_ref | by_val | by_move | by_copy { /* ok */ }
          by_mutbl_ref {
            cx.tcx.sess.span_err(sp, msg);
          }
        }
      }
      def_upvar(_, def1, _) {
        check_imm_free_var(cx, *def1, sp);
      }
      def_binding(*) | def_self(*) { /*ok*/ }
      _ {
        cx.tcx.sess.span_bug(
            sp,
            #fmt["unknown def for free variable: %?", def]);
      }
    }
}

fn check_copy(cx: ctx, ty: ty::t, sp: span, implicit_copy: bool) {
    let k = ty::type_kind(cx.tcx, ty);
    if !ty::kind_can_be_copied(k) {
        cx.tcx.sess.span_err(sp, "copying a noncopyable value");
    } else if implicit_copy && !ty::kind_can_be_implicitly_copied(k) {
        cx.tcx.sess.span_warn(
            sp,
            "implicitly copying a non-implicitly-copyable value");
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
