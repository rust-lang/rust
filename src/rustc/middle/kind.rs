use syntax::{visit, ast_util};
use syntax::ast::*;
use syntax::codemap::span;
use ty::{kind, kind_copyable, kind_noncopyable, kind_const};
use driver::session::session;
use std::map::HashMap;
use util::ppaux::{ty_to_str, tys_to_str};
use syntax::print::pprust::expr_to_str;
use freevars::freevar_entry;
use lint::{non_implicitly_copyable_typarams,implicit_copies};

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
//  owned: Things that do not contain borrowed pointers.
//
// Send includes scalar types as well as classes and unique types containing
// only sendable types.
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

fn kind_to_str(k: kind) -> ~str {
    let mut kinds = ~[];

    if ty::kind_lteq(kind_const(), k) {
        kinds.push(~"const");
    }

    if ty::kind_can_be_copied(k) {
        kinds.push(~"copy");
    }

    if ty::kind_can_be_sent(k) {
        kinds.push(~"send");
    } else if ty::kind_is_owned(k) {
        kinds.push(~"owned");
    }

    str::connect(kinds, ~" ")
}

type rval_map = std::map::HashMap<node_id, ()>;

type ctx = {tcx: ty::ctxt,
            method_map: typeck::method_map,
            last_use_map: liveness::last_use_map,
            current_item: node_id};

fn check_crate(tcx: ty::ctxt,
               method_map: typeck::method_map,
               last_use_map: liveness::last_use_map,
               crate: @crate) {
    let ctx = {tcx: tcx,
               method_map: method_map,
               last_use_map: last_use_map,
               current_item: -1};
    let visit = visit::mk_vt(@{
        visit_arm: check_arm,
        visit_expr: check_expr,
        visit_stmt: check_stmt,
        visit_block: check_block,
        visit_fn: check_fn,
        visit_ty: check_ty,
        visit_item: fn@(i: @item, cx: ctx, v: visit::vt<ctx>) {
            visit::visit_item(i, {current_item: i.id,.. cx}, v);
        },
        .. *visit::default_visitor()
    });
    visit::visit_crate(*crate, ctx, visit);
    tcx.sess.abort_if_errors();
}

type check_fn = fn@(ctx, node_id, Option<@freevar_entry>,
                    bool, ty::t, sp: span);

// Yields the appropriate function to check the kind of closed over
// variables. `id` is the node_id for some expression that creates the
// closure.
fn with_appropriate_checker(cx: ctx, id: node_id, b: fn(check_fn)) {
    fn check_for_uniq(cx: ctx, id: node_id, fv: Option<@freevar_entry>,
                      is_move: bool, var_t: ty::t, sp: span) {
        // all captured data must be sendable, regardless of whether it is
        // moved in or copied in.  Note that send implies owned.
        if !check_send(cx, var_t, sp) { return; }

        // copied in data must be copyable, but moved in data can be anything
        let is_implicit = fv.is_some();
        if !is_move {
            check_copy(cx, id, var_t, sp, is_implicit,
                       Some(("non-copyable value cannot be copied into a \
                              ~fn closure",
                             "to copy values into a ~fn closure, use a \
                              capture clause: `fn~(copy x)` or `|copy x|`")));
        }

        // check that only immutable variables are implicitly copied in
        for fv.each |fv| {
            check_imm_free_var(cx, fv.def, fv.span);
        }
    }

    fn check_for_box(cx: ctx, id: node_id, fv: Option<@freevar_entry>,
                     is_move: bool, var_t: ty::t, sp: span) {
        // all captured data must be owned
        if !check_owned(cx.tcx, var_t, sp) { return; }

        // copied in data must be copyable, but moved in data can be anything
        let is_implicit = fv.is_some();
        if !is_move {
            check_copy(cx, id, var_t, sp, is_implicit,
                       Some(("non-copyable value cannot be copied into a \
                              @fn closure",
                             "to copy values into a @fn closure, use a \
                              capture clause: `fn~(copy x)` or `|copy x|`")));
        }

        // check that only immutable variables are implicitly copied in
        for fv.each |fv| {
            check_imm_free_var(cx, fv.def, fv.span);
        }
    }

    fn check_for_block(cx: ctx, _id: node_id, fv: Option<@freevar_entry>,
                       _is_move: bool, _var_t: ty::t, sp: span) {
        // only restriction: no capture clauses (we would have to take
        // ownership of the moved/copied in data).
        if fv.is_none() {
            cx.tcx.sess.span_err(
                sp,
                ~"cannot capture values explicitly with a block closure");
        }
    }

    fn check_for_bare(cx: ctx, _id: node_id, _fv: Option<@freevar_entry>,
                      _is_move: bool,_var_t: ty::t, sp: span) {
        cx.tcx.sess.span_err(sp, ~"attempted dynamic environment capture");
    }

    let fty = ty::node_id_to_type(cx.tcx, id);
    match ty::ty_fn_proto(fty) {
      ty::proto_vstore(ty::vstore_uniq) => b(check_for_uniq),
      ty::proto_vstore(ty::vstore_box) => b(check_for_box),
      ty::proto_bare => b(check_for_bare),
      ty::proto_vstore(ty::vstore_slice(_)) => b(check_for_block),
      ty::proto_vstore(ty::vstore_fixed(_)) =>
        fail ~"fixed vstore not allowed here"
    }
}

// Check that the free variables used in a shared/sendable closure conform
// to the copy/move kind bounds. Then recursively check the function body.
fn check_fn(fk: visit::fn_kind, decl: fn_decl, body: blk, sp: span,
            fn_id: node_id, cx: ctx, v: visit::vt<ctx>) {

    // Find the check function that enforces the appropriate bounds for this
    // kind of function:
    do with_appropriate_checker(cx, fn_id) |chk| {

        // Begin by checking the variables in the capture clause, if any.
        // Here we slightly abuse the map function to both check and report
        // errors and produce a list of the def id's for all capture
        // variables.  This list is used below to avoid checking and reporting
        // on a given variable twice.
        let cap_clause = match fk {
          visit::fk_anon(_, cc) | visit::fk_fn_block(cc) => cc,
          visit::fk_item_fn(*) | visit::fk_method(*) |
          visit::fk_ctor(*) | visit::fk_dtor(*) => @~[]
        };
        let captured_vars = do (*cap_clause).map |cap_item| {
            let cap_def = cx.tcx.def_map.get(cap_item.id);
            let cap_def_id = ast_util::def_id_of_def(cap_def).node;
            let ty = ty::node_id_to_type(cx.tcx, cap_def_id);
            chk(cx, fn_id, None, cap_item.is_move, ty, cap_item.span);
            cap_def_id
        };

        // Iterate over any free variables that may not have appeared in the
        // capture list.  Ensure that they too are of the appropriate kind.
        for vec::each(*freevars::get_freevars(cx.tcx, fn_id)) |fv| {
            let id = ast_util::def_id_of_def(fv.def).node;

            // skip over free variables that appear in the cap clause
            if captured_vars.contains(&id) { loop; }

            // if this is the last use of the variable, then it will be
            // a move and not a copy
            let is_move = {
                match cx.last_use_map.find(fn_id) {
                  Some(vars) => (*vars).contains(&id),
                  None => false
                }
            };

            let ty = ty::node_id_to_type(cx.tcx, id);
            chk(cx, fn_id, Some(*fv), is_move, ty, fv.span);
        }
    }

    visit::visit_fn(fk, decl, body, sp, fn_id, cx, v);
}

fn check_block(b: blk, cx: ctx, v: visit::vt<ctx>) {
    match b.node.expr {
      Some(ex) => maybe_copy(cx, ex, None),
      _ => ()
    }
    visit::visit_block(b, cx, v);
}

fn check_arm(a: arm, cx: ctx, v: visit::vt<ctx>) {
    for vec::each(a.pats) |p| {
        do pat_util::pat_bindings(cx.tcx.def_map, *p) |mode, id, span, _pth| {
            if mode == bind_by_value {
                let t = ty::node_id_to_type(cx.tcx, id);
                let reason = "consider binding with `ref` or `move` instead";
                check_copy(cx, id, t, span, false, Some((reason,reason)));
            }
        }
    }
    visit::visit_arm(a, cx, v);
}

fn check_expr(e: @expr, cx: ctx, v: visit::vt<ctx>) {
    debug!("kind::check_expr(%s)", expr_to_str(e, cx.tcx.sess.intr()));
    let id_to_use = match e.node {
        expr_index(*)|expr_assign_op(*)|
        expr_unary(*)|expr_binary(*) => e.callee_id,
        _ => e.id
    };

    // Handle any kind bounds on type parameters
    do option::iter(&cx.tcx.node_type_substs.find(id_to_use)) |ts| {
        let bounds = match e.node {
          expr_path(_) => {
            let did = ast_util::def_id_of_def(cx.tcx.def_map.get(e.id));
            ty::lookup_item_type(cx.tcx, did).bounds
          }
          _ => {
            // Type substitutions should only occur on paths and
            // method calls, so this needs to be a method call.

            // Even though the callee_id may have been the id with
            // node_type_substs, e.id is correct here.
            ty::method_call_bounds(cx.tcx, cx.method_map, e.id).expect(
                ~"non path/method call expr has type substs??")
          }
        };
        if vec::len(*ts) != vec::len(*bounds) {
            // Fail earlier to make debugging easier
            fail fmt!("Internal error: in kind::check_expr, length \
                       mismatch between actual and declared bounds: actual = \
                        %s (%u tys), declared = %? (%u tys)",
                      tys_to_str(cx.tcx, *ts), ts.len(),
                      *bounds, (*bounds).len());
        }
        do vec::iter2(*ts, *bounds) |ty, bound| {
            check_bounds(cx, id_to_use, e.span, ty, bound)
        }
    }

    match e.node {
      expr_assign(_, ex) |
      expr_unary(box(_), ex) | expr_unary(uniq(_), ex) |
      expr_ret(Some(ex)) => {
        maybe_copy(cx, ex, None);
      }
      expr_cast(source, _) => {
        maybe_copy(cx, source, None);
        check_cast_for_escaping_regions(cx, source, e);
      }
      expr_copy(expr) => check_copy_ex(cx, expr, false, None),
      // Vector add copies, but not "implicitly"
      expr_assign_op(_, _, ex) => check_copy_ex(cx, ex, false, None),
      expr_binary(add, ls, rs) => {
        check_copy_ex(cx, ls, false, None);
        check_copy_ex(cx, rs, false, None);
      }
      expr_rec(fields, def) => {
        for fields.each |field| { maybe_copy(cx, field.node.expr, None); }
        match def {
          Some(ex) => {
            // All noncopyable fields must be overridden
            let t = ty::expr_ty(cx.tcx, ex);
            let ty_fields = match ty::get(t).sty {
              ty::ty_rec(f) => f,
              _ => cx.tcx.sess.span_bug(ex.span, ~"bad expr type in record")
            };
            for ty_fields.each |tf| {
                if !vec::any(fields, |f| f.node.ident == tf.ident ) &&
                    !ty::kind_can_be_copied(ty::type_kind(cx.tcx, tf.mt.ty)) {
                    cx.tcx.sess.span_err(ex.span,
                                         ~"copying a noncopyable value");
                }
            }
          }
          _ => {}
        }
      }
      expr_tup(exprs) | expr_vec(exprs, _) => {
        for exprs.each |expr| { maybe_copy(cx, *expr, None); }
      }
      expr_call(f, args, _) => {
        let mut i = 0u;
        for ty::ty_fn_args(ty::expr_ty(cx.tcx, f)).each |arg_t| {
            match ty::arg_mode(cx.tcx, *arg_t) {
              by_copy => maybe_copy(cx, args[i], None),
              by_ref | by_val | by_mutbl_ref | by_move => ()
            }
            i += 1u;
        }
      }
      expr_field(lhs, _, _) => {
        // If this is a method call with a by-val argument, we need
        // to check the copy
        match cx.method_map.find(e.id) {
            Some(ref mme) => {
                match ty::arg_mode(cx.tcx, mme.self_arg) {
                    by_copy => maybe_copy(cx, lhs, None),
                    by_ref | by_val | by_mutbl_ref | by_move => ()
                }
            }
            _ => ()
        }
      }
      expr_repeat(element, count_expr, _) => {
        let count = ty::eval_repeat_count(cx.tcx, count_expr, e.span);
        if count == 1 {
            maybe_copy(cx, element, None);
        } else {
            let element_ty = ty::expr_ty(cx.tcx, element);
            check_copy(cx, element.id, element_ty, element.span, true, None);
        }
      }
      _ => { }
    }
    visit::visit_expr(e, cx, v);
}

fn check_stmt(stmt: @stmt, cx: ctx, v: visit::vt<ctx>) {
    match stmt.node {
      stmt_decl(@{node: decl_local(locals), _}, _) => {
        for locals.each |local| {
            match local.node.init {
              Some({op: init_assign, expr}) => maybe_copy(cx, expr, None),
              _ => {}
            }
        }
      }
      _ => {}
    }
    visit::visit_stmt(stmt, cx, v);
}

fn check_ty(aty: @ty, cx: ctx, v: visit::vt<ctx>) {
    match aty.node {
      ty_path(_, id) => {
        do option::iter(&cx.tcx.node_type_substs.find(id)) |ts| {
            let did = ast_util::def_id_of_def(cx.tcx.def_map.get(id));
            let bounds = ty::lookup_item_type(cx.tcx, did).bounds;
            do vec::iter2(*ts, *bounds) |ty, bound| {
                check_bounds(cx, aty.id, aty.span, ty, bound)
            }
        }
      }
      _ => {}
    }
    visit::visit_ty(aty, cx, v);
}

fn check_bounds(cx: ctx, id: node_id, sp: span,
                ty: ty::t, bounds: ty::param_bounds) {
    let kind = ty::type_kind(cx.tcx, ty);
    let p_kind = ty::param_bounds_to_kind(bounds);
    if !ty::kind_lteq(p_kind, kind) {
        // If the only reason the kind check fails is because the
        // argument type isn't implicitly copyable, consult the warning
        // settings to figure out what to do.
        let implicit = ty::kind_implicitly_copyable() - ty::kind_copyable();
        if ty::kind_lteq(p_kind, kind | implicit) {
            cx.tcx.sess.span_lint(
                non_implicitly_copyable_typarams,
                id, cx.current_item, sp,
                ~"instantiating copy type parameter with a \
                 not implicitly copyable type");
        } else {
            cx.tcx.sess.span_err(
                sp,
                ~"instantiating a type parameter with an incompatible type " +
                ~"(needs `" + kind_to_str(p_kind) +
                ~"`, got `" + kind_to_str(kind) +
                ~"`, missing `" + kind_to_str(p_kind - kind) + ~"`)");
        }
    }
}

fn maybe_copy(cx: ctx, ex: @expr, why: Option<(&str,&str)>) {
    check_copy_ex(cx, ex, true, why);
}

fn is_nullary_variant(cx: ctx, ex: @expr) -> bool {
    match ex.node {
      expr_path(_) => {
        match cx.tcx.def_map.get(ex.id) {
          def_variant(edid, vdid) => {
            vec::len(ty::enum_variant_with_id(cx.tcx, edid, vdid).args) == 0u
          }
          _ => false
        }
      }
      _ => false
    }
}

fn check_copy_ex(cx: ctx, ex: @expr, implicit_copy: bool,
                 why: Option<(&str,&str)>) {
    if ty::expr_is_lval(cx.tcx, cx.method_map, ex) &&

        // this is a move
        !cx.last_use_map.contains_key(ex.id) &&

        // a reference to a constant like `none`... no need to warn
        // about *this* even if the type is Option<~int>
        !is_nullary_variant(cx, ex) &&

        // borrowed unique value isn't really a copy
        !is_autorefd(cx, ex)
    {
        let ty = ty::expr_ty(cx.tcx, ex);
        check_copy(cx, ex.id, ty, ex.span, implicit_copy, why);
    }

    fn is_autorefd(cx: ctx, ex: @expr) -> bool {
        match cx.tcx.adjustments.find(ex.id) {
            None => false,
            Some(ref adj) => adj.autoref.is_some()
        }
    }
}

fn check_imm_free_var(cx: ctx, def: def, sp: span) {
    let msg = ~"mutable variables cannot be implicitly captured; \
               use a capture clause";
    match def {
      def_local(_, is_mutbl) => {
        if is_mutbl {
            cx.tcx.sess.span_err(sp, msg);
        }
      }
      def_arg(_, mode) => {
        match ty::resolved_mode(cx.tcx, mode) {
          by_ref | by_val | by_move | by_copy => { /* ok */ }
          by_mutbl_ref => {
            cx.tcx.sess.span_err(sp, msg);
          }
        }
      }
      def_upvar(_, def1, _, _) => {
        check_imm_free_var(cx, *def1, sp);
      }
      def_binding(*) | def_self(*) => { /*ok*/ }
      _ => {
        cx.tcx.sess.span_bug(
            sp,
            fmt!("unknown def for free variable: %?", def));
      }
    }
}

fn check_copy(cx: ctx, id: node_id, ty: ty::t, sp: span,
              implicit_copy: bool, why: Option<(&str,&str)>) {
    let k = ty::type_kind(cx.tcx, ty);
    if !ty::kind_can_be_copied(k) {
        cx.tcx.sess.span_err(sp, ~"copying a noncopyable value");
        do why.map |reason| {
            cx.tcx.sess.span_note(sp, fmt!("%s", reason.first()));
        };
    } else if implicit_copy && !ty::kind_can_be_implicitly_copied(k) {
        cx.tcx.sess.span_lint(
            implicit_copies, id, cx.current_item,
            sp,
            ~"implicitly copying a non-implicitly-copyable value");
        do why.map |reason| {
            cx.tcx.sess.span_note(sp, fmt!("%s", reason.second()));
        };
    }
}

fn check_send(cx: ctx, ty: ty::t, sp: span) -> bool {
    if !ty::kind_can_be_sent(ty::type_kind(cx.tcx, ty)) {
        cx.tcx.sess.span_err(sp, ~"not a sendable value");
        false
    } else {
        true
    }
}

// note: also used from middle::typeck::regionck!
fn check_owned(tcx: ty::ctxt, ty: ty::t, sp: span) -> bool {
    if !ty::kind_is_owned(ty::type_kind(tcx, ty)) {
        match ty::get(ty).sty {
          ty::ty_param(*) => {
            tcx.sess.span_err(sp, ~"value may contain borrowed \
                                    pointers; use `owned` bound");
          }
          _ => {
            tcx.sess.span_err(sp, ~"value may contain borrowed \
                                    pointers");
          }
        }
        false
    } else {
        true
    }
}

/// This is rather subtle.  When we are casting a value to a
/// instantiated trait like `a as trait/&r`, regionck already ensures
/// that any borrowed pointers that appear in the type of `a` are
/// bounded by `&r`.  However, it is possible that there are *type
/// parameters* in the type of `a`, and those *type parameters* may
/// have borrowed pointers within them.  We have to guarantee that the
/// regions which appear in those type parameters are not obscured.
///
/// Therefore, we ensure that one of three conditions holds:
///
/// (1) The trait instance cannot escape the current fn.  This is
/// guaranteed if the region bound `&r` is some scope within the fn
/// itself.  This case is safe because whatever borrowed pointers are
/// found within the type parameter, they must enclose the fn body
/// itself.
///
/// (2) The type parameter appears in the type of the trait.  For
/// example, if the type parameter is `T` and the trait type is
/// `deque<T>`, then whatever borrowed ptrs may appear in `T` also
/// appear in `deque<T>`.
///
/// (3) The type parameter is owned (and therefore does not contain
/// borrowed ptrs).
fn check_cast_for_escaping_regions(
    cx: ctx,
    source: @expr,
    target: @expr) {

    // Determine what type we are casting to; if it is not an trait, then no
    // worries.
    let target_ty = ty::expr_ty(cx.tcx, target);
    let target_substs = match ty::get(target_ty).sty {
      ty::ty_trait(_, substs, _) => {substs}
      _ => { return; /* not a cast to a trait */ }
    };

    // Check, based on the region associated with the trait, whether it can
    // possibly escape the enclosing fn item (note that all type parameters
    // must have been declared on the enclosing fn item):
    match target_substs.self_r {
      Some(ty::re_scope(*)) => { return; /* case (1) */ }
      None | Some(ty::re_static) | Some(ty::re_free(*)) => {}
      Some(ty::re_bound(*)) | Some(ty::re_var(*)) => {
        cx.tcx.sess.span_bug(
            source.span,
            fmt!("bad region found in kind: %?", target_substs.self_r));
      }
    }

    // Assuming the trait instance can escape, then ensure that each parameter
    // either appears in the trait type or is owned:
    let target_params = ty::param_tys_in_type(target_ty);
    let source_ty = ty::expr_ty(cx.tcx, source);
    do ty::walk_ty(source_ty) |ty| {
        match ty::get(ty).sty {
          ty::ty_param(source_param) => {
            if target_params.contains(&source_param) {
                /* case (2) */
            } else {
                check_owned(cx.tcx, ty, source.span); /* case (3) */
            }
          }
          _ => {}
        }
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
