// Determines the ways in which a generic function body depends
// on its type parameters. Used to aggressively reuse compiled
// function bodies for different types.

// This unfortunately depends on quite a bit of knowledge about the
// details of the language semantics, and is likely to accidentally go
// out of sync when something is changed. It could be made more
// powerful by distinguishing between functions that only need to know
// the size and alignment of a type, and those that also use its
// drop/take glue. But this would increase the fragility of the code
// to a ridiculous level, and probably not catch all that many extra
// opportunities for reuse.

// (An other approach to doing what this code does is to instrument
// the translation code to set flags whenever it does something like
// alloca a type or get a tydesc. This would not duplicate quite as
// much information, but have the disadvantage of being very
// invasive.)

use std::map::hashmap;
use std::list;
use std::list::{List, Cons, Nil};
use driver::session::session;
use metadata::csearch;
use syntax::ast::*, syntax::ast_util, syntax::visit;
use syntax::ast_map;
use common::*;

type type_uses = uint; // Bitmask
const use_repr: uint = 1u; // Dependency on size/alignment and take/drop glue
const use_tydesc: uint = 2u; // Takes the tydesc, or compares

type ctx = {ccx: @crate_ctxt,
            uses: ~[mut type_uses]};

fn type_uses_for(ccx: @crate_ctxt, fn_id: def_id, n_tps: uint)
    -> ~[type_uses] {
    match ccx.type_use_cache.find(fn_id) {
      Some(uses) => return uses,
      None => ()
    }
    let fn_id_loc = if fn_id.crate == local_crate { fn_id }
                    else { base::maybe_instantiate_inline(ccx, fn_id) };
    // Conservatively assume full use for recursive loops
    ccx.type_use_cache.insert(fn_id, vec::from_elem(n_tps, 3u));

    let cx = {ccx: ccx, uses: vec::to_mut(vec::from_elem(n_tps, 0u))};
    match ty::get(ty::lookup_item_type(cx.ccx.tcx, fn_id).ty).struct {
      ty::ty_fn({inputs, _}) => {
        for vec::each(inputs) |arg| {
            if arg.mode == expl(by_val) { type_needs(cx, use_repr, arg.ty); }
        }
      }
      _ => ()
    }

    if fn_id_loc.crate != local_crate {
        let uses = vec::from_mut(copy cx.uses);
        ccx.type_use_cache.insert(fn_id, uses);
        return uses;
    }
    let map_node = match ccx.tcx.items.find(fn_id_loc.node) {
        Some(x) => x,
        None    => ccx.sess.bug(fmt!("type_uses_for: unbound item ID %?",
                                     fn_id_loc))
    };
    match map_node {
      ast_map::node_item(@{node: item_fn(_, _, _, body), _}, _) |
      ast_map::node_method(@{body, _}, _, _) => {
        handle_body(cx, body);
      }
      ast_map::node_trait_method(*) => {
        // This will be a static trait method. For now, we just assume
        // it fully depends on all of the type information. (Doing
        // otherwise would require finding the actual implementation).
        for uint::range(0u, n_tps) |n| { cx.uses[n] |= use_repr|use_tydesc;}
      }
      ast_map::node_variant(_, _, _) => {
        for uint::range(0u, n_tps) |n| { cx.uses[n] |= use_repr;}
      }
      ast_map::node_foreign_item(i@@{node: foreign_item_fn(*), _},
                                 abi, _) => {
        if abi == foreign_abi_rust_intrinsic {
            let flags = match cx.ccx.sess.str_of(i.ident) {
              ~"size_of" |  ~"pref_align_of" | ~"min_align_of" |
              ~"init" |  ~"reinterpret_cast" |
              ~"move_val" | ~"move_val_init" => {
                use_repr
              }
              ~"get_tydesc" | ~"needs_drop" => {
                use_tydesc
              }
              ~"atomic_xchg"     | ~"atomic_xadd"     | ~"atomic_xsub" |
              ~"atomic_xchg_acq" | ~"atomic_xadd_acq" | ~"atomic_xsub_acq" |
              ~"atomic_xchg_rel" | ~"atomic_xadd_rel" | ~"atomic_xsub_rel" =>
              { 0u }
              ~"visit_tydesc" | ~"forget" | ~"addr_of" |
              ~"frame_address" | ~"morestack_addr" => {
                0u
              }
              // would be cool to make these an enum instead of strings!
              _ => fail fmt!("unknown intrinsic in type_use %?",
                             cx.ccx.sess.str_of(i.ident))
            };
            for uint::range(0u, n_tps) |n| { cx.uses[n] |= flags;}
        }
      }
      ast_map::node_ctor(_, _, ctor, _, _) => {
        handle_body(cx, ctor.node.body);
      }
      ast_map::node_dtor(_, dtor, _, _) => {
        handle_body(cx, dtor.node.body);
      }
      _ => fail ~"unknown node type in type_use"
    }
    let uses = vec::from_mut(copy cx.uses);
    ccx.type_use_cache.insert(fn_id, uses);
    uses
}

fn type_needs(cx: ctx, use: uint, ty: ty::t) {
    // Optimization -- don't descend type if all params already have this use
    for vec::each_mut(cx.uses) |u| {
        if *u & use != use {
            type_needs_inner(cx, use, ty, @Nil);
            return;
        }
    }
}

fn type_needs_inner(cx: ctx, use: uint, ty: ty::t,
                    enums_seen: @List<def_id>) {
    do ty::maybe_walk_ty(ty) |ty| {
        if ty::type_has_params(ty) {
            match ty::get(ty).struct {
                /*
                 This previously included ty_box -- that was wrong
                 because if we cast an @T to an trait (for example) and return
                 it, we depend on the drop glue for T (we have to write the
                 right tydesc into the result)
                 */
              ty::ty_fn(_) | ty::ty_ptr(_) | ty::ty_rptr(_, _)
               | ty::ty_trait(_, _, _) => false,
              ty::ty_enum(did, substs) => {
                if option::is_none(list::find(enums_seen, |id| *id == did)) {
                    let seen = @Cons(did, enums_seen);
                    for vec::each(*ty::enum_variants(cx.ccx.tcx, did)) |v| {
                        for vec::each(v.args) |aty| {
                            let t = ty::subst(cx.ccx.tcx, &substs, aty);
                            type_needs_inner(cx, use, t, seen);
                        }
                    }
                }
                false
              }
              ty::ty_param(p) => {
                cx.uses[p.idx] |= use;
                false
              }
              _ => true
            }
        } else { false }
    }
}

fn node_type_needs(cx: ctx, use: uint, id: node_id) {
    type_needs(cx, use, ty::node_id_to_type(cx.ccx.tcx, id));
}

fn mark_for_expr(cx: ctx, e: @expr) {
    match e.node {
      expr_vstore(_, _) |
      expr_vec(_, _) |
      expr_rec(_, _) | expr_struct(*) | expr_tup(_) |
      expr_unary(box(_), _) | expr_unary(uniq(_), _) |
      expr_binary(add, _, _) |
      expr_copy(_) | expr_move(_, _) | expr_unary_move(_) |
      expr_repeat(*) => {
        node_type_needs(cx, use_repr, e.id);
      }
      expr_cast(base, _) => {
        let result_t = ty::node_id_to_type(cx.ccx.tcx, e.id);
        match ty::get(result_t).struct {
            ty::ty_trait(*) => {
              // When we're casting to an trait, we need the
              // tydesc for the expr that's being cast.
              node_type_needs(cx, use_tydesc, base.id);
            }
            _ => ()
        }
      }
      expr_binary(op, lhs, _) => {
        match op {
          eq | lt | le | ne | ge | gt => {
            node_type_needs(cx, use_tydesc, lhs.id)
          }
          _ => ()
        }
      }
      expr_path(_) => {
        do cx.ccx.tcx.node_type_substs.find(e.id).iter |ts| {
            let id = ast_util::def_id_of_def(cx.ccx.tcx.def_map.get(e.id));
            vec::iter2(type_uses_for(cx.ccx, id, ts.len()), ts,
                       |uses, subst| {
                           type_needs(cx, uses, subst)
                       })
        }
      }
      expr_fn(*) | expr_fn_block(*) => {
        match ty::ty_fn_proto(ty::expr_ty(cx.ccx.tcx, e)) {
          ty::proto_bare | ty::proto_vstore(ty::vstore_uniq) => {}
          ty::proto_vstore(ty::vstore_box) |
          ty::proto_vstore(ty::vstore_slice(_)) => {
            for vec::each(*freevars::get_freevars(cx.ccx.tcx, e.id)) |fv| {
                let node_id = ast_util::def_id_of_def(fv.def).node;
                node_type_needs(cx, use_repr, node_id);
            }
          }
          ty::proto_vstore(ty::vstore_fixed(_)) =>
            fail ~"vstore_fixed not allowed here"
        }
      }
      expr_assign(val, _) | expr_swap(val, _) | expr_assign_op(_, val, _) |
      expr_ret(Some(val)) => {
        node_type_needs(cx, use_repr, val.id);
      }
      expr_index(base, _) | expr_field(base, _, _) => {
        // FIXME (#2537): could be more careful and not count fields after
        // the chosen field.
        let base_ty = ty::node_id_to_type(cx.ccx.tcx, base.id);
        type_needs(cx, use_repr, ty::type_autoderef(cx.ccx.tcx, base_ty));

        do option::iter(cx.ccx.maps.method_map.find(e.id)) |mth| {
            match mth.origin {
              typeck::method_static(did) => {
                do option::iter(cx.ccx.tcx.node_type_substs.find(e.id)) |ts| {
                    do vec::iter2(type_uses_for(cx.ccx, did, ts.len()), ts)
                        |uses, subst| { type_needs(cx, uses, subst)}
                }
              }
              typeck::method_param({param_num: param, _}) => {
                cx.uses[param] |= use_tydesc;
              }
              typeck::method_trait(_, _) => (),
            }
        }
      }
      expr_log(_, _, val) => {
        node_type_needs(cx, use_tydesc, val.id);
      }
      expr_call(f, _, _) => {
        vec::iter(ty::ty_fn_args(ty::node_id_to_type(cx.ccx.tcx, f.id)), |a| {
            match a.mode {
              expl(by_move) | expl(by_copy) | expl(by_val) => {
                type_needs(cx, use_repr, a.ty);
              }
              _ => ()
            }
        })
      }
      expr_match(*) | expr_block(_) | expr_if(*) |
      expr_while(*) | expr_fail(_) | expr_break(_) | expr_again(_) |
      expr_unary(_, _) | expr_lit(_) | expr_assert(_) |
      expr_mac(_) | expr_addr_of(_, _) |
      expr_ret(_) | expr_loop(_, _) |
      expr_loop_body(_) | expr_do_body(_) => ()
    }
}

fn handle_body(cx: ctx, body: blk) {
    let v = visit::mk_vt(@{
        visit_expr: |e, cx, v| {
            visit::visit_expr(e, cx, v);
            mark_for_expr(cx, e);
        },
        visit_local: |l, cx, v| {
            visit::visit_local(l, cx, v);
            node_type_needs(cx, use_repr, l.node.id);
        },
        visit_pat: |p, cx, v| {
            visit::visit_pat(p, cx, v);
            node_type_needs(cx, use_repr, p.id);
        },
        visit_block: |b, cx, v| {
            visit::visit_block(b, cx, v);
            do option::iter(b.node.expr) |e| {
                node_type_needs(cx, use_repr, e.id);
            }
        },
        visit_item: |_i, _cx, _v| { }
        with *visit::default_visitor()
    });
    v.visit_block(body, cx, v);
}
