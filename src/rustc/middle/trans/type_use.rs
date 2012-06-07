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

import std::map::hashmap;
import std::list;
import std::list::{list, cons, nil};
import driver::session::session;
import metadata::csearch;
import syntax::ast::*, syntax::ast_util, syntax::visit;
import syntax::ast_map;
import common::*;

type type_uses = uint; // Bitmask
const use_repr: uint = 1u; // Dependency on size/alignment and take/drop glue
const use_tydesc: uint = 2u; // Takes the tydesc, or compares

type ctx = {ccx: @crate_ctxt,
            uses: [mut type_uses]};

fn type_uses_for(ccx: @crate_ctxt, fn_id: def_id, n_tps: uint)
    -> [type_uses] {
    alt ccx.type_use_cache.find(fn_id) {
      some(uses) { ret uses; }
      none {}
    }
    let fn_id_loc = if fn_id.crate == local_crate { fn_id }
                    else { base::maybe_instantiate_inline(ccx, fn_id) };
    // Conservatively assume full use for recursive loops
    ccx.type_use_cache.insert(fn_id, vec::from_elem(n_tps, 3u));

    let cx = {ccx: ccx, uses: vec::to_mut(vec::from_elem(n_tps, 0u))};
    alt ty::get(ty::lookup_item_type(cx.ccx.tcx, fn_id).ty).struct {
      ty::ty_fn({inputs, _}) {
        for vec::each(inputs) {|arg|
            if arg.mode == expl(by_val) { type_needs(cx, use_repr, arg.ty); }
        }
      }
      _ {}
    }

    if fn_id_loc.crate != local_crate {
        let uses = vec::from_mut(copy cx.uses);
        ccx.type_use_cache.insert(fn_id, uses);
        ret uses;
    }
    let map_node = alt ccx.tcx.items.find(fn_id_loc.node) {
        some(x) { x }
        none    { ccx.sess.bug(#fmt("type_uses_for: unbound item ID %?",
                                    fn_id_loc)); }
    };
    alt check map_node {
      ast_map::node_item(@{node: item_fn(_, _, body), _}, _) |
      ast_map::node_item(@{node: item_res(_, _, body, _, _, _), _}, _) |
      ast_map::node_method(@{body, _}, _, _) {
        handle_body(cx, body);
      }
      ast_map::node_ctor(_, _, ast_map::res_ctor(_, _, _), _) |
      ast_map::node_variant(_, _, _) {
        for uint::range(0u, n_tps) {|n| cx.uses[n] |= use_repr;}
      }
      ast_map::node_native_item(i@@{node: native_item_fn(_, _), _}, abi, _) {
        if abi == native_abi_rust_intrinsic {
            let flags = alt check i.ident {
              "visit_ty" { 3u }
              "size_of" |  "pref_align_of" | "min_align_of" |
              "init" |  "reinterpret_cast" { use_repr }
              "get_tydesc" | "needs_drop" { use_tydesc }
              "forget" | "addr_of" { 0u }
            };
            for uint::range(0u, n_tps) {|n| cx.uses[n] |= flags;}
        }
      }
      ast_map::node_ctor(_, _, ast_map::class_ctor(ctor, _), _){
        handle_body(cx, ctor.node.body);
      }
      ast_map::node_dtor(_, dtor, _, _){
        handle_body(cx, dtor.node.body);
      }

    }
    let uses = vec::from_mut(copy cx.uses);
    ccx.type_use_cache.insert(fn_id, uses);
    uses
}

fn type_needs(cx: ctx, use: uint, ty: ty::t) {
    let mut done = true;
    // Optimization -- don't descend type if all params already have this use
    for vec::each(cx.uses) {|u| if u & use != use { done = false } }
    if !done { type_needs_inner(cx, use, ty, @nil); }
}

fn type_needs_inner(cx: ctx, use: uint, ty: ty::t,
                    enums_seen: @list<def_id>) {
    ty::maybe_walk_ty(ty) {|ty|
        if ty::type_has_params(ty) {
            alt ty::get(ty).struct {
              ty::ty_fn(_) | ty::ty_ptr(_) | ty::ty_rptr(_, _) |
              ty::ty_box(_) | ty::ty_iface(_, _) { false }
              ty::ty_enum(did, substs) {
                if option::is_none(list::find(enums_seen, {|id| id == did})) {
                    let seen = @cons(did, enums_seen);
                    for vec::each(*ty::enum_variants(cx.ccx.tcx, did)) {|v|
                        for vec::each(v.args) {|aty|
                            let t = ty::subst(cx.ccx.tcx, substs, aty);
                            type_needs_inner(cx, use, t, seen);
                        }
                    }
                }
                false
              }
              ty::ty_param(n, _) {
                cx.uses[n] |= use;
                false
              }
              _ { true }
            }
        } else { false }
    }
}

fn node_type_needs(cx: ctx, use: uint, id: node_id) {
    type_needs(cx, use, ty::node_id_to_type(cx.ccx.tcx, id));
}

fn mark_for_expr(cx: ctx, e: @expr) {
    alt e.node {
      expr_vstore(_, _) |
      expr_vec(_, _) |
      expr_rec(_, _) | expr_tup(_) |
      expr_unary(box(_), _) | expr_unary(uniq(_), _) |
      expr_cast(_, _) | expr_binary(add, _, _) |
      expr_copy(_) | expr_move(_, _) {
        node_type_needs(cx, use_repr, e.id);
      }
      expr_binary(op, lhs, _) {
        alt op {
          eq | lt | le | ne | ge | gt {
            node_type_needs(cx, use_tydesc, lhs.id)
          }
          _ {}
        }
      }
      expr_path(_) {
        cx.ccx.tcx.node_type_substs.find(e.id).iter {|ts|
            let id = ast_util::def_id_of_def(cx.ccx.tcx.def_map.get(e.id));
            vec::iter2(type_uses_for(cx.ccx, id, ts.len()), ts) {|uses, subst|
                type_needs(cx, uses, subst)
            }
        }
      }
      expr_fn(*) | expr_fn_block(*) {
        alt ty::ty_fn_proto(ty::expr_ty(cx.ccx.tcx, e)) {
          proto_bare | proto_any | proto_uniq {}
          proto_box | proto_block {
            for vec::each(*freevars::get_freevars(cx.ccx.tcx, e.id)) {|fv|
                let node_id = ast_util::def_id_of_def(fv.def).node;
                node_type_needs(cx, use_repr, node_id);
            }
          }
        }
      }
      expr_assign(val, _) | expr_swap(val, _) | expr_assign_op(_, val, _) |
      expr_ret(some(val)) {
        node_type_needs(cx, use_repr, val.id);
      }
      expr_index(base, _) | expr_field(base, _, _) {
        // FIXME could be more careful and not count fields
        // after the chosen field (#2537)
        let base_ty = ty::node_id_to_type(cx.ccx.tcx, base.id);
        type_needs(cx, use_repr, ty::type_autoderef(cx.ccx.tcx, base_ty));

        option::iter(cx.ccx.maps.method_map.find(e.id)) {|mth|
            alt mth {
              typeck::method_static(did) {
                option::iter(cx.ccx.tcx.node_type_substs.find(e.id)) {|ts|
                    vec::iter2(type_uses_for(cx.ccx, did, ts.len()), ts)
                        {|uses, subst| type_needs(cx, uses, subst)}
                }
              }
              typeck::method_param(_, _, param, _) {
                cx.uses[param] |= use_tydesc;
              }
              typeck::method_iface(_, _) {}
            }
        }
      }
      expr_log(_, _, val) {
        node_type_needs(cx, use_tydesc, val.id);
      }
      expr_new(_, _, v) {
        node_type_needs(cx, use_repr, v.id);
      }
      expr_call(f, _, _) {
        vec::iter(ty::ty_fn_args(ty::node_id_to_type(cx.ccx.tcx, f.id))) {|a|
            alt a.mode {
              expl(by_move) | expl(by_copy) | expl(by_val) {
                type_needs(cx, use_repr, a.ty);
              }
              _ {}
            }
        }
      }
      expr_alt(_, _, _) | expr_block(_) | expr_if(_, _, _) |
      expr_while(_, _) | expr_fail(_) | expr_break | expr_cont |
      expr_unary(_, _) | expr_lit(_) | expr_assert(_) | expr_check(_, _) |
      expr_if_check(_, _, _) | expr_mac(_) | expr_addr_of(_, _) |
      expr_ret(_) | expr_loop(_) | expr_bind(_, _) | expr_loop_body(_) {}
    }
}

fn handle_body(cx: ctx, body: blk) {
    let v = visit::mk_vt(@{
        visit_expr: {|e, cx, v|
            visit::visit_expr(e, cx, v);
            mark_for_expr(cx, e);
        },
        visit_local: {|l, cx, v|
            visit::visit_local(l, cx, v);
            node_type_needs(cx, use_repr, l.node.id);
        },
        visit_pat: {|p, cx, v|
            visit::visit_pat(p, cx, v);
            node_type_needs(cx, use_repr, p.id);
        },
        visit_block: {|b, cx, v|
            visit::visit_block(b, cx, v);
            option::iter(b.node.expr) {|e|
                node_type_needs(cx, use_repr, e.id);
            }
        },
        visit_item: {|_i, _cx, _v|}
        with *visit::default_visitor()
    });
    v.visit_block(body, cx, v);
}
