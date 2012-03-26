// Determines the ways in which a generic function body is dependant
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
import driver::session::session;
import metadata::csearch;
import syntax::ast::*, syntax::ast_util, syntax::visit;
import common::*;

// FIXME distinguish between size/alignment and take/drop dependencies
type type_uses = uint; // Bitmask
const use_repr: uint = 1u; // Dependency on size/alignment and take/drop glue
const use_tydesc: uint = 2u; // Takes the tydesc, or compares

type ctx = {ccx: @crate_ctxt,
            uses: [mutable type_uses]};

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
        for arg in inputs {
            if arg.mode == expl(by_val) { type_needs(cx, use_repr, arg.ty); }
        }
      }
      _ {}
    }

    if fn_id_loc.crate != local_crate {
        let uses = vec::from_mut(cx.uses);
        ccx.type_use_cache.insert(fn_id, uses);
        ret uses;
    }
    alt check ccx.tcx.items.get(fn_id_loc.node) {
      ast_map::node_item(@{node: item_fn(_, _, body), _}, _) |
      ast_map::node_item(@{node: item_res(_, _, body, _, _), _}, _) |
      ast_map::node_method(@{body, _}, _, _) {
        handle_body(cx, body);
      }
      ast_map::node_ctor(@{node: item_res(_, _, _, _, _), _},_) |
      ast_map::node_variant(_, _, _) {
        uint::range(0u, n_tps) {|n| cx.uses[n] |= use_repr;}
      }
      ast_map::node_native_item(i@@{node: native_item_fn(_, _), _}, abi, _) {
        if abi == native_abi_rust_intrinsic {
            let flags = alt check i.ident {
              "size_of" | "align_of" | "init" |
              "reinterpret_cast" { use_repr }
              "get_tydesc" { use_tydesc }
              "forget" | "addr_of" { 0u }
            };
            uint::range(0u, n_tps) {|n| cx.uses[n] |= flags;}
        }
      }
      ast_map::node_ctor(@{node: item_class(_, _, ctor), _}, _) {
        ccx.sess.unimpl("type uses in class constructor");
      }
    }
    let uses = vec::from_mut(cx.uses);
    ccx.type_use_cache.insert(fn_id, uses);
    uses
}

fn type_needs(cx: ctx, use: uint, ty: ty::t) {
    let mut done = true;
    // Optimization -- don't descend type if all params already have this use
    for u in cx.uses { if u & use != use { done = false } }
    if !done { type_needs_inner(cx, use, ty); }
}

fn type_needs_inner(cx: ctx, use: uint, ty: ty::t) {
    ty::maybe_walk_ty(ty) {|ty|
        if ty::type_has_params(ty) {
            alt ty::get(ty).struct {
              ty::ty_fn(_) | ty::ty_ptr(_) | ty::ty_rptr(_, _) |
              ty::ty_box(_) | ty::ty_iface(_, _) { false }
              ty::ty_enum(did, tps) {
                for v in *ty::enum_variants(cx.ccx.tcx, did) {
                    for aty in v.args {
                        let t = ty::substitute_type_params(cx.ccx.tcx, tps,
                                                           aty);
                        type_needs_inner(cx, use, t);
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
      expr_vec(_, _) | expr_rec(_, _) | expr_tup(_) |
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
        option::may(cx.ccx.tcx.node_type_substs.find(e.id)) {|ts|
            let id = ast_util::def_id_of_def(cx.ccx.tcx.def_map.get(e.id));
            vec::iter2(type_uses_for(cx.ccx, id, ts.len()), ts) {|uses, subst|
                type_needs(cx, uses, subst)
            }
        }
      }
      expr_fn(_, _, _, _) | expr_fn_block(_, _) {
        alt ty::ty_fn_proto(ty::expr_ty(cx.ccx.tcx, e)) {
          proto_bare | proto_any | proto_uniq {}
          proto_box | proto_block {
            for fv in *freevars::get_freevars(cx.ccx.tcx, e.id) {
                let node_id = ast_util::def_id_of_def(fv.def).node;
                node_type_needs(cx, use_repr, node_id);
            }
          }
        }
      }
      expr_assign(val, _) | expr_swap(val, _) | expr_assign_op(_, val, _) |
      expr_ret(some(val)) | expr_be(val) {
        node_type_needs(cx, use_repr, val.id);
      }
      expr_index(base, _) | expr_field(base, _, _) {
        // FIXME could be more careful and not count fields
        // after the chosen field
        let base_ty = ty::node_id_to_type(cx.ccx.tcx, base.id);
        type_needs(cx, use_repr, ty::type_autoderef(cx.ccx.tcx, base_ty));
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
      expr_for(_, _, _) | expr_do_while(_, _) | expr_alt(_, _, _) |
      expr_block(_) | expr_if(_, _, _) | expr_while(_, _) |
      expr_fail(_) | expr_break | expr_cont | expr_unary(_, _) |
      expr_lit(_) | expr_assert(_) | expr_check(_, _) |
      expr_if_check(_, _, _) | expr_mac(_) | expr_addr_of(_, _) |
      expr_ret(_) | expr_loop(_) | expr_bind(_, _) {}
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
            option::may(b.node.expr) {|e|
                node_type_needs(cx, use_repr, e.id);
            }
        },
        visit_item: {|_i, _cx, _v|}
        with *visit::default_visitor()
    });
    v.visit_block(body, cx, v);
}
