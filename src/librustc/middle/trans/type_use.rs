// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

use middle::freevars;
use middle::trans::common::*;
use middle::trans::inline;
use middle::ty;
use middle::typeck;

use core::option::{Some, None};
use core::uint;
use core::vec;
use std::list::{List, Cons, Nil};
use std::list;
use syntax::ast;
use syntax::ast::*;
use syntax::ast_map;
use syntax::ast_util;
use syntax::visit;

pub type type_uses = uint; // Bitmask
pub static use_repr: uint = 1;   /* Dependency on size/alignment/mode and
                                     take/drop glue */
pub static use_tydesc: uint = 2; /* Takes the tydesc, or compares */

pub struct Context {
    ccx: @CrateContext,
    uses: @mut ~[type_uses]
}

pub fn type_uses_for(ccx: @CrateContext, fn_id: def_id, n_tps: uint)
    -> @~[type_uses] {
    match ccx.type_use_cache.find(&fn_id) {
      Some(uses) => return *uses,
      None => ()
    }

    let fn_id_loc = if fn_id.crate == local_crate {
        fn_id
    } else {
        inline::maybe_instantiate_inline(ccx, fn_id, true)
    };

    // Conservatively assume full use for recursive loops
    ccx.type_use_cache.insert(fn_id, @vec::from_elem(n_tps, 3u));

    let cx = Context {
        ccx: ccx,
        uses: @mut vec::from_elem(n_tps, 0)
    };
    match ty::get(ty::lookup_item_type(cx.ccx.tcx, fn_id).ty).sty {
        ty::ty_bare_fn(ty::BareFnTy {sig: ref sig, _}) |
        ty::ty_closure(ty::ClosureTy {sig: ref sig, _}) => {
            for vec::each(sig.inputs) |arg| {
                type_needs(cx, use_repr, arg.ty);
            }
        }
        _ => ()
    }

    if fn_id_loc.crate != local_crate {
        let Context { uses: @uses, _ } = cx;
        let uses = @uses; // mutability
        ccx.type_use_cache.insert(fn_id, uses);
        return uses;
    }
    let map_node = match ccx.tcx.items.find(&fn_id_loc.node) {
        Some(x) => (/*bad*/copy *x),
        None    => ccx.sess.bug(fmt!("type_uses_for: unbound item ID %?",
                                     fn_id_loc))
    };
    match map_node {
      ast_map::node_item(@ast::item { node: item_fn(_, _, _, _, ref body),
                                      _ }, _) |
      ast_map::node_method(@ast::method {body: ref body, _}, _, _) => {
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
      ast_map::node_foreign_item(i@@foreign_item { node: foreign_item_fn(*),
                                                   _ },
                                 abi,
                                 _,
                                 _) => {
        if abi.is_intrinsic() {
            let flags = match *cx.ccx.sess.str_of(i.ident) {
                ~"size_of"  | ~"pref_align_of" | ~"min_align_of" |
                ~"init"     | ~"transmute"     | ~"move_val"     |
                ~"move_val_init" => use_repr,

                ~"get_tydesc" | ~"needs_drop" => use_tydesc,

                ~"atomic_cxchg"    | ~"atomic_cxchg_acq"|
                ~"atomic_cxchg_rel"| ~"atomic_xchg"     |
                ~"atomic_xadd"     | ~"atomic_xsub"     |
                ~"atomic_xchg_acq" | ~"atomic_xadd_acq" |
                ~"atomic_xsub_acq" | ~"atomic_xchg_rel" |
                ~"atomic_xadd_rel" | ~"atomic_xsub_rel" => 0,

                ~"visit_tydesc"  | ~"forget" | ~"frame_address" |
                ~"morestack_addr" => 0,

                ~"memmove32" | ~"memmove64" => 0,

                ~"sqrtf32" | ~"sqrtf64" | ~"powif32" | ~"powif64" |
                ~"sinf32"  | ~"sinf64"  | ~"cosf32"  | ~"cosf64"  |
                ~"powf32"  | ~"powf64"  | ~"expf32"  | ~"expf64"  |
                ~"exp2f32" | ~"exp2f64" | ~"logf32"  | ~"logf64"  |
                ~"log10f32"| ~"log10f64"| ~"log2f32" | ~"log2f64" |
                ~"fmaf32"  | ~"fmaf64"  | ~"fabsf32" | ~"fabsf64" |
                ~"floorf32"| ~"floorf64"| ~"ceilf32" | ~"ceilf64" |
                ~"truncf32"| ~"truncf64" => 0,

                ~"ctpop8" | ~"ctpop16" | ~"ctpop32" | ~"ctpop64" => 0,

                ~"ctlz8" | ~"ctlz16" | ~"ctlz32" | ~"ctlz64" => 0,
                ~"cttz8" | ~"cttz16" | ~"cttz32" | ~"cttz64" => 0,

                ~"bswap16" | ~"bswap32" | ~"bswap64" => 0,

                // would be cool to make these an enum instead of strings!
                _ => fail!(~"unknown intrinsic in type_use")
            };
            for uint::range(0u, n_tps) |n| { cx.uses[n] |= flags;}
        }
      }
      ast_map::node_struct_ctor(*) => {
        // Similarly to node_variant, this monomorphized function just uses
        // the representations of all of its type parameters.
        for uint::range(0, n_tps) |n| { cx.uses[n] |= use_repr; }
      }
      _ => {
        ccx.tcx.sess.bug(fmt!("unknown node type in type_use: %s",
                              ast_map::node_id_to_str(
                                ccx.tcx.items,
                                fn_id_loc.node,
                                ccx.tcx.sess.parse_sess.interner)));
      }
    }
    let Context { uses: @uses, _ } = cx;
    let uses = @uses; // mutability
    ccx.type_use_cache.insert(fn_id, uses);
    uses
}

pub fn type_needs(cx: Context, use_: uint, ty: ty::t) {
    // Optimization -- don't descend type if all params already have this use
    let len = {
        let uses = &*cx.uses;
        uses.len()
    };
    for uint::range(0, len) |i| {
        if cx.uses[i] & use_ != use_ {
            type_needs_inner(cx, use_, ty, @Nil);
            return;
        }
    }
}

pub fn type_needs_inner(cx: Context,
                        use_: uint,
                        ty: ty::t,
                        enums_seen: @List<def_id>) {
    do ty::maybe_walk_ty(ty) |ty| {
        if ty::type_has_params(ty) {
            match ty::get(ty).sty {
                /*
                 This previously included ty_box -- that was wrong
                 because if we cast an @T to an trait (for example) and return
                 it, we depend on the drop glue for T (we have to write the
                 right tydesc into the result)
                 */
                ty::ty_closure(*) |
                ty::ty_bare_fn(*) |
                ty::ty_ptr(_) |
                ty::ty_rptr(_, _) |
                ty::ty_trait(_, _, _, _) => false,

              ty::ty_enum(did, ref substs) => {
                if list::find(enums_seen, |id| *id == did).is_none() {
                    let seen = @Cons(did, enums_seen);
                    for vec::each(*ty::enum_variants(cx.ccx.tcx, did)) |v| {
                        for vec::each(v.args) |aty| {
                            let t = ty::subst(cx.ccx.tcx, &(*substs), *aty);
                            type_needs_inner(cx, use_, t, seen);
                        }
                    }
                }
                false
              }
              ty::ty_param(p) => {
                cx.uses[p.idx] |= use_;
                false
              }
              _ => true
            }
        } else { false }
    }
}

pub fn node_type_needs(cx: Context, use_: uint, id: node_id) {
    type_needs(cx, use_, ty::node_id_to_type(cx.ccx.tcx, id));
}

pub fn mark_for_method_call(cx: Context, e_id: node_id, callee_id: node_id) {
    for cx.ccx.maps.method_map.find(&e_id).each |mth| {
        match mth.origin {
          typeck::method_static(did) => {
            for cx.ccx.tcx.node_type_substs.find(&callee_id).each |ts| {
                // FIXME(#5562): removing this copy causes a segfault
                //               before stage2
                let ts = /*bad*/ copy **ts;
                let type_uses = type_uses_for(cx.ccx, did, ts.len());
                for vec::each2(*type_uses, ts) |uses, subst| {
                    type_needs(cx, *uses, *subst)
                }
            }
          }
          typeck::method_param(typeck::method_param {
              param_num: param,
              _
          }) => {
            cx.uses[param] |= use_tydesc;
          }
          typeck::method_trait(*) | typeck::method_self(*)
              | typeck::method_super(*) => (),
        }
    }
}

pub fn mark_for_expr(cx: Context, e: @expr) {
    match e.node {
      expr_vstore(_, _) | expr_vec(_, _) | expr_struct(*) | expr_tup(_) |
      expr_unary(box(_), _) | expr_unary(uniq(_), _) |
      expr_binary(add, _, _) | expr_copy(_) | expr_repeat(*) => {
        node_type_needs(cx, use_repr, e.id);
      }
      expr_cast(base, _) => {
        let result_t = ty::node_id_to_type(cx.ccx.tcx, e.id);
        match ty::get(result_t).sty {
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
        for cx.ccx.tcx.node_type_substs.find(&e.id).each |ts| {
            // FIXME(#5562): removing this copy causes a segfault before stage2
            let ts = copy **ts;
            let id = ast_util::def_id_of_def(*cx.ccx.tcx.def_map.get(&e.id));
            let uses_for_ts = type_uses_for(cx.ccx, id, ts.len());
            for vec::each2(*uses_for_ts, ts) |uses, subst| {
                type_needs(cx, *uses, *subst)
            }
        }
      }
      expr_fn_block(*) => {
          match ty::ty_closure_sigil(ty::expr_ty(cx.ccx.tcx, e)) {
              ast::OwnedSigil => {}
              ast::BorrowedSigil | ast::ManagedSigil => {
                  for freevars::get_freevars(cx.ccx.tcx, e.id).each |fv| {
                      let node_id = ast_util::def_id_of_def(fv.def).node;
                      node_type_needs(cx, use_repr, node_id);
                  }
              }
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
        mark_for_method_call(cx, e.id, e.callee_id);
      }
      expr_log(_, val) => {
        node_type_needs(cx, use_tydesc, val.id);
      }
      expr_call(f, _, _) => {
          for vec::each(ty::ty_fn_args(ty::node_id_to_type(cx.ccx.tcx,
                                                           f.id))) |a| {
              type_needs(cx, use_repr, a.ty);
          }
      }
      expr_method_call(rcvr, _, _, _, _) => {
        let base_ty = ty::node_id_to_type(cx.ccx.tcx, rcvr.id);
        type_needs(cx, use_repr, ty::type_autoderef(cx.ccx.tcx, base_ty));

        for ty::ty_fn_args(ty::node_id_to_type(cx.ccx.tcx,
                                               e.callee_id)).each |a| {
            type_needs(cx, use_repr, a.ty);
        }
        mark_for_method_call(cx, e.id, e.callee_id);
      }

      expr_inline_asm(ref ia) => {
        for ia.inputs.each |&(_, in)| {
          node_type_needs(cx, use_repr, in.id);
        }
        for ia.outputs.each |&(_, out)| {
          node_type_needs(cx, use_repr, out.id);
        }
      }

      expr_paren(e) => mark_for_expr(cx, e),

      expr_match(*) | expr_block(_) | expr_if(*) | expr_while(*) |
      expr_break(_) | expr_again(_) | expr_unary(_, _) | expr_lit(_) |
      expr_mac(_) | expr_addr_of(_, _) | expr_ret(_) | expr_loop(_, _) |
      expr_loop_body(_) | expr_do_body(_) => ()
    }
}

pub fn handle_body(cx: Context, body: &blk) {
    let v = visit::mk_vt(@visit::Visitor {
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
            for b.node.expr.each |e| {
                node_type_needs(cx, use_repr, e.id);
            }
        },
        visit_item: |_i, _cx, _v| { },
        ..*visit::default_visitor()
    });
    (v.visit_block)(body, cx, v);
}

