// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::const_eval::{compare_const_vals, lookup_const_by_id};
use middle::const_eval::{eval_const_expr, const_val, const_bool};
use middle::pat_util::*;
use middle::ty::*;
use middle::ty;
use middle::typeck::method_map;
use middle::moves;
use util::ppaux::ty_to_str;

use std::sort;
use syntax::ast::*;
use syntax::ast_util::{unguarded_pat, walk_pat};
use syntax::codemap::{span, dummy_sp, spanned};
use syntax::visit;

pub struct MatchCheckCtxt {
    tcx: ty::ctxt,
    method_map: method_map,
    moves_map: moves::MovesMap
}

pub fn check_crate(tcx: ty::ctxt,
                   method_map: method_map,
                   moves_map: moves::MovesMap,
                   crate: @crate) {
    let cx = @MatchCheckCtxt {tcx: tcx,
                              method_map: method_map,
                              moves_map: moves_map};
    visit::visit_crate(crate, (), visit::mk_vt(@visit::Visitor {
        visit_expr: |a,b,c| check_expr(cx, a, b, c),
        visit_local: |a,b,c| check_local(cx, a, b, c),
        visit_fn: |kind, decl, body, sp, id, e, v|
            check_fn(cx, kind, decl, body, sp, id, e, v),
        .. *visit::default_visitor::<()>()
    }));
    tcx.sess.abort_if_errors();
}

pub fn expr_is_non_moving_lvalue(cx: @MatchCheckCtxt, expr: @expr) -> bool {
    if !ty::expr_is_lval(cx.tcx, cx.method_map, expr) {
        return false;
    }

    !cx.moves_map.contains(&expr.id)
}

pub fn check_expr(cx: @MatchCheckCtxt, ex: @expr, s: (), v: visit::vt<()>) {
    visit::visit_expr(ex, s, v);
    match ex.node {
      expr_match(scrut, ref arms) => {
        // First, check legality of move bindings.
        let is_non_moving_lvalue = expr_is_non_moving_lvalue(cx, ex);
        for arms.each |arm| {
            check_legality_of_move_bindings(cx,
                                            is_non_moving_lvalue,
                                            arm.guard.is_some(),
                                            arm.pats);
        }

        check_arms(cx, *arms);
        /* Check for exhaustiveness */
         // Check for empty enum, because is_useful only works on inhabited
         // types.
       let pat_ty = node_id_to_type(cx.tcx, scrut.id);
       if (*arms).is_empty() {
           if !type_is_empty(cx.tcx, pat_ty) {
               // We know the type is inhabited, so this must be wrong
               cx.tcx.sess.span_err(ex.span, fmt!("non-exhaustive patterns: \
                            type %s is non-empty",
                            ty_to_str(cx.tcx, pat_ty)));
           }
           // If the type *is* empty, it's vacuously exhaustive
           return;
       }
       match ty::get(pat_ty).sty {
          ty_enum(did, _) => {
              if (*enum_variants(cx.tcx, did)).is_empty() &&
                    (*arms).is_empty() {

               return;
            }
          }
          _ => { /* We assume only enum types can be uninhabited */ }
       }
       let arms = vec::concat(arms.filter_mapped(unguarded_pat));
       if arms.is_empty() {
           cx.tcx.sess.span_err(ex.span, "non-exhaustive patterns");
       } else {
           check_exhaustive(cx, ex.span, arms);
       }
     }
     _ => ()
    }
}

// Check for unreachable patterns
pub fn check_arms(cx: @MatchCheckCtxt, arms: &[arm]) {
    let mut seen = ~[];
    for arms.each |arm| {
        for arm.pats.each |pat| {
            let v = ~[*pat];
            match is_useful(cx, &seen, v) {
              not_useful => {
                cx.tcx.sess.span_err(pat.span, "unreachable pattern");
              }
              _ => ()
            }
            if arm.guard.is_none() { seen.push(v); }
        }
    }
}

pub fn raw_pat(p: @pat) -> @pat {
    match p.node {
      pat_ident(_, _, Some(s)) => { raw_pat(s) }
      _ => { p }
    }
}

pub fn check_exhaustive(cx: @MatchCheckCtxt, sp: span, pats: ~[@pat]) {
    assert!((!pats.is_empty()));
    let ext = match is_useful(cx, &pats.map(|p| ~[*p]), ~[wild()]) {
        not_useful => {
            // This is good, wildcard pattern isn't reachable
            return;
        }
        useful_ => None,
        useful(ty, ref ctor) => {
            match ty::get(ty).sty {
                ty::ty_bool => {
                    match (*ctor) {
                        val(const_bool(true)) => Some(@~"true"),
                        val(const_bool(false)) => Some(@~"false"),
                        _ => None
                    }
                }
                ty::ty_enum(id, _) => {
                    let vid = match *ctor {
                        variant(id) => id,
                        _ => fail!(~"check_exhaustive: non-variant ctor"),
                    };
                    let variants = ty::enum_variants(cx.tcx, id);

                    match variants.find(|v| v.id == vid) {
                        Some(v) => Some(cx.tcx.sess.str_of(v.name)),
                        None => {
                            fail!(~"check_exhaustive: bad variant in ctor")
                        }
                    }
                }
                ty::ty_unboxed_vec(*) | ty::ty_evec(*) => {
                    match *ctor {
                        vec(n) => Some(@fmt!("vectors of length %u", n)),
                        _ => None
                    }
                }
                _ => None
            }
        }
    };
    let msg = ~"non-exhaustive patterns" + match ext {
        Some(ref s) => ~": " + **s + ~" not covered",
        None => ~""
    };
    cx.tcx.sess.span_err(sp, msg);
}

pub type matrix = ~[~[@pat]];

pub enum useful { useful(ty::t, ctor), useful_, not_useful }

#[deriving(Eq)]
pub enum ctor {
    single,
    variant(def_id),
    val(const_val),
    range(const_val, const_val),
    vec(uint)
}

// Algorithm from http://moscova.inria.fr/~maranget/papers/warn/index.html
//
// Whether a vector `v` of patterns is 'useful' in relation to a set of such
// vectors `m` is defined as there being a set of inputs that will match `v`
// but not any of the sets in `m`.
//
// This is used both for reachability checking (if a pattern isn't useful in
// relation to preceding patterns, it is not reachable) and exhaustiveness
// checking (if a wildcard pattern is useful in relation to a matrix, the
// matrix isn't exhaustive).

// Note: is_useful doesn't work on empty types, as the paper notes.
// So it assumes that v is non-empty.
pub fn is_useful(cx: @MatchCheckCtxt, m: &matrix, v: &[@pat]) -> useful {
    if m.len() == 0u { return useful_; }
    if m[0].len() == 0u { return not_useful; }
    let real_pat = match m.find(|r| r[0].id != 0) {
      Some(r) => r[0], None => v[0]
    };
    let left_ty = if real_pat.id == 0 { ty::mk_nil() }
                  else { ty::node_id_to_type(cx.tcx, real_pat.id) };

    match pat_ctor_id(cx, v[0]) {
      None => {
        match missing_ctor(cx, m, left_ty) {
          None => {
            match ty::get(left_ty).sty {
              ty::ty_bool => {
                match is_useful_specialized(cx, m, v,
                                            val(const_bool(true)),
                                            0u, left_ty){
                  not_useful => {
                    is_useful_specialized(cx, m, v,
                                          val(const_bool(false)),
                                          0u, left_ty)
                  }
                  ref u => (/*bad*/copy *u)
                }
              }
              ty::ty_enum(eid, _) => {
                for (*ty::enum_variants(cx.tcx, eid)).each |va| {
                    match is_useful_specialized(cx, m, v, variant(va.id),
                                                va.args.len(), left_ty) {
                      not_useful => (),
                      ref u => return (/*bad*/copy *u)
                    }
                }
                not_useful
              }
              ty::ty_unboxed_vec(*) | ty::ty_evec(*) => {
                let max_len = do m.foldr(0) |r, max_len| {
                  match r[0].node {
                    pat_vec(ref before, _, ref after) => {
                      uint::max(before.len() + after.len(), max_len)
                    }
                    _ => max_len
                  }
                };
                for uint::range(0, max_len + 1) |n| {
                  match is_useful_specialized(cx, m, v, vec(n), n, left_ty) {
                    not_useful => (),
                    ref u => return (/*bad*/copy *u)
                  }
                }
                not_useful
              }
              _ => {
                let arity = ctor_arity(cx, &single, left_ty);
                is_useful_specialized(cx, m, v, single, arity, left_ty)
              }
            }
          }
          Some(ref ctor) => {
            match is_useful(cx,
                            &m.filter_mapped(|r| default(cx, *r)),
                            v.tail()) {
              useful_ => useful(left_ty, /*bad*/copy *ctor),
              ref u => (/*bad*/copy *u)
            }
          }
        }
      }
      Some(ref v0_ctor) => {
        let arity = ctor_arity(cx, v0_ctor, left_ty);
        is_useful_specialized(cx, m, v, /*bad*/copy *v0_ctor, arity, left_ty)
      }
    }
}

pub fn is_useful_specialized(cx: @MatchCheckCtxt,
                             m: &matrix,
                             v: &[@pat],
                             ctor: ctor,
                             arity: uint,
                             lty: ty::t)
                          -> useful {
    let ms = m.filter_mapped(|r| specialize(cx, *r, &ctor, arity, lty));
    let could_be_useful = is_useful(
        cx, &ms, specialize(cx, v, &ctor, arity, lty).get());
    match could_be_useful {
      useful_ => useful(lty, ctor),
      ref u => (/*bad*/copy *u)
    }
}

pub fn pat_ctor_id(cx: @MatchCheckCtxt, p: @pat) -> Option<ctor> {
    let pat = raw_pat(p);
    match pat.node {
      pat_wild => { None }
      pat_ident(_, _, _) | pat_enum(_, _) => {
        match cx.tcx.def_map.find(&pat.id) {
          Some(&def_variant(_, id)) => Some(variant(id)),
          Some(&def_const(did)) => {
            let const_expr = lookup_const_by_id(cx.tcx, did).get();
            Some(val(eval_const_expr(cx.tcx, const_expr)))
          }
          _ => None
        }
      }
      pat_lit(expr) => { Some(val(eval_const_expr(cx.tcx, expr))) }
      pat_range(lo, hi) => {
        Some(range(eval_const_expr(cx.tcx, lo), eval_const_expr(cx.tcx, hi)))
      }
      pat_struct(*) => {
        match cx.tcx.def_map.find(&pat.id) {
          Some(&def_variant(_, id)) => Some(variant(id)),
          _ => Some(single)
        }
      }
      pat_box(_) | pat_uniq(_) | pat_tup(_) | pat_region(*) => {
        Some(single)
      }
      pat_vec(ref before, slice, ref after) => {
        match slice {
          Some(_) => None,
          None => Some(vec(before.len() + after.len()))
        }
      }
    }
}

pub fn is_wild(cx: @MatchCheckCtxt, p: @pat) -> bool {
    let pat = raw_pat(p);
    match pat.node {
      pat_wild => { true }
      pat_ident(_, _, _) => {
        match cx.tcx.def_map.find(&pat.id) {
          Some(&def_variant(_, _)) | Some(&def_const(*)) => { false }
          _ => { true }
        }
      }
      _ => { false }
    }
}

pub fn missing_ctor(cx: @MatchCheckCtxt,
                    m: &matrix,
                    left_ty: ty::t)
                 -> Option<ctor> {
    match ty::get(left_ty).sty {
      ty::ty_box(_) | ty::ty_uniq(_) | ty::ty_rptr(*) | ty::ty_tup(_) |
      ty::ty_struct(*) => {
        for m.each |r| {
            if !is_wild(cx, r[0]) { return None; }
        }
        return Some(single);
      }
      ty::ty_enum(eid, _) => {
        let mut found = ~[];
        for m.each |r| {
            for pat_ctor_id(cx, r[0]).each |id| {
                if !vec::contains(found, id) {
                    found.push(/*bad*/copy *id);
                }
            }
        }
        let variants = ty::enum_variants(cx.tcx, eid);
        if found.len() != (*variants).len() {
            for vec::each(*variants) |v| {
                if !found.contains(&(variant(v.id))) {
                    return Some(variant(v.id));
                }
            }
            fail!();
        } else { None }
      }
      ty::ty_nil => None,
      ty::ty_bool => {
        let mut true_found = false, false_found = false;
        for m.each |r| {
            match pat_ctor_id(cx, r[0]) {
              None => (),
              Some(val(const_bool(true))) => true_found = true,
              Some(val(const_bool(false))) => false_found = true,
              _ => fail!(~"impossible case")
            }
        }
        if true_found && false_found { None }
        else if true_found { Some(val(const_bool(false))) }
        else { Some(val(const_bool(true))) }
      }
      ty::ty_unboxed_vec(*) | ty::ty_evec(*) => {

        // Find the lengths and slices of all vector patterns.
        let vec_pat_lens = do m.filter_mapped |r| {
            match r[0].node {
                pat_vec(ref before, ref slice, ref after) => {
                    Some((before.len() + after.len(), slice.is_some()))
                }
                _ => None
            }
        };

        // Sort them by length such that for patterns of the same length,
        // those with a destructured slice come first.
        let mut sorted_vec_lens = sort::merge_sort(vec_pat_lens,
            |&(len1, slice1), &(len2, slice2)| {
                if len1 == len2 {
                    slice1 > slice2
                } else {
                    len1 <= len2
                }
            }
        );
        vec::dedup(&mut sorted_vec_lens);

        let mut found_slice = false;
        let mut next = 0;
        let mut missing = None;
        for sorted_vec_lens.each |&(length, slice)| {
            if length != next {
                missing = Some(next);
                break;
            }
            if slice {
                found_slice = true;
                break;
            }
            next += 1;
        }

        // We found patterns of all lengths within <0, next), yet there was no
        // pattern with a slice - therefore, we report vec(next) as missing.
        if !found_slice {
            missing = Some(next);
        }
        match missing {
          Some(k) => Some(vec(k)),
          None => None
        }
      }
      _ => Some(single)
    }
}

pub fn ctor_arity(cx: @MatchCheckCtxt, ctor: &ctor, ty: ty::t) -> uint {
    match ty::get(ty).sty {
      ty::ty_tup(ref fs) => fs.len(),
      ty::ty_box(_) | ty::ty_uniq(_) | ty::ty_rptr(*) => 1u,
      ty::ty_enum(eid, _) => {
          let id = match *ctor { variant(id) => id,
          _ => fail!(~"impossible case") };
        match vec::find(*ty::enum_variants(cx.tcx, eid), |v| v.id == id ) {
            Some(v) => v.args.len(),
            None => fail!(~"impossible case")
        }
      }
      ty::ty_struct(cid, _) => ty::lookup_struct_fields(cx.tcx, cid).len(),
      ty::ty_unboxed_vec(*) | ty::ty_evec(*) => {
        match *ctor {
          vec(n) => n,
          _ => 0u
        }
      }
      _ => 0u
    }
}

pub fn wild() -> @pat {
    @pat {id: 0, node: pat_wild, span: dummy_sp()}
}

pub fn specialize(cx: @MatchCheckCtxt,
                  r: &[@pat],
                  ctor_id: &ctor,
                  arity: uint,
                  left_ty: ty::t)
               -> Option<~[@pat]> {
    // Sad, but I can't get rid of this easily
    let r0 = copy *raw_pat(r[0]);
    match r0 {
        pat{id: pat_id, node: n, span: pat_span} =>
            match n {
            pat_wild => {
                Some(vec::append(vec::from_elem(arity, wild()), r.tail()))
            }
            pat_ident(_, _, _) => {
                match cx.tcx.def_map.find(&pat_id) {
                    Some(&def_variant(_, id)) => {
                        if variant(id) == *ctor_id {
                            Some(vec::from_slice(r.tail()))
                        } else {
                            None
                        }
                    }
                    Some(&def_const(did)) => {
                        let const_expr =
                            lookup_const_by_id(cx.tcx, did).get();
                        let e_v = eval_const_expr(cx.tcx, const_expr);
                        let match_ = match *ctor_id {
                            val(ref v) => compare_const_vals(&e_v, v) == 0,
                            range(ref c_lo, ref c_hi) => {
                                compare_const_vals(c_lo, &e_v) >= 0 &&
                                    compare_const_vals(c_hi, &e_v) <= 0
                            }
                            single => true,
                            _ => fail!(~"type error")
                        };
                        if match_ {
                            Some(vec::from_slice(r.tail()))
                        } else {
                            None
                        }
                    }
                    _ => {
                        Some(
                            vec::append(
                                vec::from_elem(arity, wild()),
                                r.tail()
                            )
                        )
                    }
                }
            }
            pat_enum(_, args) => {
                match *cx.tcx.def_map.get(&pat_id) {
                    def_const(did) => {
                        let const_expr =
                            lookup_const_by_id(cx.tcx, did).get();
                        let e_v = eval_const_expr(cx.tcx, const_expr);
                        let match_ = match *ctor_id {
                            val(ref v) => compare_const_vals(&e_v, v) == 0,
                            range(ref c_lo, ref c_hi) => {
                                compare_const_vals(c_lo, &e_v) >= 0 &&
                                    compare_const_vals(c_hi, &e_v) <= 0
                            }
                            single => true,
                            _ => fail!(~"type error")
                        };
                        if match_ {
                            Some(vec::from_slice(r.tail()))
                        } else {
                            None
                        }
                    }
                    def_variant(_, id) if variant(id) == *ctor_id => {
                        let args = match args {
                            Some(args) => args,
                            None => vec::from_elem(arity, wild())
                        };
                        Some(vec::append(args, vec::from_slice(r.tail())))
                    }
                    def_variant(_, _) => None,

                    def_fn(*) |
                    def_struct(*) => {
                        // FIXME #4731: Is this right? --pcw
                        let new_args;
                        match args {
                            Some(args) => new_args = args,
                            None => new_args = vec::from_elem(arity, wild())
                        }
                        Some(vec::append(new_args, vec::from_slice(r.tail())))
                    }
                    _ => None
                }
            }
            pat_struct(_, ref flds, _) => {
                // Is this a struct or an enum variant?
                match *cx.tcx.def_map.get(&pat_id) {
                    def_variant(_, variant_id) => {
                        if variant(variant_id) == *ctor_id {
                            // FIXME #4731: Is this right? --pcw
                            let args = flds.map(|ty_field| {
                                match flds.find(|f|
                                                f.ident == ty_field.ident) {
                                    Some(f) => f.pat,
                                    _ => wild()
                                }
                            });
                            Some(vec::append(args, vec::from_slice(r.tail())))
                        } else {
                            None
                        }
                    }
                    _ => {
                        // Grab the class data that we care about.
                        let class_fields, class_id;
                        match ty::get(left_ty).sty {
                            ty::ty_struct(cid, _) => {
                                class_id = cid;
                                class_fields =
                                    ty::lookup_struct_fields(cx.tcx,
                                                             class_id);
                            }
                            _ => {
                                cx.tcx.sess.span_bug(
                                    pat_span,
                                    fmt!("struct pattern resolved to %s, \
                                          not a struct",
                                         ty_to_str(cx.tcx, left_ty)));
                            }
                        }
                        let args = vec::map(class_fields, |class_field| {
                            match flds.find(|f|
                                            f.ident == class_field.ident) {
                                Some(f) => f.pat,
                                _ => wild()
                            }
                        });
                        Some(vec::append(args, vec::from_slice(r.tail())))
                    }
                }
            }
            pat_tup(args) => Some(vec::append(args, r.tail())),
            pat_box(a) | pat_uniq(a) | pat_region(a) => {
                Some(vec::append(~[a], r.tail()))
            }
            pat_lit(expr) => {
                let e_v = eval_const_expr(cx.tcx, expr);
                let match_ = match *ctor_id {
                    val(ref v) => compare_const_vals(&e_v, v) == 0,
                    range(ref c_lo, ref c_hi) => {
                        compare_const_vals(c_lo, &e_v) >= 0 &&
                            compare_const_vals(c_hi, &e_v) <= 0
                    }
                    single => true,
                    _ => fail!(~"type error")
                };
                if match_ { Some(vec::from_slice(r.tail())) } else { None }
            }
            pat_range(lo, hi) => {
                let (c_lo, c_hi) = match *ctor_id {
                    val(ref v) => ((/*bad*/copy *v), (/*bad*/copy *v)),
                    range(ref lo, ref hi) =>
                        ((/*bad*/copy *lo), (/*bad*/copy *hi)),
                    single => return Some(vec::from_slice(r.tail())),
                    _ => fail!(~"type error")
                };
                let v_lo = eval_const_expr(cx.tcx, lo),
                v_hi = eval_const_expr(cx.tcx, hi);
                let match_ = compare_const_vals(&c_lo, &v_lo) >= 0 &&
                    compare_const_vals(&c_hi, &v_hi) <= 0;
          if match_ { Some(vec::from_slice(r.tail())) } else { None }
      }
            pat_vec(before, slice, after) => {
                match *ctor_id {
                    vec(_) => {
                        let num_elements = before.len() + after.len();
                        if num_elements < arity && slice.is_some() {
                            Some(vec::append(
                                vec::concat(&[
                                    before,
                                    vec::from_elem(
                                        arity - num_elements, wild()),
                                    after
                                ]),
                                r.tail()
                            ))
                        } else if num_elements == arity {
                            Some(vec::append(
                                vec::append(before, after),
                                r.tail()
                            ))
                        } else {
                            None
                        }
                    }
                    _ => None
                }
            }
        }
    }
}

pub fn default(cx: @MatchCheckCtxt, r: &[@pat]) -> Option<~[@pat]> {
    if is_wild(cx, r[0]) { Some(vec::from_slice(r.tail())) }
    else { None }
}

pub fn check_local(cx: @MatchCheckCtxt,
                   loc: @local,
                   s: (),
                   v: visit::vt<()>) {
    visit::visit_local(loc, s, v);
    if is_refutable(cx, loc.node.pat) {
        cx.tcx.sess.span_err(loc.node.pat.span,
                             "refutable pattern in local binding");
    }

    // Check legality of move bindings.
    let is_lvalue = match loc.node.init {
        Some(init) => expr_is_non_moving_lvalue(cx, init),
        None => true
    };
    check_legality_of_move_bindings(cx, is_lvalue, false, [ loc.node.pat ]);
}

pub fn check_fn(cx: @MatchCheckCtxt,
                kind: &visit::fn_kind,
                decl: &fn_decl,
                body: &blk,
                sp: span,
                id: node_id,
                s: (),
                v: visit::vt<()>) {
    visit::visit_fn(kind, decl, body, sp, id, s, v);
    for decl.inputs.each |input| {
        if is_refutable(cx, input.pat) {
            cx.tcx.sess.span_err(input.pat.span,
                                 "refutable pattern in function argument");
        }
    }
}

pub fn is_refutable(cx: @MatchCheckCtxt, pat: &pat) -> bool {
    match cx.tcx.def_map.find(&pat.id) {
      Some(&def_variant(enum_id, _)) => {
        if vec::len(*ty::enum_variants(cx.tcx, enum_id)) != 1u {
            return true;
        }
      }
      Some(&def_const(*)) => return true,
      _ => ()
    }

    match pat.node {
      pat_box(sub) | pat_uniq(sub) | pat_region(sub) |
      pat_ident(_, _, Some(sub)) => {
        is_refutable(cx, sub)
      }
      pat_wild | pat_ident(_, _, None) => { false }
      pat_lit(@expr {node: expr_lit(@spanned { node: lit_nil, _}), _}) => {
        // "()"
        false
      }
      pat_lit(_) | pat_range(_, _) => { true }
      pat_struct(_, ref fields, _) => {
        fields.any(|f| is_refutable(cx, f.pat))
      }
      pat_tup(ref elts) => {
        elts.any(|elt| is_refutable(cx, *elt))
      }
      pat_enum(_, Some(ref args)) => {
        args.any(|a| is_refutable(cx, *a))
      }
      pat_enum(_,_) => { false }
      pat_vec(*) => { true }
    }
}

// Legality of move bindings checking

pub fn check_legality_of_move_bindings(cx: @MatchCheckCtxt,
                                       is_lvalue: bool,
                                       has_guard: bool,
                                       pats: &[@pat]) {
    let tcx = cx.tcx;
    let def_map = tcx.def_map;
    let mut by_ref_span = None;
    let mut any_by_move = false;
    for pats.each |pat| {
        do pat_bindings(def_map, *pat) |bm, id, span, _path| {
            match bm {
                bind_by_copy => {}
                bind_by_ref(_) => {
                    by_ref_span = Some(span);
                }
                bind_infer => {
                    if cx.moves_map.contains(&id) {
                        any_by_move = true;
                    }
                }
            }
        }
    }

    let check_move: &fn(@pat, Option<@pat>) = |p, sub| {
        // check legality of moving out of the enum
        if sub.is_some() {
            tcx.sess.span_err(
                p.span,
                "cannot bind by-move with sub-bindings");
        } else if has_guard {
            tcx.sess.span_err(
                p.span,
                "cannot bind by-move into a pattern guard");
        } else if by_ref_span.is_some() {
            tcx.sess.span_err(
                p.span,
                "cannot bind by-move and by-ref \
                 in the same pattern");
            tcx.sess.span_note(
                by_ref_span.get(),
                "by-ref binding occurs here");
        } else if is_lvalue {
            tcx.sess.span_err(
                p.span,
                "cannot bind by-move when \
                 matching an lvalue");
        }
    };

    if !any_by_move { return; } // pointless micro-optimization
    for pats.each |pat| {
        do walk_pat(*pat) |p| {
            if pat_is_binding(def_map, p) {
                match p.node {
                    pat_ident(_, _, sub) => {
                        if cx.moves_map.contains(&p.id) {
                            check_move(p, sub);
                        }
                    }
                    _ => {
                        cx.tcx.sess.span_bug(
                            p.span,
                            fmt!("Binding pattern %d is \
                                  not an identifier: %?",
                                 p.id, p.node));
                    }
                }
            }
        }

        // Now check to ensure that any move binding is not behind an @ or &.
        // This is always illegal.
        let vt = visit::mk_vt(@visit::Visitor {
            visit_pat: |pat, behind_bad_pointer: bool, v| {
                match pat.node {
                    pat_ident(_, _, sub) => {
                        debug!("(check legality of move) checking pat \
                                ident with behind_bad_pointer %?",
                                behind_bad_pointer);

                        if behind_bad_pointer &&
                            cx.moves_map.contains(&pat.id)
                        {
                            cx.tcx.sess.span_err(
                                pat.span,
                                "by-move pattern \
                                 bindings may not occur \
                                 behind @ or & bindings");
                        }

                        match sub {
                            None => {}
                            Some(subpat) => {
                                (v.visit_pat)(subpat, behind_bad_pointer, v);
                            }
                        }
                    }

                    pat_box(subpat) | pat_region(subpat) => {
                        (v.visit_pat)(subpat, true, v);
                    }

                    _ => visit::visit_pat(pat, behind_bad_pointer, v)
                }
            },
            .. *visit::default_visitor::<bool>()
        });
        (vt.visit_pat)(*pat, false, vt);
    }
}
