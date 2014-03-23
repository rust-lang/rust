// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use middle::const_eval::{compare_const_vals, lookup_const_by_id};
use middle::const_eval::{eval_const_expr, const_val, const_bool, const_float};
use middle::pat_util::*;
use middle::ty::*;
use middle::ty;
use middle::typeck::MethodMap;
use util::nodemap::NodeSet;
use util::ppaux::ty_to_str;

use std::cmp;
use std::iter;
use std::vec;
use syntax::ast::*;
use syntax::ast_util::{unguarded_pat, walk_pat};
use syntax::codemap::{DUMMY_SP, Span};
use syntax::parse::token;
use syntax::visit;
use syntax::visit::{Visitor, FnKind};

struct MatchCheckCtxt<'a> {
    tcx: &'a ty::ctxt,
    method_map: MethodMap,
    moves_map: &'a NodeSet
}

impl<'a> Visitor<()> for MatchCheckCtxt<'a> {
    fn visit_expr(&mut self, ex: &Expr, _: ()) {
        check_expr(self, ex);
    }
    fn visit_local(&mut self, l: &Local, _: ()) {
        check_local(self, l);
    }
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block, s: Span, n: NodeId, _: ()) {
        check_fn(self, fk, fd, b, s, n);
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   method_map: MethodMap,
                   moves_map: &NodeSet,
                   krate: &Crate) {
    let mut cx = MatchCheckCtxt {
        tcx: tcx,
        method_map: method_map,
        moves_map: moves_map
    };

    visit::walk_crate(&mut cx, krate, ());

    tcx.sess.abort_if_errors();
}

fn check_expr(cx: &mut MatchCheckCtxt, ex: &Expr) {
    visit::walk_expr(cx, ex, ());
    match ex.node {
      ExprMatch(scrut, ref arms) => {
        // First, check legality of move bindings.
        for arm in arms.iter() {
            check_legality_of_move_bindings(cx,
                                            arm.guard.is_some(),
                                            arm.pats.as_slice());
        }

        check_arms(cx, arms.as_slice());
        /* Check for exhaustiveness */
         // Check for empty enum, because is_useful only works on inhabited
         // types.
       let pat_ty = node_id_to_type(cx.tcx, scrut.id);
       if (*arms).is_empty() {
           if !type_is_empty(cx.tcx, pat_ty) {
               // We know the type is inhabited, so this must be wrong
               cx.tcx.sess.span_err(ex.span, format!("non-exhaustive patterns: \
                            type {} is non-empty",
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

       let pats: Vec<@Pat> = arms.iter()
                               .filter_map(unguarded_pat)
                               .flat_map(|pats| pats.move_iter())
                               .collect();
       if pats.is_empty() {
           cx.tcx.sess.span_err(ex.span, "non-exhaustive patterns");
       } else {
           check_exhaustive(cx, ex.span, pats);
       }
     }
     _ => ()
    }
}

// Check for unreachable patterns
fn check_arms(cx: &MatchCheckCtxt, arms: &[Arm]) {
    let mut seen = Vec::new();
    for arm in arms.iter() {
        for pat in arm.pats.iter() {

            // Check that we do not match against a static NaN (#6804)
            let pat_matches_nan: |&Pat| -> bool = |p| {
                let opt_def = cx.tcx.def_map.borrow().find_copy(&p.id);
                match opt_def {
                    Some(DefStatic(did, false)) => {
                        let const_expr = lookup_const_by_id(cx.tcx, did).unwrap();
                        match eval_const_expr(cx.tcx, const_expr) {
                            const_float(f) if f.is_nan() => true,
                            _ => false
                        }
                    }
                    _ => false
                }
            };

            walk_pat(*pat, |p| {
                if pat_matches_nan(p) {
                    cx.tcx.sess.span_warn(p.span, "unmatchable NaN in pattern, \
                                                   use the is_nan method in a guard instead");
                }
                true
            });

            let v = vec!(*pat);
            match is_useful(cx, &seen, v.as_slice()) {
              not_useful => {
                cx.tcx.sess.span_err(pat.span, "unreachable pattern");
              }
              _ => ()
            }
            if arm.guard.is_none() { seen.push(v); }
        }
    }
}

fn raw_pat(p: @Pat) -> @Pat {
    match p.node {
      PatIdent(_, _, Some(s)) => { raw_pat(s) }
      _ => { p }
    }
}

fn check_exhaustive(cx: &MatchCheckCtxt, sp: Span, pats: Vec<@Pat> ) {
    assert!((!pats.is_empty()));
    let ext = match is_useful(cx, &pats.map(|p| vec!(*p)), [wild()]) {
        not_useful => {
            // This is good, wildcard pattern isn't reachable
            return;
        }
        useful_ => None,
        useful(ty, ref ctor) => {
            match ty::get(ty).sty {
                ty::ty_bool => {
                    match *ctor {
                        val(const_bool(true)) => Some(~"true"),
                        val(const_bool(false)) => Some(~"false"),
                        _ => None
                    }
                }
                ty::ty_enum(id, _) => {
                    let vid = match *ctor {
                        variant(id) => id,
                        _ => fail!("check_exhaustive: non-variant ctor"),
                    };
                    let variants = ty::enum_variants(cx.tcx, id);

                    match variants.iter().find(|v| v.id == vid) {
                        Some(v) => Some(token::get_ident(v.name).get().to_str()),
                        None => {
                            fail!("check_exhaustive: bad variant in ctor")
                        }
                    }
                }
                ty::ty_unboxed_vec(..) | ty::ty_vec(..) => {
                    match *ctor {
                        vec(n) => Some(format!("vectors of length {}", n)),
                        _ => None
                    }
                }
                _ => None
            }
        }
    };
    let msg = ~"non-exhaustive patterns" + match ext {
        Some(ref s) => format!(": {} not covered",  *s),
        None => ~""
    };
    cx.tcx.sess.span_err(sp, msg);
}

type matrix = Vec<Vec<@Pat> > ;

#[deriving(Clone)]
enum useful {
    useful(ty::t, ctor),
    useful_,
    not_useful,
}

#[deriving(Clone, Eq)]
enum ctor {
    single,
    variant(DefId),
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
fn is_useful(cx: &MatchCheckCtxt, m: &matrix, v: &[@Pat]) -> useful {
    if m.len() == 0u {
        return useful_;
    }
    if m.get(0).len() == 0u {
        return not_useful
    }
    let real_pat = match m.iter().find(|r| r.get(0).id != 0) {
      Some(r) => *r.get(0), None => v[0]
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
                  ref u => (*u).clone(),
                }
              }
              ty::ty_enum(eid, _) => {
                for va in (*ty::enum_variants(cx.tcx, eid)).iter() {
                    match is_useful_specialized(cx, m, v, variant(va.id),
                                                va.args.len(), left_ty) {
                      not_useful => (),
                      ref u => return (*u).clone(),
                    }
                }
                not_useful
              }
              ty::ty_vec(_, ty::vstore_fixed(n)) => {
                is_useful_specialized(cx, m, v, vec(n), n, left_ty)
              }
              ty::ty_unboxed_vec(..) | ty::ty_vec(..) => {
                let max_len = m.iter().rev().fold(0, |max_len, r| {
                  match r.get(0).node {
                    PatVec(ref before, _, ref after) => {
                      cmp::max(before.len() + after.len(), max_len)
                    }
                    _ => max_len
                  }
                });
                for n in iter::range(0u, max_len + 1) {
                  match is_useful_specialized(cx, m, v, vec(n), n, left_ty) {
                    not_useful => (),
                    ref u => return (*u).clone(),
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
                            &m.iter().filter_map(|r| {
                                default(cx, r.as_slice())
                            }).collect::<matrix>(),
                            v.tail()) {
              useful_ => useful(left_ty, (*ctor).clone()),
              ref u => (*u).clone(),
            }
          }
        }
      }
      Some(ref v0_ctor) => {
        let arity = ctor_arity(cx, v0_ctor, left_ty);
        is_useful_specialized(cx, m, v, (*v0_ctor).clone(), arity, left_ty)
      }
    }
}

fn is_useful_specialized(cx: &MatchCheckCtxt,
                             m: &matrix,
                             v: &[@Pat],
                             ctor: ctor,
                             arity: uint,
                             lty: ty::t)
                             -> useful {
    let ms = m.iter().filter_map(|r| {
        specialize(cx, r.as_slice(), &ctor, arity, lty)
    }).collect::<matrix>();
    let could_be_useful = is_useful(
        cx, &ms, specialize(cx, v, &ctor, arity, lty).unwrap().as_slice());
    match could_be_useful {
      useful_ => useful(lty, ctor),
      ref u => (*u).clone(),
    }
}

fn pat_ctor_id(cx: &MatchCheckCtxt, p: @Pat) -> Option<ctor> {
    let pat = raw_pat(p);
    match pat.node {
      PatWild | PatWildMulti => { None }
      PatIdent(_, _, _) | PatEnum(_, _) => {
        let opt_def = cx.tcx.def_map.borrow().find_copy(&pat.id);
        match opt_def {
          Some(DefVariant(_, id, _)) => Some(variant(id)),
          Some(DefStatic(did, false)) => {
            let const_expr = lookup_const_by_id(cx.tcx, did).unwrap();
            Some(val(eval_const_expr(cx.tcx, const_expr)))
          }
          _ => None
        }
      }
      PatLit(expr) => { Some(val(eval_const_expr(cx.tcx, expr))) }
      PatRange(lo, hi) => {
        Some(range(eval_const_expr(cx.tcx, lo), eval_const_expr(cx.tcx, hi)))
      }
      PatStruct(..) => {
        match cx.tcx.def_map.borrow().find(&pat.id) {
          Some(&DefVariant(_, id, _)) => Some(variant(id)),
          _ => Some(single)
        }
      }
      PatUniq(_) | PatTup(_) | PatRegion(..) => {
        Some(single)
      }
      PatVec(ref before, slice, ref after) => {
        match slice {
          Some(_) => None,
          None => Some(vec(before.len() + after.len()))
        }
      }
    }
}

fn is_wild(cx: &MatchCheckCtxt, p: @Pat) -> bool {
    let pat = raw_pat(p);
    match pat.node {
      PatWild | PatWildMulti => { true }
      PatIdent(_, _, _) => {
        match cx.tcx.def_map.borrow().find(&pat.id) {
          Some(&DefVariant(_, _, _)) | Some(&DefStatic(..)) => { false }
          _ => { true }
        }
      }
      _ => { false }
    }
}

fn missing_ctor(cx: &MatchCheckCtxt,
                    m: &matrix,
                    left_ty: ty::t)
                 -> Option<ctor> {
    match ty::get(left_ty).sty {
      ty::ty_box(_) | ty::ty_uniq(_) | ty::ty_rptr(..) | ty::ty_tup(_) |
      ty::ty_struct(..) => {
        for r in m.iter() {
            if !is_wild(cx, *r.get(0)) { return None; }
        }
        return Some(single);
      }
      ty::ty_enum(eid, _) => {
        let mut found = Vec::new();
        for r in m.iter() {
            let r = pat_ctor_id(cx, *r.get(0));
            for id in r.iter() {
                if !found.contains(id) {
                    found.push((*id).clone());
                }
            }
        }
        let variants = ty::enum_variants(cx.tcx, eid);
        if found.len() != (*variants).len() {
            for v in (*variants).iter() {
                if !found.iter().any(|x| x == &(variant(v.id))) {
                    return Some(variant(v.id));
                }
            }
            fail!();
        } else { None }
      }
      ty::ty_nil => None,
      ty::ty_bool => {
        let mut true_found = false;
        let mut false_found = false;
        for r in m.iter() {
            match pat_ctor_id(cx, *r.get(0)) {
              None => (),
              Some(val(const_bool(true))) => true_found = true,
              Some(val(const_bool(false))) => false_found = true,
              _ => fail!("impossible case")
            }
        }
        if true_found && false_found { None }
        else if true_found { Some(val(const_bool(false))) }
        else { Some(val(const_bool(true))) }
      }
      ty::ty_vec(_, ty::vstore_fixed(n)) => {
        let mut missing = true;
        let mut wrong = false;
        for r in m.iter() {
          match r.get(0).node {
            PatVec(ref before, ref slice, ref after) => {
              let count = before.len() + after.len();
              if (count < n && slice.is_none()) || count > n {
                wrong = true;
              }
              if count == n || (count < n && slice.is_some()) {
                missing = false;
              }
            }
            _ => {}
          }
        }
        match (wrong, missing) {
          (true, _) => Some(vec(n)), // should be compile-time error
          (_, true) => Some(vec(n)),
          _         => None
        }
      }
      ty::ty_unboxed_vec(..) | ty::ty_vec(..) => {

        // Find the lengths and slices of all vector patterns.
        let mut vec_pat_lens = m.iter().filter_map(|r| {
            match r.get(0).node {
                PatVec(ref before, ref slice, ref after) => {
                    Some((before.len() + after.len(), slice.is_some()))
                }
                _ => None
            }
        }).collect::<Vec<(uint, bool)> >();

        // Sort them by length such that for patterns of the same length,
        // those with a destructured slice come first.
        vec_pat_lens.sort_by(|&(len1, slice1), &(len2, slice2)| {
                    if len1 == len2 {
                        slice2.cmp(&slice1)
                    } else {
                        len1.cmp(&len2)
                    }
                });
        vec_pat_lens.dedup();

        let mut found_slice = false;
        let mut next = 0;
        let mut missing = None;
        for &(length, slice) in vec_pat_lens.iter() {
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

fn ctor_arity(cx: &MatchCheckCtxt, ctor: &ctor, ty: ty::t) -> uint {
    match ty::get(ty).sty {
      ty::ty_tup(ref fs) => fs.len(),
      ty::ty_box(_) | ty::ty_uniq(_) | ty::ty_rptr(..) => 1u,
      ty::ty_enum(eid, _) => {
          let id = match *ctor { variant(id) => id,
          _ => fail!("impossible case") };
        match ty::enum_variants(cx.tcx, eid).iter().find(|v| v.id == id ) {
            Some(v) => v.args.len(),
            None => fail!("impossible case")
        }
      }
      ty::ty_struct(cid, _) => ty::lookup_struct_fields(cx.tcx, cid).len(),
      ty::ty_unboxed_vec(..) | ty::ty_vec(..) => {
        match *ctor {
          vec(n) => n,
          _ => 0u
        }
      }
      _ => 0u
    }
}

fn wild() -> @Pat {
    @Pat {id: 0, node: PatWild, span: DUMMY_SP}
}

fn wild_multi() -> @Pat {
    @Pat {id: 0, node: PatWildMulti, span: DUMMY_SP}
}

fn specialize(cx: &MatchCheckCtxt,
                  r: &[@Pat],
                  ctor_id: &ctor,
                  arity: uint,
                  left_ty: ty::t)
               -> Option<Vec<@Pat> > {
    // Sad, but I can't get rid of this easily
    let r0 = (*raw_pat(r[0])).clone();
    match r0 {
        Pat{id: pat_id, node: n, span: pat_span} =>
            match n {
            PatWild => {
                Some(vec::append(Vec::from_elem(arity, wild()), r.tail()))
            }
            PatWildMulti => {
                Some(vec::append(Vec::from_elem(arity, wild_multi()),
                                    r.tail()))
            }
            PatIdent(_, _, _) => {
                let opt_def = cx.tcx.def_map.borrow().find_copy(&pat_id);
                match opt_def {
                    Some(DefVariant(_, id, _)) => {
                        if variant(id) == *ctor_id {
                            Some(Vec::from_slice(r.tail()))
                        } else {
                            None
                        }
                    }
                    Some(DefStatic(did, _)) => {
                        let const_expr =
                            lookup_const_by_id(cx.tcx, did).unwrap();
                        let e_v = eval_const_expr(cx.tcx, const_expr);
                        let match_ = match *ctor_id {
                            val(ref v) => {
                                match compare_const_vals(&e_v, v) {
                                    Some(val1) => (val1 == 0),
                                    None => {
                                        cx.tcx.sess.span_err(pat_span,
                                            "mismatched types between arms");
                                        false
                                    }
                                }
                            },
                            range(ref c_lo, ref c_hi) => {
                                let m1 = compare_const_vals(c_lo, &e_v);
                                let m2 = compare_const_vals(c_hi, &e_v);
                                match (m1, m2) {
                                    (Some(val1), Some(val2)) => {
                                        (val1 >= 0 && val2 <= 0)
                                    }
                                    _ => {
                                        cx.tcx.sess.span_err(pat_span,
                                            "mismatched types between ranges");
                                        false
                                    }
                                }
                            }
                            single => true,
                            _ => fail!("type error")
                        };
                        if match_ {
                            Some(Vec::from_slice(r.tail()))
                        } else {
                            None
                        }
                    }
                    _ => {
                        Some(
                            vec::append(
                                Vec::from_elem(arity, wild()),
                                r.tail()
                            )
                        )
                    }
                }
            }
            PatEnum(_, args) => {
                let def = cx.tcx.def_map.borrow().get_copy(&pat_id);
                match def {
                    DefStatic(did, _) => {
                        let const_expr =
                            lookup_const_by_id(cx.tcx, did).unwrap();
                        let e_v = eval_const_expr(cx.tcx, const_expr);
                        let match_ = match *ctor_id {
                            val(ref v) =>
                                match compare_const_vals(&e_v, v) {
                                    Some(val1) => (val1 == 0),
                                    None => {
                                        cx.tcx.sess.span_err(pat_span,
                                            "mismatched types between arms");
                                        false
                                    }
                                },
                            range(ref c_lo, ref c_hi) => {
                                let m1 = compare_const_vals(c_lo, &e_v);
                                let m2 = compare_const_vals(c_hi, &e_v);
                                match (m1, m2) {
                                    (Some(val1), Some(val2)) => (val1 >= 0 && val2 <= 0),
                                    _ => {
                                        cx.tcx.sess.span_err(pat_span,
                                            "mismatched types between ranges");
                                        false
                                    }
                                }
                            }
                            single => true,
                            _ => fail!("type error")
                        };
                        if match_ {
                            Some(Vec::from_slice(r.tail()))
                        } else {
                            None
                        }
                    }
                    DefVariant(_, id, _) if variant(id) == *ctor_id => {
                        let args = match args {
                            Some(args) => args.iter().map(|x| *x).collect(),
                            None => Vec::from_elem(arity, wild())
                        };
                        Some(vec::append(args, r.tail()))
                    }
                    DefVariant(_, _, _) => None,

                    DefFn(..) |
                    DefStruct(..) => {
                        let new_args;
                        match args {
                            Some(args) => {
                                new_args = args.iter().map(|x| *x).collect()
                            }
                            None => new_args = Vec::from_elem(arity, wild())
                        }
                        Some(vec::append(new_args, r.tail()))
                    }
                    _ => None
                }
            }
            PatStruct(_, ref pattern_fields, _) => {
                // Is this a struct or an enum variant?
                let def = cx.tcx.def_map.borrow().get_copy(&pat_id);
                match def {
                    DefVariant(_, variant_id, _) => {
                        if variant(variant_id) == *ctor_id {
                            let struct_fields = ty::lookup_struct_fields(cx.tcx, variant_id);
                            let args = struct_fields.map(|sf| {
                                match pattern_fields.iter().find(|f| f.ident.name == sf.name) {
                                    Some(f) => f.pat,
                                    _ => wild()
                                }
                            });
                            Some(vec::append(args, r.tail()))
                        } else {
                            None
                        }
                    }
                    _ => {
                        // Grab the class data that we care about.
                        let class_fields;
                        let class_id;
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
                                    format!("struct pattern resolved to {}, \
                                          not a struct",
                                         ty_to_str(cx.tcx, left_ty)));
                            }
                        }
                        let args = class_fields.iter().map(|class_field| {
                            match pattern_fields.iter().find(|f|
                                            f.ident.name == class_field.name) {
                                Some(f) => f.pat,
                                _ => wild()
                            }
                        }).collect();
                        Some(vec::append(args, r.tail()))
                    }
                }
            }
            PatTup(args) => {
                Some(vec::append(args.iter().map(|x| *x).collect(), r.tail()))
            }
            PatUniq(a) | PatRegion(a) => {
                Some(vec::append(vec!(a), r.tail()))
            }
            PatLit(expr) => {
                let e_v = eval_const_expr(cx.tcx, expr);
                let match_ = match *ctor_id {
                    val(ref v) => {
                        match compare_const_vals(&e_v, v) {
                            Some(val1) => val1 == 0,
                            None => {
                                cx.tcx.sess.span_err(pat_span,
                                    "mismatched types between arms");
                                false
                            }
                        }
                    },
                    range(ref c_lo, ref c_hi) => {
                        let m1 = compare_const_vals(c_lo, &e_v);
                        let m2 = compare_const_vals(c_hi, &e_v);
                        match (m1, m2) {
                            (Some(val1), Some(val2)) => (val1 >= 0 && val2 <= 0),
                            _ => {
                                cx.tcx.sess.span_err(pat_span,
                                    "mismatched types between ranges");
                                false
                            }
                        }
                    }
                    single => true,
                    _ => fail!("type error")
                };
                if match_ {
                    Some(Vec::from_slice(r.tail()))
                } else {
                    None
                }
            }
            PatRange(lo, hi) => {
                let (c_lo, c_hi) = match *ctor_id {
                    val(ref v) => ((*v).clone(), (*v).clone()),
                    range(ref lo, ref hi) => ((*lo).clone(), (*hi).clone()),
                    single => return Some(Vec::from_slice(r.tail())),
                    _ => fail!("type error")
                };
                let v_lo = eval_const_expr(cx.tcx, lo);
                let v_hi = eval_const_expr(cx.tcx, hi);

                let m1 = compare_const_vals(&c_lo, &v_lo);
                let m2 = compare_const_vals(&c_hi, &v_hi);
                match (m1, m2) {
                    (Some(val1), Some(val2)) if val1 >= 0 && val2 <= 0 => {
                        Some(Vec::from_slice(r.tail()))
                    },
                    (Some(_), Some(_)) => None,
                    _ => {
                        cx.tcx.sess.span_err(pat_span,
                            "mismatched types between ranges");
                        None
                    }
                }
            }
            PatVec(before, slice, after) => {
                match *ctor_id {
                    vec(_) => {
                        let num_elements = before.len() + after.len();
                        if num_elements < arity && slice.is_some() {
                            let mut result = Vec::new();
                            for pat in before.iter() {
                                result.push((*pat).clone());
                            }
                            for _ in iter::range(0, arity - num_elements) {
                                result.push(wild())
                            }
                            for pat in after.iter() {
                                result.push((*pat).clone());
                            }
                            for pat in r.tail().iter() {
                                result.push((*pat).clone());
                            }
                            Some(result)
                        } else if num_elements == arity {
                            let mut result = Vec::new();
                            for pat in before.iter() {
                                result.push((*pat).clone());
                            }
                            for pat in after.iter() {
                                result.push((*pat).clone());
                            }
                            for pat in r.tail().iter() {
                                result.push((*pat).clone());
                            }
                            Some(result)
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

fn default(cx: &MatchCheckCtxt, r: &[@Pat]) -> Option<Vec<@Pat> > {
    if is_wild(cx, r[0]) {
        Some(Vec::from_slice(r.tail()))
    } else {
        None
    }
}

fn check_local(cx: &mut MatchCheckCtxt, loc: &Local) {
    visit::walk_local(cx, loc, ());
    if is_refutable(cx, loc.pat) {
        cx.tcx.sess.span_err(loc.pat.span,
                             "refutable pattern in local binding");
    }

    // Check legality of move bindings.
    check_legality_of_move_bindings(cx, false, [ loc.pat ]);
}

fn check_fn(cx: &mut MatchCheckCtxt,
            kind: &FnKind,
            decl: &FnDecl,
            body: &Block,
            sp: Span,
            id: NodeId) {
    visit::walk_fn(cx, kind, decl, body, sp, id, ());
    for input in decl.inputs.iter() {
        if is_refutable(cx, input.pat) {
            cx.tcx.sess.span_err(input.pat.span,
                                 "refutable pattern in function argument");
        }
    }
}

fn is_refutable(cx: &MatchCheckCtxt, pat: &Pat) -> bool {
    let opt_def = cx.tcx.def_map.borrow().find_copy(&pat.id);
    match opt_def {
      Some(DefVariant(enum_id, _, _)) => {
        if ty::enum_variants(cx.tcx, enum_id).len() != 1u {
            return true;
        }
      }
      Some(DefStatic(..)) => return true,
      _ => ()
    }

    match pat.node {
      PatUniq(sub) | PatRegion(sub) | PatIdent(_, _, Some(sub)) => {
        is_refutable(cx, sub)
      }
      PatWild | PatWildMulti | PatIdent(_, _, None) => { false }
      PatLit(lit) => {
          match lit.node {
            ExprLit(lit) => {
                match lit.node {
                    LitNil => false,    // `()`
                    _ => true,
                }
            }
            _ => true,
          }
      }
      PatRange(_, _) => { true }
      PatStruct(_, ref fields, _) => {
        fields.iter().any(|f| is_refutable(cx, f.pat))
      }
      PatTup(ref elts) => {
        elts.iter().any(|elt| is_refutable(cx, *elt))
      }
      PatEnum(_, Some(ref args)) => {
        args.iter().any(|a| is_refutable(cx, *a))
      }
      PatEnum(_,_) => { false }
      PatVec(..) => { true }
    }
}

// Legality of move bindings checking

fn check_legality_of_move_bindings(cx: &MatchCheckCtxt,
                                       has_guard: bool,
                                       pats: &[@Pat]) {
    let tcx = cx.tcx;
    let def_map = tcx.def_map;
    let mut by_ref_span = None;
    let mut any_by_move = false;
    for pat in pats.iter() {
        pat_bindings(def_map, *pat, |bm, id, span, _path| {
            match bm {
                BindByRef(_) => {
                    by_ref_span = Some(span);
                }
                BindByValue(_) => {
                    if cx.moves_map.contains(&id) {
                        any_by_move = true;
                    }
                }
            }
        })
    }

    let check_move: |&Pat, Option<@Pat>| = |p, sub| {
        // check legality of moving out of the enum

        // x @ Foo(..) is legal, but x @ Foo(y) isn't.
        if sub.map_or(false, |p| pat_contains_bindings(def_map, p)) {
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
                by_ref_span.unwrap(),
                "by-ref binding occurs here");
        }
    };

    if !any_by_move { return; } // pointless micro-optimization
    for pat in pats.iter() {
        walk_pat(*pat, |p| {
            if pat_is_binding(def_map, p) {
                match p.node {
                    PatIdent(_, _, sub) => {
                        if cx.moves_map.contains(&p.id) {
                            check_move(p, sub);
                        }
                    }
                    _ => {
                        cx.tcx.sess.span_bug(
                            p.span,
                            format!("binding pattern {} is \
                                  not an identifier: {:?}",
                                 p.id, p.node));
                    }
                }
            }
            true
        });
    }
}
