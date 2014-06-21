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

use middle::const_eval::{compare_const_vals, const_bool, const_float, const_val};
use middle::const_eval::{eval_const_expr, lookup_const_by_id};
use middle::def::*;
use middle::pat_util::*;
use middle::ty::*;
use middle::ty;

use std::gc::{Gc, GC};
use std::iter;
use syntax::ast::*;
use syntax::ast_util::{is_unguarded, walk_pat};
use syntax::codemap::{Span, Spanned, DUMMY_SP};
use syntax::owned_slice::OwnedSlice;
use syntax::print::pprust::pat_to_str;
use syntax::visit;
use syntax::visit::{Visitor, FnKind};
use util::ppaux::ty_to_str;

type Matrix = Vec<Vec<Gc<Pat>>>;

#[deriving(Clone)]
enum Usefulness {
    Useful(Vec<Gc<Pat>>),
    NotUseful
}

enum WitnessPreference {
    ConstructWitness,
    LeaveOutWitness
}

impl Usefulness {
    fn useful(self) -> Option<Vec<Gc<Pat>>> {
        match self {
            Useful(pats) => Some(pats),
            _ => None
        }
    }
}

fn def_to_path(tcx: &ty::ctxt, id: DefId) -> Path {
    ty::with_path(tcx, id, |mut path| Path {
        global: false,
        segments: path.last().map(|elem| PathSegment {
            identifier: Ident::new(elem.name()),
            lifetimes: vec!(),
            types: OwnedSlice::empty()
        }).move_iter().collect(),
        span: DUMMY_SP,
    })
}

struct MatchCheckCtxt<'a> {
    tcx: &'a ty::ctxt,
}

impl<'a> Visitor<()> for MatchCheckCtxt<'a> {
    fn visit_expr(&mut self, ex: &Expr, _: ()) {
        check_expr(self, ex);
    }
    fn visit_local(&mut self, l: &Local, _: ()) {
        check_local(self, l);
    }
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block, s: Span, _: NodeId, _: ()) {
        check_fn(self, fk, fd, b, s);
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   krate: &Crate) {
    let mut cx = MatchCheckCtxt {
        tcx: tcx,
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

            // Second, check for unreachable arms.
            check_arms(cx, arms.as_slice());

            // Finally, check if the whole match expression is exhaustive.
            // Check for empty enum, because is_useful only works on inhabited types.
            let pat_ty = node_id_to_type(cx.tcx, scrut.id);
            if (*arms).is_empty() {
               if !type_is_empty(cx.tcx, pat_ty) {
                   // We know the type is inhabited, so this must be wrong
                   cx.tcx.sess.span_err(ex.span, format!("non-exhaustive patterns: \
                                type {} is non-empty",
                                ty_to_str(cx.tcx, pat_ty)).as_slice());
               }
               // If the type *is* empty, it's vacuously exhaustive
               return;
            }
            let m: Matrix = arms
                .iter()
                .filter(|&arm| is_unguarded(arm))
                .flat_map(|arm| arm.pats.iter())
                .map(|pat| vec!(pat.clone()))
                .collect();
            check_exhaustive(cx, ex.span, &m);
        },
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
                        match eval_const_expr(cx.tcx, &*const_expr) {
                            const_float(f) if f.is_nan() => true,
                            _ => false
                        }
                    }
                    _ => false
                }
            };

            walk_pat(&**pat, |p| {
                if pat_matches_nan(p) {
                    cx.tcx.sess.span_warn(p.span, "unmatchable NaN in pattern, \
                                                   use the is_nan method in a guard instead");
                }
                true
            });

            let v = vec!(*pat);
            match is_useful(cx, &seen, v.as_slice(), LeaveOutWitness) {
                NotUseful => cx.tcx.sess.span_err(pat.span, "unreachable pattern"),
                _ => ()
            }
            if arm.guard.is_none() { seen.push(v); }
        }
    }
}

fn raw_pat(p: Gc<Pat>) -> Gc<Pat> {
    match p.node {
        PatIdent(_, _, Some(s)) => { raw_pat(s) }
        _ => { p }
    }
}

fn check_exhaustive(cx: &MatchCheckCtxt, sp: Span, m: &Matrix) {
    match is_useful(cx, m, [wild()], ConstructWitness) {
        NotUseful => {
            // This is good, wildcard pattern isn't reachable
            return;
        }
        Useful(pats) => {
            let witness = match pats.as_slice() {
                [witness] => witness,
                [] => wild(),
                _ => unreachable!()
            };
            let msg = format!("non-exhaustive patterns: `{0}` not covered", pat_to_str(&*witness));
            cx.tcx.sess.span_err(sp, msg.as_slice());
        }
    }
}

#[deriving(Clone, PartialEq)]
enum ctor {
    single,
    variant(DefId /* variant */, bool /* is_structure */),
    val(const_val),
    range(const_val, const_val),
    vec(uint)
}

fn const_val_to_expr(value: &const_val) -> Gc<Expr> {
    let node = match value {
        &const_bool(b) => LitBool(b),
        _ => unreachable!()
    };
    box(GC) Expr {
        id: 0,
        node: ExprLit(box(GC) Spanned { node: node, span: DUMMY_SP }),
        span: DUMMY_SP
    }
}

fn construct_witness(cx: &MatchCheckCtxt, ctor: &ctor, pats: Vec<Gc<Pat>>, lty: ty::t) -> Gc<Pat> {
    let pat = match ty::get(lty).sty {
        ty::ty_tup(_) => PatTup(pats),

        ty::ty_enum(cid, _) | ty::ty_struct(cid, _)  => {
            let (vid, is_structure) = match ctor {
                &variant(vid, is_structure) => (vid, is_structure),
                _ => (cid, true)
            };
            if is_structure {
                let fields = ty::lookup_struct_fields(cx.tcx, vid);
                let field_pats = fields.move_iter()
                    .zip(pats.iter())
                    .map(|(field, pat)| FieldPat {
                        ident: Ident::new(field.name),
                        pat: pat.clone()
                    }).collect();
                PatStruct(def_to_path(cx.tcx, vid), field_pats, false)
            } else {
                PatEnum(def_to_path(cx.tcx, vid), Some(pats))
            }
        },

        ty::ty_rptr(_, ty::mt { ty: ty, .. }) => {
            match ty::get(ty).sty {
                ty::ty_vec(_, None) => match ctor {
                    &vec(_) => PatVec(pats, None, vec!()),
                    _ => unreachable!()
                },
                ty::ty_str => PatWild,
                _ => {
                    assert_eq!(pats.len(), 1);
                    PatRegion(pats.get(0).clone())
                }
            }
        },

        ty::ty_box(_) => {
            assert_eq!(pats.len(), 1);
            PatBox(pats.get(0).clone())
        },

        _ => {
            match ctor {
                &vec(_) => PatVec(pats, None, vec!()),
                &val(ref v) => PatLit(const_val_to_expr(v)),
                _ => PatWild
            }
        }
    };

    box(GC) Pat {
        id: 0,
        node: pat,
        span: DUMMY_SP
    }
}

fn missing_constructor(cx: &MatchCheckCtxt, m: &Matrix, left_ty: ty::t) -> Option<ctor> {
    let used_constructors: Vec<ctor> = m.iter()
        .filter_map(|r| pat_ctor_id(cx, left_ty, *r.get(0)))
        .collect();

    all_constructors(cx, m, left_ty)
        .move_iter()
        .find(|c| !used_constructors.contains(c))
}

fn all_constructors(cx: &MatchCheckCtxt, m: &Matrix, left_ty: ty::t) -> Vec<ctor> {
    fn vec_constructors(m: &Matrix) -> Vec<ctor> {
        let max_vec_len = m.iter().map(|r| match r.get(0).node {
            PatVec(ref before, _, ref after) => before.len() + after.len(),
            _ => 0u
        }).max().unwrap_or(0u);
        let contains_slice = m.iter().any(|r| match r.get(0).node {
            PatVec(_, ref slice, _) => slice.is_some(),
            _ => false
        });
        let lengths = iter::range_inclusive(0u, if contains_slice {
            max_vec_len
        } else {
            max_vec_len + 1
        });
        lengths.map(|len| vec(len)).collect()
    }

    match ty::get(left_ty).sty {
        ty::ty_bool =>
            [true, false].iter().map(|b| val(const_bool(*b))).collect(),

        ty::ty_rptr(_, ty::mt { ty: ty, .. }) => match ty::get(ty).sty {
            ty::ty_vec(_, None) => vec_constructors(m),
            _ => vec!(single)
        },

        ty::ty_enum(eid, _) =>
            ty::enum_variants(cx.tcx, eid)
                .iter()
                .map(|va| variant(va.id, va.arg_names.is_some()))
                .collect(),

        ty::ty_vec(_, None) =>
            vec_constructors(m),

        ty::ty_vec(_, Some(n)) =>
            vec!(vec(n)),

        ty::ty_nil if !m.iter().all(|r| is_wild(cx, *r.get(0))) =>
            vec!(),

        _ =>
            vec!(single)
    }
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
fn is_useful(cx: &MatchCheckCtxt, m: &Matrix, v: &[Gc<Pat>],
             witness: WitnessPreference) -> Usefulness {
    if m.len() == 0u {
        return Useful(vec!());
    }
    if m.get(0).len() == 0u {
        return NotUseful;
    }
    let real_pat = match m.iter().find(|r| r.get(0).id != 0) {
        Some(r) => {
            match r.get(0).node {
                // An arm of the form `ref x @ sub_pat` has type
                // `sub_pat`, not `&sub_pat` as `x` itself does.
                PatIdent(BindByRef(_), _, Some(sub)) => sub,
                _ => *r.get(0)
            }
        }
        None if v.len() == 0 => return NotUseful,
        None => v[0]
    };
    let left_ty = if real_pat.id == 0 {
        ty::mk_nil()
    } else {
        ty::pat_ty(cx.tcx, &*real_pat)
    };

    match pat_ctor_id(cx, left_ty, v[0]) {
        None => match missing_constructor(cx, m, left_ty) {
            None => {
                all_constructors(cx, m, left_ty).move_iter().filter_map(|c| {
                    is_useful_specialized(cx, m, v, c.clone(),
                                          left_ty, witness).useful().map(|pats| {
                        Useful(match witness {
                            ConstructWitness => {
                                let arity = constructor_arity(cx, &c, left_ty);
                                let subpats = {
                                    let pat_slice = pats.as_slice();
                                    Vec::from_fn(arity, |i| {
                                        pat_slice.get(i).map(|p| p.clone())
                                            .unwrap_or_else(|| wild())
                                    })
                                };
                                let mut result = vec!(construct_witness(cx, &c, subpats, left_ty));
                                result.extend(pats.move_iter().skip(arity));
                                result
                            }
                            LeaveOutWitness => vec!()
                        })
                    })
                }).nth(0).unwrap_or(NotUseful)
            },

            Some(ctor) => {
                let matrix = m.iter().filter_map(|r| default(cx, r.as_slice())).collect();
                match is_useful(cx, &matrix, v.tail(), witness) {
                    Useful(pats) => Useful(match witness {
                        ConstructWitness => {
                            let arity = constructor_arity(cx, &ctor, left_ty);
                            let wild_pats = Vec::from_elem(arity, wild());
                            let enum_pat = construct_witness(cx, &ctor, wild_pats, left_ty);
                            (vec!(enum_pat)).append(pats.as_slice())
                        }
                        LeaveOutWitness => vec!()
                    }),
                    result => result
                }
            }
        },

        Some(v0_ctor) => is_useful_specialized(cx, m, v, v0_ctor, left_ty, witness)
    }
}

fn is_useful_specialized(cx: &MatchCheckCtxt, m: &Matrix, v: &[Gc<Pat>],
                         ctor: ctor, lty: ty::t, witness: WitnessPreference) -> Usefulness {
    let arity = constructor_arity(cx, &ctor, lty);
    let matrix = m.iter().filter_map(|r| {
        specialize(cx, r.as_slice(), &ctor, arity)
    }).collect();
    match specialize(cx, v, &ctor, arity) {
        Some(v) => is_useful(cx, &matrix, v.as_slice(), witness),
        None => NotUseful
    }
}

fn pat_ctor_id(cx: &MatchCheckCtxt, left_ty: ty::t, p: Gc<Pat>) -> Option<ctor> {
    let pat = raw_pat(p);
    match pat.node {
        PatIdent(..) =>
            match cx.tcx.def_map.borrow().find(&pat.id) {
                Some(&DefStatic(did, false)) => {
                    let const_expr = lookup_const_by_id(cx.tcx, did).unwrap();
                    Some(val(eval_const_expr(cx.tcx, &*const_expr)))
                },
                Some(&DefVariant(_, id, is_structure)) => Some(variant(id, is_structure)),
                _ => None
            },
        PatEnum(..) =>
            match cx.tcx.def_map.borrow().find(&pat.id) {
                Some(&DefStatic(did, false)) => {
                    let const_expr = lookup_const_by_id(cx.tcx, did).unwrap();
                    Some(val(eval_const_expr(cx.tcx, &*const_expr)))
                },
                Some(&DefVariant(_, id, is_structure)) => Some(variant(id, is_structure)),
                _ => Some(single)
            },
        PatStruct(..) =>
            match cx.tcx.def_map.borrow().find(&pat.id) {
                Some(&DefVariant(_, id, is_structure)) => Some(variant(id, is_structure)),
                _ => Some(single)
            },
        PatLit(expr) =>
            Some(val(eval_const_expr(cx.tcx, &*expr))),
        PatRange(lo, hi) =>
            Some(range(eval_const_expr(cx.tcx, &*lo), eval_const_expr(cx.tcx, &*hi))),
        PatVec(ref before, _, ref after) => match ty::get(left_ty).sty {
            ty::ty_vec(_, Some(n)) =>
                Some(vec(n)),
            _ =>
                Some(vec(before.len() + after.len()))
        },
        PatBox(_) | PatTup(_) | PatRegion(..) =>
            Some(single),
        PatWild | PatWildMulti =>
            None,
        PatMac(_) =>
            cx.tcx.sess.bug("unexpanded macro")
    }
}

fn is_wild(cx: &MatchCheckCtxt, p: Gc<Pat>) -> bool {
    let pat = raw_pat(p);
    match pat.node {
        PatWild | PatWildMulti => true,
        PatIdent(_, _, _) => {
            match cx.tcx.def_map.borrow().find(&pat.id) {
                Some(&DefVariant(_, _, _)) | Some(&DefStatic(..)) => false,
                _ => true
            }
        }
        _ => false
    }
}

fn constructor_arity(cx: &MatchCheckCtxt, ctor: &ctor, ty: ty::t) -> uint {
    match ty::get(ty).sty {
        ty::ty_tup(ref fs) => fs.len(),
        ty::ty_box(_) | ty::ty_uniq(_) => 1u,
        ty::ty_rptr(_, ty::mt { ty: ty, .. }) => match ty::get(ty).sty {
            ty::ty_vec(_, None) => match *ctor {
                vec(n) => n,
                _ => 0u
            },
            ty::ty_str => 0u,
            _ => 1u
        },
        ty::ty_enum(eid, _) => {
            match *ctor {
                variant(id, _) => enum_variant_with_id(cx.tcx, eid, id).args.len(),
                _ => unreachable!()
            }
        }
        ty::ty_struct(cid, _) => ty::lookup_struct_fields(cx.tcx, cid).len(),
        ty::ty_vec(_, _) => match *ctor {
            vec(n) => n,
            _ => 0u
        },
        _ => 0u
    }
}

fn wild() -> Gc<Pat> {
    box(GC) Pat {id: 0, node: PatWild, span: DUMMY_SP}
}

fn range_covered_by_constructor(ctor_id: &ctor, from: &const_val, to: &const_val) -> Option<bool> {
    let (c_from, c_to) = match *ctor_id {
        val(ref value)          => (value, value),
        range(ref from, ref to) => (from, to),
        single                  => return Some(true),
        _                       => unreachable!()
    };
    let cmp_from = compare_const_vals(c_from, from);
    let cmp_to = compare_const_vals(c_to, to);
    match (cmp_from, cmp_to) {
        (Some(val1), Some(val2)) => Some(val1 >= 0 && val2 <= 0),
        _ => None
    }
}

fn specialize(cx: &MatchCheckCtxt, r: &[Gc<Pat>],
              ctor_id: &ctor, arity: uint) -> Option<Vec<Gc<Pat>>> {
    let &Pat {
        id: ref pat_id, node: ref n, span: ref pat_span
    } = &(*raw_pat(r[0]));
    let head: Option<Vec<Gc<Pat>>> = match n {
        &PatWild => {
            Some(Vec::from_elem(arity, wild()))
        }
        &PatWildMulti => {
            Some(Vec::from_elem(arity, wild()))
        }
        &PatIdent(_, _, _) => {
            let opt_def = cx.tcx.def_map.borrow().find_copy(pat_id);
            match opt_def {
                Some(DefVariant(_, id, _)) => match *ctor_id {
                    variant(vid, _) if vid == id => Some(vec!()),
                    _ => None
                },
                Some(DefStatic(did, _)) => {
                    let const_expr = lookup_const_by_id(cx.tcx, did).unwrap();
                    let e_v = eval_const_expr(cx.tcx, &*const_expr);
                    match range_covered_by_constructor(ctor_id, &e_v, &e_v) {
                        Some(true) => Some(vec!()),
                        Some(false) => None,
                        None => {
                            cx.tcx.sess.span_err(*pat_span, "mismatched types between arms");
                            None
                        }
                    }
                }
                _ => {
                    Some(Vec::from_elem(arity, wild()))
                }
            }
        }
        &PatEnum(_, ref args) => {
            let def = cx.tcx.def_map.borrow().get_copy(pat_id);
            match def {
                DefStatic(did, _) => {
                    let const_expr = lookup_const_by_id(cx.tcx, did).unwrap();
                    let e_v = eval_const_expr(cx.tcx, &*const_expr);
                    match range_covered_by_constructor(ctor_id, &e_v, &e_v) {
                        Some(true) => Some(vec!()),
                        Some(false) => None,
                        None => {
                            cx.tcx.sess.span_err(*pat_span, "mismatched types between arms");
                            None
                        }
                    }
                }
                DefVariant(_, id, _) if variant(id, false) != *ctor_id => None,
                DefVariant(..) | DefFn(..) | DefStruct(..) => {
                    Some(match args {
                        &Some(ref args) => args.clone(),
                        &None => Vec::from_elem(arity, wild())
                    })
                }
                _ => None
            }
        }

        &PatStruct(_, ref pattern_fields, _) => {
            // Is this a struct or an enum variant?
            let def = cx.tcx.def_map.borrow().get_copy(pat_id);
            let class_id = match def {
                DefVariant(_, variant_id, _) => if *ctor_id == variant(variant_id, true) {
                    Some(variant_id)
                } else {
                    None
                },
                DefStruct(struct_id) => Some(struct_id),
                _ => None
            };
            class_id.map(|variant_id| {
                let struct_fields = ty::lookup_struct_fields(cx.tcx, variant_id);
                let args = struct_fields.iter().map(|sf| {
                    match pattern_fields.iter().find(|f| f.ident.name == sf.name) {
                        Some(f) => f.pat,
                        _ => wild()
                    }
                }).collect();
                args
            })
        }

        &PatTup(ref args) =>
            Some(args.clone()),

        &PatBox(ref inner) | &PatRegion(ref inner) =>
            Some(vec!(inner.clone())),

        &PatLit(ref expr) => {
            let expr_value = eval_const_expr(cx.tcx, &**expr);
            match range_covered_by_constructor(ctor_id, &expr_value, &expr_value) {
                Some(true) => Some(vec!()),
                Some(false) => None,
                None => {
                    cx.tcx.sess.span_err(*pat_span, "mismatched types between arms");
                    None
                }
            }
        }

        &PatRange(ref from, ref to) => {
            let from_value = eval_const_expr(cx.tcx, &**from);
            let to_value = eval_const_expr(cx.tcx, &**to);
            match range_covered_by_constructor(ctor_id, &from_value, &to_value) {
                Some(true) => Some(vec!()),
                Some(false) => None,
                None => {
                    cx.tcx.sess.span_err(*pat_span, "mismatched types between arms");
                    None
                }
            }
        }

        &PatVec(ref before, ref slice, ref after) => {
            match *ctor_id {
                vec(_) => {
                    let num_elements = before.len() + after.len();
                    if num_elements < arity && slice.is_some() {
                        let mut result = Vec::new();
                        result.push_all(before.as_slice());
                        result.grow_fn(arity - num_elements, |_| wild());
                        result.push_all(after.as_slice());
                        Some(result)
                    } else if num_elements == arity {
                        let mut result = Vec::new();
                        result.push_all(before.as_slice());
                        result.push_all(after.as_slice());
                        Some(result)
                    } else {
                        None
                    }
                }
                _ => None
            }
        }

        &PatMac(_) => {
            cx.tcx.sess.span_err(*pat_span, "unexpanded macro");
            None
        }
    };
    head.map(|head| head.append(r.tail()))
}

fn default(cx: &MatchCheckCtxt, r: &[Gc<Pat>]) -> Option<Vec<Gc<Pat>>> {
    if is_wild(cx, r[0]) {
        Some(Vec::from_slice(r.tail()))
    } else {
        None
    }
}

fn check_local(cx: &mut MatchCheckCtxt, loc: &Local) {
    visit::walk_local(cx, loc, ());

    let name = match loc.source {
        LocalLet => "local",
        LocalFor => "`for` loop"
    };

    match is_refutable(cx, loc.pat) {
        Some(pat) => {
            let msg = format!(
                "refutable pattern in {} binding: `{}` not covered",
                name, pat_to_str(&*pat)
            );
            cx.tcx.sess.span_err(loc.pat.span, msg.as_slice());
        },
        None => ()
    }

    // Check legality of move bindings.
    check_legality_of_move_bindings(cx, false, [ loc.pat ]);
}

fn check_fn(cx: &mut MatchCheckCtxt,
            kind: &FnKind,
            decl: &FnDecl,
            body: &Block,
            sp: Span) {
    visit::walk_fn(cx, kind, decl, body, sp, ());
    for input in decl.inputs.iter() {
        match is_refutable(cx, input.pat) {
            Some(pat) => {
                let msg = format!(
                    "refutable pattern in function argument: `{}` not covered",
                    pat_to_str(&*pat)
                );
                cx.tcx.sess.span_err(input.pat.span, msg.as_slice());
            },
            None => ()
        }
    }
}

fn is_refutable(cx: &MatchCheckCtxt, pat: Gc<Pat>) -> Option<Gc<Pat>> {
    let pats = vec!(vec!(pat));
    is_useful(cx, &pats, [wild()], ConstructWitness)
        .useful()
        .map(|pats| {
            assert_eq!(pats.len(), 1);
            pats.get(0).clone()
        })
}

// Legality of move bindings checking

fn check_legality_of_move_bindings(cx: &MatchCheckCtxt,
                                   has_guard: bool,
                                   pats: &[Gc<Pat>]) {
    let tcx = cx.tcx;
    let def_map = &tcx.def_map;
    let mut by_ref_span = None;
    for pat in pats.iter() {
        pat_bindings(def_map, &**pat, |bm, _, span, _path| {
            match bm {
                BindByRef(_) => {
                    by_ref_span = Some(span);
                }
                BindByValue(_) => {
                }
            }
        })
    }

    let check_move: |&Pat, Option<Gc<Pat>>| = |p, sub| {
        // check legality of moving out of the enum

        // x @ Foo(..) is legal, but x @ Foo(y) isn't.
        if sub.map_or(false, |p| pat_contains_bindings(def_map, &*p)) {
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

    for pat in pats.iter() {
        walk_pat(&**pat, |p| {
            if pat_is_binding(def_map, &*p) {
                match p.node {
                    PatIdent(BindByValue(_), _, sub) => {
                        let pat_ty = ty::node_id_to_type(tcx, p.id);
                        if ty::type_moves_by_default(tcx, pat_ty) {
                            check_move(p, sub);
                        }
                    }
                    PatIdent(BindByRef(_), _, _) => {
                    }
                    _ => {
                        cx.tcx.sess.span_bug(
                            p.span,
                            format!("binding pattern {} is not an \
                                     identifier: {:?}",
                                    p.id,
                                    p.node).as_slice());
                    }
                }
            }
            true
        });
    }
}
