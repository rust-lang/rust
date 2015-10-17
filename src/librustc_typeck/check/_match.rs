// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def;
use middle::infer;
use middle::pat_util::{PatIdMap, pat_id_map, pat_is_binding};
use middle::pat_util::pat_is_resolved_const;
use middle::privacy::{AllPublic, LastMod};
use middle::subst::Substs;
use middle::ty::{self, Ty, HasTypeFlags, LvaluePreference};
use check::{check_expr, check_expr_has_type, check_expr_with_expectation};
use check::{check_expr_coercable_to_type, demand, FnCtxt, Expectation};
use check::{check_expr_with_lvalue_pref};
use check::{instantiate_path, resolve_ty_and_def_ufcs, structurally_resolved_type};
use require_same_types;
use util::nodemap::FnvHashMap;

use std::cmp;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use syntax::ast;
use syntax::codemap::{Span, Spanned};
use syntax::ptr::P;

use rustc_front::hir;
use rustc_front::print::pprust;
use rustc_front::util as hir_util;

pub fn check_pat<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                           pat: &'tcx hir::Pat,
                           expected: Ty<'tcx>)
{
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    debug!("check_pat(pat={:?},expected={:?})",
           pat,
           expected);

    match pat.node {
        hir::PatWild(_) => {
            fcx.write_ty(pat.id, expected);
        }
        hir::PatLit(ref lt) => {
            check_expr(fcx, &**lt);
            let expr_ty = fcx.expr_ty(&**lt);

            // Byte string patterns behave the same way as array patterns
            // They can denote both statically and dynamically sized byte arrays
            let mut pat_ty = expr_ty;
            if let hir::ExprLit(ref lt) = lt.node {
                if let ast::LitByteStr(_) = lt.node {
                    let expected_ty = structurally_resolved_type(fcx, pat.span, expected);
                    if let ty::TyRef(_, mt) = expected_ty.sty {
                        if let ty::TySlice(_) = mt.ty.sty {
                            pat_ty = tcx.mk_imm_ref(tcx.mk_region(ty::ReStatic),
                                                     tcx.mk_slice(tcx.types.u8))
                        }
                    }
                }
            }

            fcx.write_ty(pat.id, pat_ty);

            // somewhat surprising: in this case, the subtyping
            // relation goes the opposite way as the other
            // cases. Actually what we really want is not a subtyping
            // relation at all but rather that there exists a LUB (so
            // that they can be compared). However, in practice,
            // constants are always scalars or strings.  For scalars
            // subtyping is irrelevant, and for strings `expr_ty` is
            // type is `&'static str`, so if we say that
            //
            //     &'static str <: expected
            //
            // that's equivalent to there existing a LUB.
            demand::suptype(fcx, pat.span, expected, pat_ty);
        }
        hir::PatRange(ref begin, ref end) => {
            check_expr(fcx, begin);
            check_expr(fcx, end);

            let lhs_ty = fcx.expr_ty(begin);
            let rhs_ty = fcx.expr_ty(end);

            // Check that both end-points are of numeric or char type.
            let numeric_or_char = |ty: Ty| ty.is_numeric() || ty.is_char();
            let lhs_compat = numeric_or_char(lhs_ty);
            let rhs_compat = numeric_or_char(rhs_ty);

            if !lhs_compat || !rhs_compat {
                let span = if !lhs_compat && !rhs_compat {
                    pat.span
                } else if !lhs_compat {
                    begin.span
                } else {
                    end.span
                };

                // Note: spacing here is intentional, we want a space before "start" and "end".
                span_err!(tcx.sess, span, E0029,
                          "only char and numeric types are allowed in range patterns\n \
                           start type: {}\n end type: {}",
                          fcx.infcx().ty_to_string(lhs_ty),
                          fcx.infcx().ty_to_string(rhs_ty)
                );
                return;
            }

            // Check that the types of the end-points can be unified.
            let types_unify = require_same_types(
                    tcx, Some(fcx.infcx()), false, pat.span, rhs_ty, lhs_ty,
                    || "mismatched types in range".to_string()
            );

            // It's ok to return without a message as `require_same_types` prints an error.
            if !types_unify {
                return;
            }

            // Now that we know the types can be unified we find the unified type and use
            // it to type the entire expression.
            let common_type = fcx.infcx().resolve_type_vars_if_possible(&lhs_ty);

            fcx.write_ty(pat.id, common_type);

            // subtyping doesn't matter here, as the value is some kind of scalar
            demand::eqtype(fcx, pat.span, expected, lhs_ty);
        }
        hir::PatEnum(..) | hir::PatIdent(..) if pat_is_resolved_const(&tcx.def_map, pat) => {
            let const_did = tcx.def_map.borrow().get(&pat.id).unwrap().def_id();
            let const_scheme = tcx.lookup_item_type(const_did);
            assert!(const_scheme.generics.is_empty());
            let const_ty = pcx.fcx.instantiate_type_scheme(pat.span,
                                                           &Substs::empty(),
                                                           &const_scheme.ty);
            fcx.write_ty(pat.id, const_ty);

            // FIXME(#20489) -- we should limit the types here to scalars or something!

            // As with PatLit, what we really want here is that there
            // exist a LUB, but for the cases that can occur, subtype
            // is good enough.
            demand::suptype(fcx, pat.span, expected, const_ty);
        }
        hir::PatIdent(bm, ref path, ref sub) if pat_is_binding(&tcx.def_map, pat) => {
            let typ = fcx.local_ty(pat.span, pat.id);
            match bm {
                hir::BindByRef(mutbl) => {
                    // if the binding is like
                    //    ref x | ref const x | ref mut x
                    // then `x` is assigned a value of type `&M T` where M is the mutability
                    // and T is the expected type.
                    let region_var = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
                    let mt = ty::TypeAndMut { ty: expected, mutbl: mutbl };
                    let region_ty = tcx.mk_ref(tcx.mk_region(region_var), mt);

                    // `x` is assigned a value of type `&M T`, hence `&M T <: typeof(x)` is
                    // required. However, we use equality, which is stronger. See (*) for
                    // an explanation.
                    demand::eqtype(fcx, pat.span, region_ty, typ);
                }
                // otherwise the type of x is the expected type T
                hir::BindByValue(_) => {
                    // As above, `T <: typeof(x)` is required but we
                    // use equality, see (*) below.
                    demand::eqtype(fcx, pat.span, expected, typ);
                }
            }

            fcx.write_ty(pat.id, typ);

            // if there are multiple arms, make sure they all agree on
            // what the type of the binding `x` ought to be
            let canon_id = *pcx.map.get(&path.node.name).unwrap();
            if canon_id != pat.id {
                let ct = fcx.local_ty(pat.span, canon_id);
                demand::eqtype(fcx, pat.span, ct, typ);
            }

            if let Some(ref p) = *sub {
                check_pat(pcx, &**p, expected);
            }
        }
        hir::PatIdent(_, ref path, _) => {
            let path = hir_util::ident_to_path(path.span, path.node);
            check_pat_enum(pcx, pat, &path, Some(&[]), expected);
        }
        hir::PatEnum(ref path, ref subpats) => {
            let subpats = subpats.as_ref().map(|v| &v[..]);
            check_pat_enum(pcx, pat, path, subpats, expected);
        }
        hir::PatQPath(ref qself, ref path) => {
            let self_ty = fcx.to_ty(&qself.ty);
            let path_res = if let Some(&d) = tcx.def_map.borrow().get(&pat.id) {
                d
            } else if qself.position == 0 {
                // This is just a sentinel for finish_resolving_def_to_ty.
                let sentinel = fcx.tcx().map.local_def_id(ast::CRATE_NODE_ID);
                def::PathResolution {
                    base_def: def::DefMod(sentinel),
                    last_private: LastMod(AllPublic),
                    depth: path.segments.len()
                }
            } else {
                tcx.sess.span_bug(pat.span,
                                  &format!("unbound path {:?}", pat))
            };
            if let Some((opt_ty, segments, def)) =
                    resolve_ty_and_def_ufcs(fcx, path_res, Some(self_ty),
                                            path, pat.span, pat.id) {
                if check_assoc_item_is_const(pcx, def, pat.span) {
                    let scheme = tcx.lookup_item_type(def.def_id());
                    let predicates = tcx.lookup_predicates(def.def_id());
                    instantiate_path(fcx, segments,
                                     scheme, &predicates,
                                     opt_ty, def, pat.span, pat.id);
                    let const_ty = fcx.node_ty(pat.id);
                    demand::suptype(fcx, pat.span, expected, const_ty);
                } else {
                    fcx.write_error(pat.id)
                }
            }
        }
        hir::PatStruct(ref path, ref fields, etc) => {
            check_pat_struct(pcx, pat, path, fields, etc, expected);
        }
        hir::PatTup(ref elements) => {
            let element_tys: Vec<_> =
                (0..elements.len()).map(|_| fcx.infcx().next_ty_var())
                                        .collect();
            let pat_ty = tcx.mk_tup(element_tys.clone());
            fcx.write_ty(pat.id, pat_ty);
            demand::eqtype(fcx, pat.span, expected, pat_ty);
            for (element_pat, element_ty) in elements.iter().zip(element_tys) {
                check_pat(pcx, &**element_pat, element_ty);
            }
        }
        hir::PatBox(ref inner) => {
            let inner_ty = fcx.infcx().next_ty_var();
            let uniq_ty = tcx.mk_box(inner_ty);

            if check_dereferencable(pcx, pat.span, expected, &**inner) {
                // Here, `demand::subtype` is good enough, but I don't
                // think any errors can be introduced by using
                // `demand::eqtype`.
                demand::eqtype(fcx, pat.span, expected, uniq_ty);
                fcx.write_ty(pat.id, uniq_ty);
                check_pat(pcx, &**inner, inner_ty);
            } else {
                fcx.write_error(pat.id);
                check_pat(pcx, &**inner, tcx.types.err);
            }
        }
        hir::PatRegion(ref inner, mutbl) => {
            let expected = fcx.infcx().shallow_resolve(expected);
            if check_dereferencable(pcx, pat.span, expected, &**inner) {
                // `demand::subtype` would be good enough, but using
                // `eqtype` turns out to be equally general. See (*)
                // below for details.

                // Take region, inner-type from expected type if we
                // can, to avoid creating needless variables.  This
                // also helps with the bad interactions of the given
                // hack detailed in (*) below.
                let (rptr_ty, inner_ty) = match expected.sty {
                    ty::TyRef(_, mt) if mt.mutbl == mutbl => {
                        (expected, mt.ty)
                    }
                    _ => {
                        let inner_ty = fcx.infcx().next_ty_var();
                        let mt = ty::TypeAndMut { ty: inner_ty, mutbl: mutbl };
                        let region = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
                        let rptr_ty = tcx.mk_ref(tcx.mk_region(region), mt);
                        demand::eqtype(fcx, pat.span, expected, rptr_ty);
                        (rptr_ty, inner_ty)
                    }
                };

                fcx.write_ty(pat.id, rptr_ty);
                check_pat(pcx, &**inner, inner_ty);
            } else {
                fcx.write_error(pat.id);
                check_pat(pcx, &**inner, tcx.types.err);
            }
        }
        hir::PatVec(ref before, ref slice, ref after) => {
            let expected_ty = structurally_resolved_type(fcx, pat.span, expected);
            let inner_ty = fcx.infcx().next_ty_var();
            let pat_ty = match expected_ty.sty {
                ty::TyArray(_, size) => tcx.mk_array(inner_ty, {
                    let min_len = before.len() + after.len();
                    match *slice {
                        Some(_) => cmp::max(min_len, size),
                        None => min_len
                    }
                }),
                _ => {
                    let region = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
                    tcx.mk_ref(tcx.mk_region(region), ty::TypeAndMut {
                        ty: tcx.mk_slice(inner_ty),
                        mutbl: expected_ty.builtin_deref(true, ty::NoPreference).map(|mt| mt.mutbl)
                                                              .unwrap_or(hir::MutImmutable)
                    })
                }
            };

            fcx.write_ty(pat.id, pat_ty);

            // `demand::subtype` would be good enough, but using
            // `eqtype` turns out to be equally general. See (*)
            // below for details.
            demand::eqtype(fcx, pat.span, expected, pat_ty);

            for elt in before {
                check_pat(pcx, &**elt, inner_ty);
            }
            if let Some(ref slice) = *slice {
                let region = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
                let mutbl = expected_ty.builtin_deref(true, ty::NoPreference)
                    .map_or(hir::MutImmutable, |mt| mt.mutbl);

                let slice_ty = tcx.mk_ref(tcx.mk_region(region), ty::TypeAndMut {
                    ty: tcx.mk_slice(inner_ty),
                    mutbl: mutbl
                });
                check_pat(pcx, &**slice, slice_ty);
            }
            for elt in after {
                check_pat(pcx, &**elt, inner_ty);
            }
        }
    }


    // (*) In most of the cases above (literals and constants being
    // the exception), we relate types using strict equality, evewn
    // though subtyping would be sufficient. There are a few reasons
    // for this, some of which are fairly subtle and which cost me
    // (nmatsakis) an hour or two debugging to remember, so I thought
    // I'd write them down this time.
    //
    // 1. There is no loss of expressiveness here, though it does
    // cause some inconvenience. What we are saying is that the type
    // of `x` becomes *exactly* what is expected. This can cause unnecessary
    // errors in some cases, such as this one:
    // it will cause errors in a case like this:
    //
    // ```
    // fn foo<'x>(x: &'x int) {
    //    let a = 1;
    //    let mut z = x;
    //    z = &a;
    // }
    // ```
    //
    // The reason we might get an error is that `z` might be
    // assigned a type like `&'x int`, and then we would have
    // a problem when we try to assign `&a` to `z`, because
    // the lifetime of `&a` (i.e., the enclosing block) is
    // shorter than `'x`.
    //
    // HOWEVER, this code works fine. The reason is that the
    // expected type here is whatever type the user wrote, not
    // the initializer's type. In this case the user wrote
    // nothing, so we are going to create a type variable `Z`.
    // Then we will assign the type of the initializer (`&'x
    // int`) as a subtype of `Z`: `&'x int <: Z`. And hence we
    // will instantiate `Z` as a type `&'0 int` where `'0` is
    // a fresh region variable, with the constraint that `'x :
    // '0`.  So basically we're all set.
    //
    // Note that there are two tests to check that this remains true
    // (`regions-reassign-{match,let}-bound-pointer.rs`).
    //
    // 2. Things go horribly wrong if we use subtype. The reason for
    // THIS is a fairly subtle case involving bound regions. See the
    // `givens` field in `region_inference`, as well as the test
    // `regions-relate-bound-regions-on-closures-to-inference-variables.rs`,
    // for details. Short version is that we must sometimes detect
    // relationships between specific region variables and regions
    // bound in a closure signature, and that detection gets thrown
    // off when we substitute fresh region variables here to enable
    // subtyping.
}

fn check_assoc_item_is_const(pcx: &pat_ctxt, def: def::Def, span: Span) -> bool {
    match def {
        def::DefAssociatedConst(..) => true,
        def::DefMethod(..) => {
            span_err!(pcx.fcx.ccx.tcx.sess, span, E0327,
                      "associated items in match patterns must be constants");
            false
        }
        _ => {
            pcx.fcx.ccx.tcx.sess.span_bug(span, "non-associated item in
                                                 check_assoc_item_is_const");
        }
    }
}

pub fn check_dereferencable<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                                      span: Span, expected: Ty<'tcx>,
                                      inner: &hir::Pat) -> bool {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;
    if pat_is_binding(&tcx.def_map, inner) {
        let expected = fcx.infcx().shallow_resolve(expected);
        expected.builtin_deref(true, ty::NoPreference).map_or(true, |mt| match mt.ty.sty {
            ty::TyTrait(_) => {
                // This is "x = SomeTrait" being reduced from
                // "let &x = &SomeTrait" or "let box x = Box<SomeTrait>", an error.
                span_err!(tcx.sess, span, E0033,
                          "type `{}` cannot be dereferenced",
                          fcx.infcx().ty_to_string(expected));
                false
            }
            _ => true
        })
    } else {
        true
    }
}

pub fn check_match<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                             expr: &'tcx hir::Expr,
                             discrim: &'tcx hir::Expr,
                             arms: &'tcx [hir::Arm],
                             expected: Expectation<'tcx>,
                             match_src: hir::MatchSource) {
    let tcx = fcx.ccx.tcx;

    // Not entirely obvious: if matches may create ref bindings, we
    // want to use the *precise* type of the discriminant, *not* some
    // supertype, as the "discriminant type" (issue #23116).
    let contains_ref_bindings = arms.iter()
                                    .filter_map(|a| tcx.arm_contains_ref_binding(a))
                                    .max_by(|m| match *m {
                                        hir::MutMutable => 1,
                                        hir::MutImmutable => 0,
                                    });
    let discrim_ty;
    if let Some(m) = contains_ref_bindings {
        check_expr_with_lvalue_pref(fcx, discrim, LvaluePreference::from_mutbl(m));
        discrim_ty = fcx.expr_ty(discrim);
    } else {
        // ...but otherwise we want to use any supertype of the
        // discriminant. This is sort of a workaround, see note (*) in
        // `check_pat` for some details.
        discrim_ty = fcx.infcx().next_ty_var();
        check_expr_has_type(fcx, discrim, discrim_ty);
    };

    // Typecheck the patterns first, so that we get types for all the
    // bindings.
    for arm in arms {
        let mut pcx = pat_ctxt {
            fcx: fcx,
            map: pat_id_map(&tcx.def_map, &*arm.pats[0]),
        };
        for p in &arm.pats {
            check_pat(&mut pcx, &**p, discrim_ty);
        }
    }

    // Now typecheck the blocks.
    //
    // The result of the match is the common supertype of all the
    // arms. Start out the value as bottom, since it's the, well,
    // bottom the type lattice, and we'll be moving up the lattice as
    // we process each arm. (Note that any match with 0 arms is matching
    // on any empty type and is therefore unreachable; should the flow
    // of execution reach it, we will panic, so bottom is an appropriate
    // type in that case)
    let expected = expected.adjust_for_branches(fcx);
    let result_ty = arms.iter().fold(fcx.infcx().next_diverging_ty_var(), |result_ty, arm| {
        let bty = match expected {
            // We don't coerce to `()` so that if the match expression is a
            // statement it's branches can have any consistent type. That allows
            // us to give better error messages (pointing to a usually better
            // arm for inconsistent arms or to the whole match when a `()` type
            // is required).
            Expectation::ExpectHasType(ety) if ety != fcx.tcx().mk_nil() => {
                check_expr_coercable_to_type(fcx, &*arm.body, ety);
                ety
            }
            _ => {
                check_expr_with_expectation(fcx, &*arm.body, expected);
                fcx.node_ty(arm.body.id)
            }
        };

        if let Some(ref e) = arm.guard {
            check_expr_has_type(fcx, &**e, tcx.types.bool);
        }

        if result_ty.references_error() || bty.references_error() {
            tcx.types.err
        } else {
            let (origin, expected, found) = match match_src {
                /* if-let construct without an else block */
                hir::MatchSource::IfLetDesugar { contains_else_clause }
                if !contains_else_clause => (
                    infer::IfExpressionWithNoElse(expr.span),
                    bty,
                    result_ty,
                ),
                _ => (
                    infer::MatchExpressionArm(expr.span, arm.body.span),
                    result_ty,
                    bty,
                ),
            };

            infer::common_supertype(
                fcx.infcx(),
                origin,
                true,
                expected,
                found,
            )
        }
    });

    fcx.write_ty(expr.id, result_ty);
}

pub struct pat_ctxt<'a, 'tcx: 'a> {
    pub fcx: &'a FnCtxt<'a, 'tcx>,
    pub map: PatIdMap,
}

pub fn check_pat_struct<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>, pat: &'tcx hir::Pat,
                                  path: &hir::Path, fields: &'tcx [Spanned<hir::FieldPat>],
                                  etc: bool, expected: Ty<'tcx>) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    let def = tcx.def_map.borrow().get(&pat.id).unwrap().full_def();
    let variant = match fcx.def_struct_variant(def, path.span) {
        Some((_, variant)) => variant,
        None => {
            let name = pprust::path_to_string(path);
            span_err!(tcx.sess, pat.span, E0163,
                      "`{}` does not name a struct or a struct variant", name);
            fcx.write_error(pat.id);

            for field in fields {
                check_pat(pcx, &field.node.pat, tcx.types.err);
            }
            return;
        }
    };

    let pat_ty = pcx.fcx.instantiate_type(def.def_id(), path);
    let item_substs = match pat_ty.sty {
        ty::TyStruct(_, substs) | ty::TyEnum(_, substs) => substs,
        _ => tcx.sess.span_bug(pat.span, "struct variant is not an ADT")
    };
    demand::eqtype(fcx, pat.span, expected, pat_ty);
    check_struct_pat_fields(pcx, pat.span, fields, variant, &item_substs, etc);

    fcx.write_ty(pat.id, pat_ty);
    fcx.write_substs(pat.id, ty::ItemSubsts { substs: item_substs.clone() });
}

pub fn check_pat_enum<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                                pat: &hir::Pat,
                                path: &hir::Path,
                                subpats: Option<&'tcx [P<hir::Pat>]>,
                                expected: Ty<'tcx>)
{
    // Typecheck the path.
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    let path_res = *tcx.def_map.borrow().get(&pat.id).unwrap();

    let (opt_ty, segments, def) = match resolve_ty_and_def_ufcs(fcx, path_res,
                                                                None, path,
                                                                pat.span, pat.id) {
        Some(resolution) => resolution,
        // Error handling done inside resolve_ty_and_def_ufcs, so if
        // resolution fails just return.
        None => {return;}
    };

    // Items that were partially resolved before should have been resolved to
    // associated constants (i.e. not methods).
    if path_res.depth != 0 && !check_assoc_item_is_const(pcx, def, pat.span) {
        fcx.write_error(pat.id);
        return;
    }

    let enum_def = def.variant_def_ids()
        .map_or_else(|| def.def_id(), |(enum_def, _)| enum_def);

    let ctor_scheme = tcx.lookup_item_type(enum_def);
    let ctor_predicates = tcx.lookup_predicates(enum_def);
    let path_scheme = if ctor_scheme.ty.is_fn() {
        let fn_ret = tcx.no_late_bound_regions(&ctor_scheme.ty.fn_ret()).unwrap();
        ty::TypeScheme {
            ty: fn_ret.unwrap(),
            generics: ctor_scheme.generics,
        }
    } else {
        ctor_scheme
    };
    instantiate_path(pcx.fcx, segments,
                     path_scheme, &ctor_predicates,
                     opt_ty, def, pat.span, pat.id);

    // If we didn't have a fully resolved path to start with, we had an
    // associated const, and we should quit now, since the rest of this
    // function uses checks specific to structs and enums.
    if path_res.depth != 0 {
        let pat_ty = fcx.node_ty(pat.id);
        demand::suptype(fcx, pat.span, expected, pat_ty);
        return;
    }

    let pat_ty = fcx.node_ty(pat.id);
    demand::eqtype(fcx, pat.span, expected, pat_ty);


    let real_path_ty = fcx.node_ty(pat.id);
    let (arg_tys, kind_name): (Vec<_>, &'static str) = match real_path_ty.sty {
        ty::TyEnum(enum_def, expected_substs)
            if def == def::DefVariant(enum_def.did, def.def_id(), false) =>
        {
            let variant = enum_def.variant_of_def(def);
            (variant.fields
                    .iter()
                    .map(|f| fcx.instantiate_type_scheme(pat.span,
                                                         expected_substs,
                                                         &f.unsubst_ty()))
                    .collect(),
             "variant")
        }
        ty::TyStruct(struct_def, expected_substs) => {
            (struct_def.struct_variant()
                       .fields
                       .iter()
                       .map(|f| fcx.instantiate_type_scheme(pat.span,
                                                            expected_substs,
                                                            &f.unsubst_ty()))
                       .collect(),
             "struct")
        }
        _ => {
            let name = pprust::path_to_string(path);
            span_err!(tcx.sess, pat.span, E0164,
                "`{}` does not name a non-struct variant or a tuple struct", name);
            fcx.write_error(pat.id);

            if let Some(subpats) = subpats {
                for pat in subpats {
                    check_pat(pcx, &**pat, tcx.types.err);
                }
            }
            return;
        }
    };

    if let Some(subpats) = subpats {
        if subpats.len() == arg_tys.len() {
            for (subpat, arg_ty) in subpats.iter().zip(arg_tys) {
                check_pat(pcx, &**subpat, arg_ty);
            }
        } else if arg_tys.is_empty() {
            span_err!(tcx.sess, pat.span, E0024,
                      "this pattern has {} field{}, but the corresponding {} has no fields",
                      subpats.len(), if subpats.len() == 1 {""} else {"s"}, kind_name);

            for pat in subpats {
                check_pat(pcx, &**pat, tcx.types.err);
            }
        } else {
            span_err!(tcx.sess, pat.span, E0023,
                      "this pattern has {} field{}, but the corresponding {} has {} field{}",
                      subpats.len(), if subpats.len() == 1 {""} else {"s"},
                      kind_name,
                      arg_tys.len(), if arg_tys.len() == 1 {""} else {"s"});

            for pat in subpats {
                check_pat(pcx, &**pat, tcx.types.err);
            }
        }
    }
}

/// `path` is the AST path item naming the type of this struct.
/// `fields` is the field patterns of the struct pattern.
/// `struct_fields` describes the type of each field of the struct.
/// `struct_id` is the ID of the struct.
/// `etc` is true if the pattern said '...' and false otherwise.
pub fn check_struct_pat_fields<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                                         span: Span,
                                         fields: &'tcx [Spanned<hir::FieldPat>],
                                         variant: ty::VariantDef<'tcx>,
                                         substs: &Substs<'tcx>,
                                         etc: bool) {
    let tcx = pcx.fcx.ccx.tcx;

    // Index the struct fields' types.
    let field_map = variant.fields
        .iter()
        .map(|field| (field.name, field))
        .collect::<FnvHashMap<_, _>>();

    // Keep track of which fields have already appeared in the pattern.
    let mut used_fields = FnvHashMap();

    // Typecheck each field.
    for &Spanned { node: ref field, span } in fields {
        let field_ty = match used_fields.entry(field.name) {
            Occupied(occupied) => {
                span_err!(tcx.sess, span, E0025,
                    "field `{}` bound multiple times in the pattern",
                    field.name);
                span_note!(tcx.sess, *occupied.get(),
                    "field `{}` previously bound here",
                    field.name);
                tcx.types.err
            }
            Vacant(vacant) => {
                vacant.insert(span);
                field_map.get(&field.name)
                    .map(|f| pcx.fcx.field_ty(span, f, substs))
                    .unwrap_or_else(|| {
                        span_err!(tcx.sess, span, E0026,
                            "struct `{}` does not have a field named `{}`",
                            tcx.item_path_str(variant.did),
                            field.name);
                        tcx.types.err
                    })
            }
        };

        check_pat(pcx, &*field.pat, field_ty);
    }

    // Report an error if not all the fields were specified.
    if !etc {
        for field in variant.fields
            .iter()
            .filter(|field| !used_fields.contains_key(&field.name)) {
            span_err!(tcx.sess, span, E0027,
                "pattern does not mention field `{}`",
                field.name);
        }
    }
}
