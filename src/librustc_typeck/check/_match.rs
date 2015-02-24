// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::const_eval;
use middle::def;
use middle::infer;
use middle::pat_util::{PatIdMap, pat_id_map, pat_is_binding, pat_is_const};
use middle::subst::{Substs};
use middle::ty::{self, Ty};
use check::{check_expr, check_expr_has_type, check_expr_with_expectation};
use check::{check_expr_coercable_to_type, demand, FnCtxt, Expectation};
use check::{instantiate_path, structurally_resolved_type};
use require_same_types;
use util::nodemap::FnvHashMap;
use util::ppaux::Repr;

use std::cmp::{self, Ordering};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::{Span, Spanned};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;

pub fn check_pat<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                           pat: &'tcx ast::Pat,
                           expected: Ty<'tcx>)
{
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    debug!("check_pat(pat={},expected={})",
           pat.repr(tcx),
           expected.repr(tcx));

    match pat.node {
        ast::PatWild(_) => {
            fcx.write_ty(pat.id, expected);
        }
        ast::PatLit(ref lt) => {
            check_expr(fcx, &**lt);
            let expr_ty = fcx.expr_ty(&**lt);
            fcx.write_ty(pat.id, expr_ty);

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
            demand::suptype(fcx, pat.span, expected, expr_ty);
        }
        ast::PatRange(ref begin, ref end) => {
            check_expr(fcx, &**begin);
            check_expr(fcx, &**end);

            let lhs_ty = fcx.expr_ty(&**begin);
            let rhs_ty = fcx.expr_ty(&**end);

            let lhs_eq_rhs =
                require_same_types(
                    tcx, Some(fcx.infcx()), false, pat.span, lhs_ty, rhs_ty,
                    || "mismatched types in range".to_string());

            let numeric_or_char =
                lhs_eq_rhs && (ty::type_is_numeric(lhs_ty) || ty::type_is_char(lhs_ty));

            if numeric_or_char {
                match const_eval::compare_lit_exprs(tcx, &**begin, &**end, Some(lhs_ty)) {
                    Some(Ordering::Less) |
                    Some(Ordering::Equal) => {}
                    Some(Ordering::Greater) => {
                        span_err!(tcx.sess, begin.span, E0030,
                            "lower range bound must be less than upper");
                    }
                    None => {
                        span_err!(tcx.sess, begin.span, E0031,
                            "mismatched types in range");
                    }
                }
            } else {
                span_err!(tcx.sess, begin.span, E0029,
                          "only char and numeric types are allowed in range");
            }

            fcx.write_ty(pat.id, lhs_ty);

            // subtyping doesn't matter here, as the value is some kind of scalar
            demand::eqtype(fcx, pat.span, expected, lhs_ty);
        }
        ast::PatEnum(..) | ast::PatIdent(..) if pat_is_const(&tcx.def_map, pat) => {
            let const_did = tcx.def_map.borrow()[pat.id].clone().def_id();
            let const_scheme = ty::lookup_item_type(tcx, const_did);
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
        ast::PatIdent(bm, ref path, ref sub) if pat_is_binding(&tcx.def_map, pat) => {
            let typ = fcx.local_ty(pat.span, pat.id);
            match bm {
                ast::BindByRef(mutbl) => {
                    // if the binding is like
                    //    ref x | ref const x | ref mut x
                    // then `x` is assigned a value of type `&M T` where M is the mutability
                    // and T is the expected type.
                    let region_var = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
                    let mt = ty::mt { ty: expected, mutbl: mutbl };
                    let region_ty = ty::mk_rptr(tcx, tcx.mk_region(region_var), mt);

                    // `x` is assigned a value of type `&M T`, hence `&M T <: typeof(x)` is
                    // required. However, we use equality, which is stronger. See (*) for
                    // an explanation.
                    demand::eqtype(fcx, pat.span, region_ty, typ);
                }
                // otherwise the type of x is the expected type T
                ast::BindByValue(_) => {
                    // As above, `T <: typeof(x)` is required but we
                    // use equality, see (*) below.
                    demand::eqtype(fcx, pat.span, expected, typ);
                }
            }

            fcx.write_ty(pat.id, typ);

            // if there are multiple arms, make sure they all agree on
            // what the type of the binding `x` ought to be
            let canon_id = pcx.map[path.node];
            if canon_id != pat.id {
                let ct = fcx.local_ty(pat.span, canon_id);
                demand::eqtype(fcx, pat.span, ct, typ);
            }

            if let Some(ref p) = *sub {
                check_pat(pcx, &**p, expected);
            }
        }
        ast::PatIdent(_, ref path, _) => {
            let path = ast_util::ident_to_path(path.span, path.node);
            check_pat_enum(pcx, pat, &path, Some(&[]), expected);
        }
        ast::PatEnum(ref path, ref subpats) => {
            let subpats = subpats.as_ref().map(|v| &v[..]);
            check_pat_enum(pcx, pat, path, subpats, expected);
        }
        ast::PatStruct(ref path, ref fields, etc) => {
            check_pat_struct(pcx, pat, path, fields, etc, expected);
        }
        ast::PatTup(ref elements) => {
            let element_tys: Vec<_> =
                (0..elements.len()).map(|_| fcx.infcx().next_ty_var())
                                        .collect();
            let pat_ty = ty::mk_tup(tcx, element_tys.clone());
            fcx.write_ty(pat.id, pat_ty);
            demand::eqtype(fcx, pat.span, expected, pat_ty);
            for (element_pat, element_ty) in elements.iter().zip(element_tys.into_iter()) {
                check_pat(pcx, &**element_pat, element_ty);
            }
        }
        ast::PatBox(ref inner) => {
            let inner_ty = fcx.infcx().next_ty_var();
            let uniq_ty = ty::mk_uniq(tcx, inner_ty);

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
        ast::PatRegion(ref inner, mutbl) => {
            let inner_ty = fcx.infcx().next_ty_var();

            let mt = ty::mt { ty: inner_ty, mutbl: mutbl };
            let region = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
            let rptr_ty = ty::mk_rptr(tcx, tcx.mk_region(region), mt);

            if check_dereferencable(pcx, pat.span, expected, &**inner) {
                // `demand::subtype` would be good enough, but using
                // `eqtype` turns out to be equally general. See (*)
                // below for details.
                demand::eqtype(fcx, pat.span, expected, rptr_ty);
                fcx.write_ty(pat.id, rptr_ty);
                check_pat(pcx, &**inner, inner_ty);
            } else {
                fcx.write_error(pat.id);
                check_pat(pcx, &**inner, tcx.types.err);
            }
        }
        ast::PatVec(ref before, ref slice, ref after) => {
            let expected_ty = structurally_resolved_type(fcx, pat.span, expected);
            let inner_ty = fcx.infcx().next_ty_var();
            let pat_ty = match expected_ty.sty {
                ty::ty_vec(_, Some(size)) => ty::mk_vec(tcx, inner_ty, Some({
                    let min_len = before.len() + after.len();
                    match *slice {
                        Some(_) => cmp::max(min_len, size),
                        None => min_len
                    }
                })),
                _ => {
                    let region = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
                    ty::mk_slice(tcx, tcx.mk_region(region), ty::mt {
                        ty: inner_ty,
                        mutbl: ty::deref(expected_ty, true).map(|mt| mt.mutbl)
                                                           .unwrap_or(ast::MutImmutable)
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
                let mutbl = ty::deref(expected_ty, true)
                    .map_or(ast::MutImmutable, |mt| mt.mutbl);

                let slice_ty = ty::mk_slice(tcx, tcx.mk_region(region), ty::mt {
                    ty: inner_ty,
                    mutbl: mutbl
                });
                check_pat(pcx, &**slice, slice_ty);
            }
            for elt in after {
                check_pat(pcx, &**elt, inner_ty);
            }
        }
        ast::PatMac(_) => tcx.sess.bug("unexpanded macro")
    }


    // (*) In most of the cases above (literals and constants being
    // the exception), we relate types using strict equality, evewn
    // though subtyping would be sufficient. There are a few reasons
    // for this, some of which are fairly subtle and which cost me
    // (nmatsakis) an hour or two debugging to remember, so I thought
    // I'd write them down this time.
    //
    // 1. Most importantly, there is no loss of expressiveness
    // here. What we are saying is that the type of `x`
    // becomes *exactly* what is expected. This might seem
    // like it will cause errors in a case like this:
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

pub fn check_dereferencable<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                                      span: Span, expected: Ty<'tcx>,
                                      inner: &ast::Pat) -> bool {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;
    if pat_is_binding(&tcx.def_map, inner) {
        let expected = fcx.infcx().shallow_resolve(expected);
        ty::deref(expected, true).map_or(true, |mt| match mt.ty.sty {
            ty::ty_trait(_) => {
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
                             expr: &'tcx ast::Expr,
                             discrim: &'tcx ast::Expr,
                             arms: &'tcx [ast::Arm],
                             expected: Expectation<'tcx>,
                             match_src: ast::MatchSource) {
    let tcx = fcx.ccx.tcx;

    let discrim_ty = fcx.infcx().next_ty_var();
    check_expr_has_type(fcx, discrim, discrim_ty);

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
            Expectation::ExpectHasType(ety) if ety != ty::mk_nil(fcx.tcx()) => {
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

        if ty::type_is_error(result_ty) || ty::type_is_error(bty) {
            tcx.types.err
        } else {
            let (origin, expected, found) = match match_src {
                /* if-let construct without an else block */
                ast::MatchSource::IfLetDesugar { contains_else_clause }
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

pub fn check_pat_struct<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>, pat: &'tcx ast::Pat,
                                  path: &ast::Path, fields: &'tcx [Spanned<ast::FieldPat>],
                                  etc: bool, expected: Ty<'tcx>) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    let def = tcx.def_map.borrow()[pat.id].clone();
    let (enum_def_id, variant_def_id) = match def {
        def::DefTrait(_) => {
            let name = pprust::path_to_string(path);
            span_err!(tcx.sess, pat.span, E0168,
                "use of trait `{}` in a struct pattern", name);
            fcx.write_error(pat.id);

            for field in fields {
                check_pat(pcx, &*field.node.pat, tcx.types.err);
            }
            return;
        },
        _ => {
            let def_type = ty::lookup_item_type(tcx, def.def_id());
            match def_type.ty.sty {
                ty::ty_struct(struct_def_id, _) =>
                    (struct_def_id, struct_def_id),
                ty::ty_enum(enum_def_id, _)
                    if def == def::DefVariant(enum_def_id, def.def_id(), true) =>
                    (enum_def_id, def.def_id()),
                _ => {
                    let name = pprust::path_to_string(path);
                    span_err!(tcx.sess, pat.span, E0163,
                        "`{}` does not name a struct or a struct variant", name);
                    fcx.write_error(pat.id);

                    for field in fields {
                        check_pat(pcx, &*field.node.pat, tcx.types.err);
                    }
                    return;
                }
            }
        }
    };

    instantiate_path(pcx.fcx,
                     path,
                     ty::lookup_item_type(tcx, enum_def_id),
                     &ty::lookup_predicates(tcx, enum_def_id),
                     None,
                     def,
                     pat.span,
                     pat.id);

    let pat_ty = fcx.node_ty(pat.id);
    demand::eqtype(fcx, pat.span, expected, pat_ty);

    let item_substs = fcx
        .item_substs()
        .get(&pat.id)
        .map(|substs| substs.substs.clone())
        .unwrap_or_else(|| Substs::empty());

    let struct_fields = ty::struct_fields(tcx, variant_def_id, &item_substs);
    check_struct_pat_fields(pcx, pat.span, fields, &struct_fields,
                            variant_def_id, etc);
}

pub fn check_pat_enum<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                                pat: &ast::Pat,
                                path: &ast::Path,
                                subpats: Option<&'tcx [P<ast::Pat>]>,
                                expected: Ty<'tcx>)
{
    // Typecheck the path.
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    let def = tcx.def_map.borrow()[pat.id].clone();
    let enum_def = def.variant_def_ids()
        .map_or_else(|| def.def_id(), |(enum_def, _)| enum_def);

    let ctor_scheme = ty::lookup_item_type(tcx, enum_def);
    let ctor_predicates = ty::lookup_predicates(tcx, enum_def);
    let path_scheme = if ty::is_fn_ty(ctor_scheme.ty) {
        let fn_ret = ty::no_late_bound_regions(tcx, &ty::ty_fn_ret(ctor_scheme.ty)).unwrap();
        ty::TypeScheme {
            ty: fn_ret.unwrap(),
            generics: ctor_scheme.generics,
        }
    } else {
        ctor_scheme
    };
    instantiate_path(pcx.fcx, path, path_scheme, &ctor_predicates, None, def, pat.span, pat.id);

    let pat_ty = fcx.node_ty(pat.id);
    demand::eqtype(fcx, pat.span, expected, pat_ty);

    let real_path_ty = fcx.node_ty(pat.id);
    let (arg_tys, kind_name): (Vec<_>, &'static str) = match real_path_ty.sty {
        ty::ty_enum(enum_def_id, expected_substs)
            if def == def::DefVariant(enum_def_id, def.def_id(), false) =>
        {
            let variant = ty::enum_variant_with_id(tcx, enum_def_id, def.def_id());
            (variant.args.iter()
                         .map(|t| fcx.instantiate_type_scheme(pat.span, expected_substs, t))
                         .collect(),
             "variant")
        }
        ty::ty_struct(struct_def_id, expected_substs) => {
            let struct_fields = ty::struct_fields(tcx, struct_def_id, expected_substs);
            (struct_fields.iter()
                          .map(|field| fcx.instantiate_type_scheme(pat.span,
                                                                   expected_substs,
                                                                   &field.mt.ty))
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
            for (subpat, arg_ty) in subpats.iter().zip(arg_tys.iter()) {
                check_pat(pcx, &**subpat, *arg_ty);
            }
        } else if arg_tys.len() == 0 {
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
                                         fields: &'tcx [Spanned<ast::FieldPat>],
                                         struct_fields: &[ty::field<'tcx>],
                                         struct_id: ast::DefId,
                                         etc: bool) {
    let tcx = pcx.fcx.ccx.tcx;

    // Index the struct fields' types.
    let field_type_map = struct_fields
        .iter()
        .map(|field| (field.name, field.mt.ty))
        .collect::<FnvHashMap<_, _>>();

    // Keep track of which fields have already appeared in the pattern.
    let mut used_fields = FnvHashMap();

    // Typecheck each field.
    for &Spanned { node: ref field, span } in fields {
        let field_type = match used_fields.entry(field.ident.name) {
            Occupied(occupied) => {
                span_err!(tcx.sess, span, E0025,
                    "field `{}` bound multiple times in the pattern",
                    token::get_ident(field.ident));
                span_note!(tcx.sess, *occupied.get(),
                    "field `{}` previously bound here",
                    token::get_ident(field.ident));
                tcx.types.err
            }
            Vacant(vacant) => {
                vacant.insert(span);
                field_type_map.get(&field.ident.name).cloned()
                    .unwrap_or_else(|| {
                        span_err!(tcx.sess, span, E0026,
                            "struct `{}` does not have a field named `{}`",
                            ty::item_path_str(tcx, struct_id),
                            token::get_ident(field.ident));
                        tcx.types.err
                    })
            }
        };

        let field_type = pcx.fcx.normalize_associated_types_in(span, &field_type);

        check_pat(pcx, &*field.pat, field_type);
    }

    // Report an error if not all the fields were specified.
    if !etc {
        for field in struct_fields
            .iter()
            .filter(|field| !used_fields.contains_key(&field.name)) {
            span_err!(tcx.sess, span, E0027,
                "pattern does not mention field `{}`",
                token::get_name(field.name));
        }
    }
}
