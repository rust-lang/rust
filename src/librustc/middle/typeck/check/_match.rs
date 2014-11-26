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
use middle::pat_util::{PatIdMap, pat_id_map, pat_is_binding, pat_is_const};
use middle::subst::{Subst, Substs};
use middle::ty::{mod, Ty};
use typeck::check::{check_expr, check_expr_has_type, demand, FnCtxt};
use typeck::check::{instantiate_path, structurally_resolved_type, valid_range_bounds};
use middle::infer::{mod, resolve};
use typeck::require_same_types;
use util::nodemap::FnvHashMap;

use std::cmp;
use std::collections::hash_map::{Occupied, Vacant};
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::{Span, Spanned};
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;

pub fn check_pat<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                           pat: &ast::Pat, expected: Ty<'tcx>) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    match pat.node {
        ast::PatWild(_) => {
            fcx.write_ty(pat.id, expected);
        }
        ast::PatLit(ref lt) => {
            check_expr(fcx, &**lt);
            let expr_ty = fcx.expr_ty(&**lt);
            fcx.write_ty(pat.id, expr_ty);
            demand::suptype(fcx, pat.span, expected, expr_ty);
        }
        ast::PatRange(ref begin, ref end) => {
            check_expr(fcx, &**begin);
            check_expr(fcx, &**end);

            let lhs_ty = fcx.expr_ty(&**begin);
            let rhs_ty = fcx.expr_ty(&**end);
            if require_same_types(
                tcx, Some(fcx.infcx()), false, pat.span, lhs_ty, rhs_ty,
                || "mismatched types in range".to_string())
                && (ty::type_is_numeric(lhs_ty) || ty::type_is_char(rhs_ty)) {
                match valid_range_bounds(fcx.ccx, &**begin, &**end) {
                    Some(false) => {
                        span_err!(tcx.sess, begin.span, E0030,
                            "lower range bound must be less than upper");
                    },
                    None => {
                        span_err!(tcx.sess, begin.span, E0031,
                            "mismatched types in range");
                    },
                    Some(true) => {}
                }
            } else {
                span_err!(tcx.sess, begin.span, E0029,
                    "only char and numeric types are allowed in range");
            }

            fcx.write_ty(pat.id, lhs_ty);
            demand::eqtype(fcx, pat.span, expected, lhs_ty);
        }
        ast::PatEnum(..) | ast::PatIdent(..) if pat_is_const(&tcx.def_map, pat) => {
            let const_did = tcx.def_map.borrow()[pat.id].clone().def_id();
            let const_pty = ty::lookup_item_type(tcx, const_did);
            fcx.write_ty(pat.id, const_pty.ty);
            demand::suptype(fcx, pat.span, expected, const_pty.ty);
        }
        ast::PatIdent(bm, ref path, ref sub) if pat_is_binding(&tcx.def_map, pat) => {
            let typ = fcx.local_ty(pat.span, pat.id);
            match bm {
                ast::BindByRef(mutbl) => {
                    // if the binding is like
                    //    ref x | ref const x | ref mut x
                    // then the type of x is &M T where M is the mutability
                    // and T is the expected type
                    let region_var = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
                    let mt = ty::mt { ty: expected, mutbl: mutbl };
                    let region_ty = ty::mk_rptr(tcx, region_var, mt);
                    demand::eqtype(fcx, pat.span, region_ty, typ);
                }
                // otherwise the type of x is the expected type T
                ast::BindByValue(_) => {
                    demand::eqtype(fcx, pat.span, expected, typ);
                }
            }
            fcx.write_ty(pat.id, typ);

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
            check_pat_enum(pcx, pat, &path, &Some(vec![]), expected);
        }
        ast::PatEnum(ref path, ref subpats) => {
            check_pat_enum(pcx, pat, path, subpats, expected);
        }
        ast::PatStruct(ref path, ref fields, etc) => {
            check_pat_struct(pcx, pat, path, fields.as_slice(), etc, expected);
        }
        ast::PatTup(ref elements) => {
            let element_tys = Vec::from_fn(elements.len(), |_| fcx.infcx().next_ty_var());
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
                demand::suptype(fcx, pat.span, expected, uniq_ty);
                fcx.write_ty(pat.id, uniq_ty);
                check_pat(pcx, &**inner, inner_ty);
            } else {
                fcx.write_error(pat.id);
                check_pat(pcx, &**inner, ty::mk_err());
            }
        }
        ast::PatRegion(ref inner) => {
            let inner_ty = fcx.infcx().next_ty_var();

            let mutbl = infer::resolve_type(
                fcx.infcx(), Some(pat.span),
                expected, resolve::try_resolve_tvar_shallow)
                .ok()
                .and_then(|t| ty::deref(t, true))
                .map_or(ast::MutImmutable, |mt| mt.mutbl);

            let mt = ty::mt { ty: inner_ty, mutbl: mutbl };
            let region = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
            let rptr_ty = ty::mk_rptr(tcx, region, mt);

            if check_dereferencable(pcx, pat.span, expected, &**inner) {
                demand::suptype(fcx, pat.span, expected, rptr_ty);
                fcx.write_ty(pat.id, rptr_ty);
                check_pat(pcx, &**inner, inner_ty);
            } else {
                fcx.write_error(pat.id);
                check_pat(pcx, &**inner, ty::mk_err());
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
                    ty::mk_slice(tcx, region, ty::mt {
                        ty: inner_ty,
                        mutbl: ty::deref(expected_ty, true)
                            .map_or(ast::MutImmutable, |mt| mt.mutbl)
                    })
                }
            };

            fcx.write_ty(pat.id, pat_ty);
            demand::suptype(fcx, pat.span, expected, pat_ty);

            for elt in before.iter() {
                check_pat(pcx, &**elt, inner_ty);
            }
            if let Some(ref slice) = *slice {
                let region = fcx.infcx().next_region_var(infer::PatternRegion(pat.span));
                let mutbl = ty::deref(expected_ty, true)
                    .map_or(ast::MutImmutable, |mt| mt.mutbl);

                let slice_ty = ty::mk_slice(tcx, region, ty::mt {
                    ty: inner_ty,
                    mutbl: mutbl
                });
                check_pat(pcx, &**slice, slice_ty);
            }
            for elt in after.iter() {
                check_pat(pcx, &**elt, inner_ty);
            }
        }
        ast::PatMac(_) => tcx.sess.bug("unexpanded macro")
    }
}

pub fn check_dereferencable<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>,
                                      span: Span, expected: Ty<'tcx>,
                                      inner: &ast::Pat) -> bool {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;
    match infer::resolve_type(
        fcx.infcx(), Some(span),
        expected, resolve::try_resolve_tvar_shallow) {
        Ok(t) if pat_is_binding(&tcx.def_map, inner) => {
            ty::deref(t, true).map_or(true, |mt| match mt.ty.sty {
                ty::ty_trait(_) => {
                    // This is "x = SomeTrait" being reduced from
                    // "let &x = &SomeTrait" or "let box x = Box<SomeTrait>", an error.
                    span_err!(tcx.sess, span, E0033,
                        "type `{}` cannot be dereferenced",
                        fcx.infcx().ty_to_string(t));
                    false
                }
                _ => true
            })
        }
        _ => true
    }
}

pub fn check_match(fcx: &FnCtxt,
                   expr: &ast::Expr,
                   discrim: &ast::Expr,
                   arms: &[ast::Arm]) {
    let tcx = fcx.ccx.tcx;

    let discrim_ty = fcx.infcx().next_ty_var();
    check_expr_has_type(fcx, discrim, discrim_ty);

    // Typecheck the patterns first, so that we get types for all the
    // bindings.
    for arm in arms.iter() {
        let mut pcx = pat_ctxt {
            fcx: fcx,
            map: pat_id_map(&tcx.def_map, &*arm.pats[0]),
        };
        for p in arm.pats.iter() {
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
    let result_ty = arms.iter().fold(fcx.infcx().next_diverging_ty_var(), |result_ty, arm| {
        check_expr(fcx, &*arm.body);
        let bty = fcx.node_ty(arm.body.id);

        if let Some(ref e) = arm.guard {
            check_expr_has_type(fcx, &**e, ty::mk_bool());
        }

        if ty::type_is_error(result_ty) || ty::type_is_error(bty) {
            ty::mk_err()
        } else {
            infer::common_supertype(
                fcx.infcx(),
                infer::MatchExpressionArm(expr.span, arm.body.span),
                true, // result_ty is "expected" here
                result_ty,
                bty
            )
        }
    });

    fcx.write_ty(expr.id, result_ty);
}

pub struct pat_ctxt<'a, 'tcx: 'a> {
    pub fcx: &'a FnCtxt<'a, 'tcx>,
    pub map: PatIdMap,
}

pub fn check_pat_struct<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>, pat: &ast::Pat,
                                  path: &ast::Path, fields: &[Spanned<ast::FieldPat>],
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

            for field in fields.iter() {
                check_pat(pcx, &*field.node.pat, ty::mk_err());
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

                    for field in fields.iter() {
                        check_pat(pcx, &*field.node.pat, ty::mk_err());
                    }
                    return;
                }
            }
        }
    };

    instantiate_path(pcx.fcx, path, ty::lookup_item_type(tcx, enum_def_id),
                     def, pat.span, pat.id);

    let pat_ty = fcx.node_ty(pat.id);
    demand::eqtype(fcx, pat.span, expected, pat_ty);

    let item_substs = fcx
        .item_substs()
        .get(&pat.id)
        .map(|substs| substs.substs.clone())
        .unwrap_or_else(|| Substs::empty());

    let struct_fields = ty::struct_fields(tcx, variant_def_id, &item_substs);
    check_struct_pat_fields(pcx, pat.span, fields, struct_fields.as_slice(),
                            variant_def_id, etc);
}

pub fn check_pat_enum<'a, 'tcx>(pcx: &pat_ctxt<'a, 'tcx>, pat: &ast::Pat,
                                path: &ast::Path, subpats: &Option<Vec<P<ast::Pat>>>,
                                expected: Ty<'tcx>) {

    // Typecheck the path.
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    let def = tcx.def_map.borrow()[pat.id].clone();
    let enum_def = def.variant_def_ids()
        .map_or_else(|| def.def_id(), |(enum_def, _)| enum_def);

    let ctor_pty = ty::lookup_item_type(tcx, enum_def);
    let path_ty = if ty::is_fn_ty(ctor_pty.ty) {
        ty::Polytype {
            ty: ty::ty_fn_ret(ctor_pty.ty).unwrap(),
            ..ctor_pty
        }
    } else {
        ctor_pty
    };
    instantiate_path(pcx.fcx, path, path_ty, def, pat.span, pat.id);

    let pat_ty = fcx.node_ty(pat.id);
    demand::eqtype(fcx, pat.span, expected, pat_ty);

    let real_path_ty = fcx.node_ty(pat.id);
    let (arg_tys, kind_name) = match real_path_ty.sty {
        ty::ty_enum(enum_def_id, ref expected_substs)
            if def == def::DefVariant(enum_def_id, def.def_id(), false) => {
            let variant = ty::enum_variant_with_id(tcx, enum_def_id, def.def_id());
            (variant.args.iter().map(|t| t.subst(tcx, expected_substs)).collect::<Vec<_>>(),
                "variant")
        }
        ty::ty_struct(struct_def_id, ref expected_substs) => {
            let struct_fields = ty::struct_fields(tcx, struct_def_id, expected_substs);
            (struct_fields.iter().map(|field| field.mt.ty).collect::<Vec<_>>(),
                "struct")
        }
        _ => {
            let name = pprust::path_to_string(path);
            span_err!(tcx.sess, pat.span, E0164,
                "`{}` does not name a non-struct variant or a tuple struct", name);
            fcx.write_error(pat.id);

            if let Some(ref subpats) = *subpats {
                for pat in subpats.iter() {
                    check_pat(pcx, &**pat, ty::mk_err());
                }
            }
            return;
        }
    };

    if let Some(ref subpats) = *subpats {
        if subpats.len() == arg_tys.len() {
            for (subpat, arg_ty) in subpats.iter().zip(arg_tys.iter()) {
                check_pat(pcx, &**subpat, *arg_ty);
            }
        } else if arg_tys.len() == 0 {
            span_err!(tcx.sess, pat.span, E0024,
                      "this pattern has {} field{}, but the corresponding {} has no fields",
                      subpats.len(), if subpats.len() == 1 {""} else {"s"}, kind_name);

            for pat in subpats.iter() {
                check_pat(pcx, &**pat, ty::mk_err());
            }
        } else {
            span_err!(tcx.sess, pat.span, E0023,
                      "this pattern has {} field{}, but the corresponding {} has {} field{}",
                      subpats.len(), if subpats.len() == 1 {""} else {"s"},
                      kind_name,
                      arg_tys.len(), if arg_tys.len() == 1 {""} else {"s"});

            for pat in subpats.iter() {
                check_pat(pcx, &**pat, ty::mk_err());
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
                                         fields: &[Spanned<ast::FieldPat>],
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
    let mut used_fields = FnvHashMap::new();

    // Typecheck each field.
    for &Spanned { node: ref field, span } in fields.iter() {
        let field_type = match used_fields.entry(field.ident.name) {
            Occupied(occupied) => {
                span_err!(tcx.sess, span, E0025,
                    "field `{}` bound multiple times in the pattern",
                    token::get_ident(field.ident));
                span_note!(tcx.sess, *occupied.get(),
                    "field `{}` previously bound here",
                    token::get_ident(field.ident));
                ty::mk_err()
            }
            Vacant(vacant) => {
                vacant.set(span);
                field_type_map.get(&field.ident.name).cloned()
                    .unwrap_or_else(|| {
                        span_err!(tcx.sess, span, E0026,
                            "struct `{}` does not have a field named `{}`",
                            ty::item_path_str(tcx, struct_id),
                            token::get_ident(field.ident));
                        ty::mk_err()
                    })
            }
        };

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
