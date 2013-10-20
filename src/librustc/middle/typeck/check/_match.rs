// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::pat_util::{PatIdMap, pat_id_map, pat_is_binding, pat_is_const};
use middle::ty;
use middle::typeck::check::demand;
use middle::typeck::check::{check_block, check_expr_has_type, FnCtxt};
use middle::typeck::check::{instantiate_path, lookup_def};
use middle::typeck::check::{structure_of, valid_range_bounds};
use middle::typeck::infer;
use middle::typeck::require_same_types;

use std::hashmap::{HashMap, HashSet};
use syntax::ast;
use syntax::ast_util;
use syntax::parse::token;
use syntax::codemap::Span;
use syntax::print::pprust;

pub fn check_match(fcx: @mut FnCtxt,
                   expr: @ast::Expr,
                   discrim: @ast::Expr,
                   arms: &[ast::Arm]) {
    let tcx = fcx.ccx.tcx;

    let discrim_ty = fcx.infcx().next_ty_var();
    check_expr_has_type(fcx, discrim, discrim_ty);

    // Typecheck the patterns first, so that we get types for all the
    // bindings.
    for arm in arms.iter() {
        let mut pcx = pat_ctxt {
            fcx: fcx,
            map: pat_id_map(tcx.def_map, arm.pats[0]),
        };

        for p in arm.pats.iter() { check_pat(&mut pcx, *p, discrim_ty);}
    }

    // The result of the match is the common supertype of all the
    // arms. Start out the value as bottom, since it's the, well,
    // bottom the type lattice, and we'll be moving up the lattice as
    // we process each arm. (Note that any match with 0 arms is matching
    // on any empty type and is therefore unreachable; should the flow
    // of execution reach it, we will fail, so bottom is an appropriate
    // type in that case)
    let mut result_ty = ty::mk_bot();

    // Now typecheck the blocks.
    let mut saw_err = ty::type_is_error(discrim_ty);
    for arm in arms.iter() {
        let mut guard_err = false;
        let mut guard_bot = false;
        match arm.guard {
          Some(e) => {
              check_expr_has_type(fcx, e, ty::mk_bool());
              let e_ty = fcx.expr_ty(e);
              if ty::type_is_error(e_ty) {
                  guard_err = true;
              }
              else if ty::type_is_bot(e_ty) {
                  guard_bot = true;
              }
          },
          None => ()
        }
        check_block(fcx, &arm.body);
        let bty = fcx.node_ty(arm.body.id);
        saw_err = saw_err || ty::type_is_error(bty);
        if guard_err {
            fcx.write_error(arm.body.id);
            saw_err = true;
        }
        else if guard_bot {
            fcx.write_bot(arm.body.id);
        }

        result_ty =
            infer::common_supertype(
                fcx.infcx(),
                infer::MatchExpression(expr.span),
                true, // result_ty is "expected" here
                result_ty,
                bty);
    }

    if saw_err {
        result_ty = ty::mk_err();
    } else if ty::type_is_bot(discrim_ty) {
        result_ty = ty::mk_bot();
    }

    fcx.write_ty(expr.id, result_ty);
}

pub struct pat_ctxt {
    fcx: @mut FnCtxt,
    map: PatIdMap,
}

pub fn check_pat_variant(pcx: &pat_ctxt, pat: @ast::Pat, path: &ast::Path,
                         subpats: &Option<~[@ast::Pat]>, expected: ty::t) {

    // Typecheck the path.
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    let arg_types;
    let kind_name;

    // structure_of requires type variables to be resolved.
    // So when we pass in <expected>, it's an error if it
    // contains type variables.

    // Check to see whether this is an enum or a struct.
    match *structure_of(pcx.fcx, pat.span, expected) {
        ty::ty_enum(_, ref expected_substs) => {
            // Lookup the enum and variant def ids:
            let v_def = lookup_def(pcx.fcx, pat.span, pat.id);
            match ast_util::variant_def_ids(v_def) {
                Some((enm, var)) => {
                    // Assign the pattern the type of the *enum*, not the variant.
                    let enum_tpt = ty::lookup_item_type(tcx, enm);
                    instantiate_path(pcx.fcx,
                                     path,
                                     enum_tpt,
                                     v_def,
                                     pat.span,
                                     pat.id);

                    // check that the type of the value being matched is a subtype
                    // of the type of the pattern:
                    let pat_ty = fcx.node_ty(pat.id);
                    demand::subtype(fcx, pat.span, expected, pat_ty);

                    // Get the expected types of the arguments.
                    arg_types = {
                        let vinfo =
                            ty::enum_variant_with_id(tcx, enm, var);
                        let var_tpt = ty::lookup_item_type(tcx, var);
                        vinfo.args.map(|t| {
                            if var_tpt.generics.type_param_defs.len() ==
                                expected_substs.tps.len()
                            {
                                ty::subst(tcx, expected_substs, *t)
                            }
                            else {
                                *t // In this case, an error was already signaled
                                    // anyway
                            }
                        })
                    };

                    kind_name = "variant";
                }
                None => {
                    // See [Note-Type-error-reporting] in middle/typeck/infer/mod.rs
                    fcx.infcx().type_error_message_str_with_expected(pat.span,
                                                       |expected, actual| {
                                                       expected.map_default(~"", |e| {
                        format!("mismatched types: expected `{}` but found {}",
                             e, actual)})},
                             Some(expected), ~"a structure pattern",
                             None);
                    fcx.write_error(pat.id);
                    kind_name = "[error]";
                    arg_types = (*subpats).clone()
                                          .unwrap_or_default()
                                          .map(|_| ty::mk_err());
                }
            }
        }
        ty::ty_struct(struct_def_id, ref expected_substs) => {
            // Lookup the struct ctor def id
            let s_def = lookup_def(pcx.fcx, pat.span, pat.id);
            let s_def_id = ast_util::def_id_of_def(s_def);

            // Assign the pattern the type of the struct.
            let ctor_tpt = ty::lookup_item_type(tcx, s_def_id);
            let struct_tpt = if ty::is_fn_ty(ctor_tpt.ty) {
                ty::ty_param_bounds_and_ty {ty: ty::ty_fn_ret(ctor_tpt.ty),
                                        ..ctor_tpt}
            } else {
                ctor_tpt
            };
            instantiate_path(pcx.fcx,
                             path,
                             struct_tpt,
                             s_def,
                             pat.span,
                             pat.id);

            // Check that the type of the value being matched is a subtype of
            // the type of the pattern.
            let pat_ty = fcx.node_ty(pat.id);
            demand::subtype(fcx, pat.span, expected, pat_ty);

            // Get the expected types of the arguments.
            let class_fields = ty::struct_fields(
                tcx, struct_def_id, expected_substs);
            arg_types = class_fields.map(|field| field.mt.ty);

            kind_name = "structure";
        }
        _ => {
            // See [Note-Type-error-reporting] in middle/typeck/infer/mod.rs
            fcx.infcx().type_error_message_str_with_expected(pat.span,
                                               |expected, actual| {
                                               expected.map_default(~"", |e| {
                    format!("mismatched types: expected `{}` but found {}",
                         e, actual)})},
                    Some(expected), ~"an enum or structure pattern",
                    None);
            fcx.write_error(pat.id);
            kind_name = "[error]";
            arg_types = (*subpats).clone()
                                  .unwrap_or_default()
                                  .map(|_| ty::mk_err());
        }
    }

    let arg_len = arg_types.len();

    // Count the number of subpatterns.
    let subpats_len;
    match *subpats {
        None => subpats_len = arg_len,
        Some(ref subpats) => subpats_len = subpats.len()
    }

    let mut error_happened = false;

    if arg_len > 0 {
        // N-ary variant.
        if arg_len != subpats_len {
            let s = format!("this pattern has {} field{}, but the corresponding {} has {} field{}",
                         subpats_len,
                         if subpats_len == 1u { ~"" } else { ~"s" },
                         kind_name,
                         arg_len,
                         if arg_len == 1u { ~"" } else { ~"s" });
            tcx.sess.span_err(pat.span, s);
            error_happened = true;
        }

        if !error_happened {
            for pats in subpats.iter() {
                for (subpat, arg_ty) in pats.iter().zip(arg_types.iter()) {
                    check_pat(pcx, *subpat, *arg_ty);
                }
            }
        }
    } else if subpats_len > 0 {
        tcx.sess.span_err(pat.span,
                          format!("this pattern has {} field{}, but the corresponding {} has no \
                                fields",
                               subpats_len,
                               if subpats_len == 1u { "" } else { "s" },
                               kind_name));
        error_happened = true;
    }

    if error_happened {
        for pats in subpats.iter() {
            for pat in pats.iter() {
                check_pat(pcx, *pat, ty::mk_err());
            }
        }
    }
}

/// `path` is the AST path item naming the type of this struct.
/// `fields` is the field patterns of the struct pattern.
/// `class_fields` describes the type of each field of the struct.
/// `class_id` is the ID of the struct.
/// `substitutions` are the type substitutions applied to this struct type
/// (e.g. K,V in HashMap<K,V>).
/// `etc` is true if the pattern said '...' and false otherwise.
pub fn check_struct_pat_fields(pcx: &pat_ctxt,
                               span: Span,
                               path: &ast::Path,
                               fields: &[ast::FieldPat],
                               class_fields: ~[ty::field_ty],
                               class_id: ast::DefId,
                               substitutions: &ty::substs,
                               etc: bool) {
    let tcx = pcx.fcx.ccx.tcx;

    // Index the class fields.
    let mut field_map = HashMap::new();
    for (i, class_field) in class_fields.iter().enumerate() {
        field_map.insert(class_field.name, i);
    }

    // Typecheck each field.
    let mut found_fields = HashSet::new();
    for field in fields.iter() {
        match field_map.find(&field.ident.name) {
            Some(&index) => {
                let class_field = class_fields[index];
                let field_type = ty::lookup_field_type(tcx,
                                                       class_id,
                                                       class_field.id,
                                                       substitutions);
                check_pat(pcx, field.pat, field_type);
                found_fields.insert(index);
            }
            None => {
                let name = pprust::path_to_str(path, tcx.sess.intr());
                // Check the pattern anyway, so that attempts to look
                // up its type won't fail
                check_pat(pcx, field.pat, ty::mk_err());
                tcx.sess.span_err(span,
                    format!("struct `{}` does not have a field named `{}`",
                         name,
                         tcx.sess.str_of(field.ident)));
            }
        }
    }

    // Report an error if not all the fields were specified.
    if !etc {
        for (i, field) in class_fields.iter().enumerate() {
            if found_fields.contains(&i) {
                continue;
            }
            tcx.sess.span_err(span,
                              format!("pattern does not mention field `{}`",
                                   token::interner_get(field.name)));
        }
    }
}

pub fn check_struct_pat(pcx: &pat_ctxt, pat_id: ast::NodeId, span: Span,
                        expected: ty::t, path: &ast::Path,
                        fields: &[ast::FieldPat], etc: bool,
                        struct_id: ast::DefId,
                        substitutions: &ty::substs) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    let class_fields = ty::lookup_struct_fields(tcx, struct_id);

    // Check to ensure that the struct is the one specified.
    match tcx.def_map.find(&pat_id) {
        Some(&ast::DefStruct(supplied_def_id))
                if supplied_def_id == struct_id => {
            // OK.
        }
        Some(&ast::DefStruct(*)) | Some(&ast::DefVariant(*)) => {
            let name = pprust::path_to_str(path, tcx.sess.intr());
            tcx.sess.span_err(span,
                              format!("mismatched types: expected `{}` but found `{}`",
                                   fcx.infcx().ty_to_str(expected),
                                   name));
        }
        _ => {
            tcx.sess.span_bug(span, "resolve didn't write in struct ID");
        }
    }

    check_struct_pat_fields(pcx, span, path, fields, class_fields, struct_id,
                            substitutions, etc);
}

pub fn check_struct_like_enum_variant_pat(pcx: &pat_ctxt,
                                          pat_id: ast::NodeId,
                                          span: Span,
                                          expected: ty::t,
                                          path: &ast::Path,
                                          fields: &[ast::FieldPat],
                                          etc: bool,
                                          enum_id: ast::DefId,
                                          substitutions: &ty::substs) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    // Find the variant that was specified.
    match tcx.def_map.find(&pat_id) {
        Some(&ast::DefVariant(found_enum_id, variant_id, _))
                if found_enum_id == enum_id => {
            // Get the struct fields from this struct-like enum variant.
            let class_fields = ty::lookup_struct_fields(tcx, variant_id);

            check_struct_pat_fields(pcx, span, path, fields, class_fields,
                                    variant_id, substitutions, etc);
        }
        Some(&ast::DefStruct(*)) | Some(&ast::DefVariant(*)) => {
            let name = pprust::path_to_str(path, tcx.sess.intr());
            tcx.sess.span_err(span,
                              format!("mismatched types: expected `{}` but \
                                    found `{}`",
                                   fcx.infcx().ty_to_str(expected),
                                   name));
        }
        _ => {
            tcx.sess.span_bug(span, "resolve didn't write in variant");
        }
    }
}

// Pattern checking is top-down rather than bottom-up so that bindings get
// their types immediately.
pub fn check_pat(pcx: &pat_ctxt, pat: @ast::Pat, expected: ty::t) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    match pat.node {
      ast::PatWild => {
        fcx.write_ty(pat.id, expected);
      }
      ast::PatLit(lt) => {
        check_expr_has_type(fcx, lt, expected);
        fcx.write_ty(pat.id, fcx.expr_ty(lt));
      }
      ast::PatRange(begin, end) => {
        check_expr_has_type(fcx, begin, expected);
        check_expr_has_type(fcx, end, expected);
        let b_ty =
            fcx.infcx().resolve_type_vars_if_possible(fcx.expr_ty(begin));
        let e_ty =
            fcx.infcx().resolve_type_vars_if_possible(fcx.expr_ty(end));
        debug!("pat_range beginning type: {:?}", b_ty);
        debug!("pat_range ending type: {:?}", e_ty);
        if !require_same_types(
            tcx, Some(fcx.infcx()), false, pat.span, b_ty, e_ty,
            || ~"mismatched types in range")
        {
            // no-op
        } else if !ty::type_is_numeric(b_ty) && !ty::type_is_char(b_ty) {
            tcx.sess.span_err(pat.span, "non-numeric type used in range");
        } else {
            match valid_range_bounds(fcx.ccx, begin, end) {
                Some(false) => {
                    tcx.sess.span_err(begin.span,
                        "lower range bound must be less than upper");
                },
                None => {
                    tcx.sess.span_err(begin.span,
                        "mismatched types in range");
                },
                _ => { },
            }
        }
        fcx.write_ty(pat.id, b_ty);
      }
      ast::PatEnum(*) |
      ast::PatIdent(*) if pat_is_const(tcx.def_map, pat) => {
        let const_did = ast_util::def_id_of_def(tcx.def_map.get_copy(&pat.id));
        let const_tpt = ty::lookup_item_type(tcx, const_did);
        demand::suptype(fcx, pat.span, expected, const_tpt.ty);
        fcx.write_ty(pat.id, const_tpt.ty);
      }
      ast::PatIdent(bm, ref name, sub) if pat_is_binding(tcx.def_map, pat) => {
        let typ = fcx.local_ty(pat.span, pat.id);

        match bm {
          ast::BindByRef(mutbl) => {
            // if the binding is like
            //    ref x | ref const x | ref mut x
            // then the type of x is &M T where M is the mutability
            // and T is the expected type
            let region_var =
                fcx.infcx().next_region_var(
                    infer::PatternRegion(pat.span));
            let mt = ty::mt {ty: expected, mutbl: mutbl};
            let region_ty = ty::mk_rptr(tcx, region_var, mt);
            demand::eqtype(fcx, pat.span, region_ty, typ);
          }
          // otherwise the type of x is the expected type T
          ast::BindByValue(_) => {
            demand::eqtype(fcx, pat.span, expected, typ);
          }
        }

        let canon_id = *pcx.map.get(&ast_util::path_to_ident(name));
        if canon_id != pat.id {
            let ct = fcx.local_ty(pat.span, canon_id);
            demand::eqtype(fcx, pat.span, ct, typ);
        }
        fcx.write_ty(pat.id, typ);

        debug!("(checking match) writing type for pat id {}", pat.id);

        match sub {
          Some(p) => check_pat(pcx, p, expected),
          _ => ()
        }
      }
      ast::PatIdent(_, ref path, _) => {
        check_pat_variant(pcx, pat, path, &Some(~[]), expected);
      }
      ast::PatEnum(ref path, ref subpats) => {
        check_pat_variant(pcx, pat, path, subpats, expected);
      }
      ast::PatStruct(ref path, ref fields, etc) => {
        // Grab the class data that we care about.
        let structure = structure_of(fcx, pat.span, expected);
        let mut error_happened = false;
        match *structure {
            ty::ty_struct(cid, ref substs) => {
                check_struct_pat(pcx, pat.id, pat.span, expected, path,
                                 *fields, etc, cid, substs);
            }
            ty::ty_enum(eid, ref substs) => {
                check_struct_like_enum_variant_pat(
                    pcx, pat.id, pat.span, expected, path, *fields, etc, eid,
                    substs);
            }
            _ => {
               // See [Note-Type-error-reporting] in middle/typeck/infer/mod.rs
               fcx.infcx().type_error_message_str_with_expected(pat.span,
                                                                |expected, actual| {
                            expected.map_default(~"", |e| {
                                    format!("mismatched types: expected `{}` but found {}",
                                         e, actual)})},
                                         Some(expected), ~"a structure pattern",
                                         None);
                match tcx.def_map.find(&pat.id) {
                    Some(&ast::DefStruct(supplied_def_id)) => {
                         check_struct_pat(pcx, pat.id, pat.span, ty::mk_err(), path, *fields, etc,
                         supplied_def_id,
                         &ty::substs { self_ty: None, tps: ~[], regions: ty::ErasedRegions} );
                    }
                    _ => () // Error, but we're already in an error case
                }
                error_happened = true;
            }
        }

        // Finally, write in the type.
        if error_happened {
            fcx.write_error(pat.id);
        } else {
            fcx.write_ty(pat.id, expected);
        }
      }
      ast::PatTup(ref elts) => {
        let s = structure_of(fcx, pat.span, expected);
        let e_count = elts.len();
        match *s {
            ty::ty_tup(ref ex_elts) if e_count == ex_elts.len() => {
                for (i, elt) in elts.iter().enumerate() {
                    check_pat(pcx, *elt, ex_elts[i]);
                }
                fcx.write_ty(pat.id, expected);
            }
            _ => {
                for elt in elts.iter() {
                    check_pat(pcx, *elt, ty::mk_err());
                }
                // use terr_tuple_size if both types are tuples
                let type_error = match *s {
                    ty::ty_tup(ref ex_elts) =>
                        ty::terr_tuple_size(ty::expected_found{expected: ex_elts.len(),
                                                           found: e_count}),
                    _ => ty::terr_mismatch
                };
                // See [Note-Type-error-reporting] in middle/typeck/infer/mod.rs
                fcx.infcx().type_error_message_str_with_expected(pat.span, |expected, actual| {
                expected.map_default(~"", |e| {
                    format!("mismatched types: expected `{}` but found {}",
                                     e, actual)})}, Some(expected), ~"tuple", Some(&type_error));
                fcx.write_error(pat.id);
            }
        }
      }
      ast::PatBox(inner) => {
          check_pointer_pat(pcx, Managed, inner, pat.id, pat.span, expected);
      }
      ast::PatUniq(inner) => {
          check_pointer_pat(pcx, Send, inner, pat.id, pat.span, expected);
      }
      ast::PatRegion(inner) => {
          check_pointer_pat(pcx, Borrowed, inner, pat.id, pat.span, expected);
      }
      ast::PatVec(ref before, slice, ref after) => {
        let default_region_var =
            fcx.infcx().next_region_var(
                infer::PatternRegion(pat.span));

        let (elt_type, region_var) = match *structure_of(fcx,
                                                         pat.span,
                                                         expected) {
          ty::ty_evec(mt, vstore) => {
            let region_var = match vstore {
                ty::vstore_slice(r) => r,
                ty::vstore_box | ty::vstore_uniq | ty::vstore_fixed(_) => {
                    default_region_var
                }
            };
            (mt, region_var)
          }
          ty::ty_unboxed_vec(mt) => {
            (mt, default_region_var)
          },
          _ => {
              for &elt in before.iter() {
                  check_pat(pcx, elt, ty::mk_err());
              }
              for &elt in slice.iter() {
                  check_pat(pcx, elt, ty::mk_err());
              }
              for &elt in after.iter() {
                  check_pat(pcx, elt, ty::mk_err());
              }
              // See [Note-Type-error-reporting] in middle/typeck/infer/mod.rs
              fcx.infcx().type_error_message_str_with_expected(
                  pat.span,
                  |expected, actual| {
                      expected.map_default(~"", |e| {
                          format!("mismatched types: expected `{}` but found {}",
                               e, actual)})},
                  Some(expected),
                  ~"a vector pattern",
                  None);
              fcx.write_error(pat.id);
              return;
          }
        };
        for elt in before.iter() {
            check_pat(pcx, *elt, elt_type.ty);
        }
        match slice {
            Some(slice_pat) => {
                let slice_ty = ty::mk_evec(tcx,
                    ty::mt {ty: elt_type.ty, mutbl: elt_type.mutbl},
                    ty::vstore_slice(region_var)
                );
                check_pat(pcx, slice_pat, slice_ty);
            }
            None => ()
        }
        for elt in after.iter() {
            check_pat(pcx, *elt, elt_type.ty);
        }
        fcx.write_ty(pat.id, expected);
      }
    }
}

// Helper function to check @, ~ and & patterns
pub fn check_pointer_pat(pcx: &pat_ctxt,
                         pointer_kind: PointerKind,
                         inner: @ast::Pat,
                         pat_id: ast::NodeId,
                         span: Span,
                         expected: ty::t) {
    let fcx = pcx.fcx;
    let check_inner: &fn(ty::mt) = |e_inner| {
        check_pat(pcx, inner, e_inner.ty);
        fcx.write_ty(pat_id, expected);
    };
    match *structure_of(fcx, span, expected) {
        ty::ty_box(e_inner) if pointer_kind == Managed => {
            check_inner(e_inner);
        }
        ty::ty_uniq(e_inner) if pointer_kind == Send => {
            check_inner(e_inner);
        }
        ty::ty_rptr(_, e_inner) if pointer_kind == Borrowed => {
            check_inner(e_inner);
        }
        _ => {
            check_pat(pcx, inner, ty::mk_err());
            // See [Note-Type-error-reporting] in middle/typeck/infer/mod.rs
            fcx.infcx().type_error_message_str_with_expected(
                span,
                |expected, actual| {
                    expected.map_default(~"", |e| {
                        format!("mismatched types: expected `{}` but found {}",
                             e, actual)})},
                Some(expected),
                format!("{} pattern", match pointer_kind {
                    Managed => "an @-box",
                    Send => "a ~-box",
                    Borrowed => "an &-pointer"
                }),
                None);
            fcx.write_error(pat_id);
          }
    }
}

#[deriving(Eq)]
enum PointerKind { Managed, Send, Borrowed }

