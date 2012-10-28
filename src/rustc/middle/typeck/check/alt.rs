use syntax::print::pprust;
use syntax::ast_util::{walk_pat};
use pat_util::{pat_is_variant};

fn check_alt(fcx: @fn_ctxt,
             expr: @ast::expr,
             discrim: @ast::expr,
             arms: ~[ast::arm]) -> bool {
    let tcx = fcx.ccx.tcx;
    let mut bot;

    let pattern_ty = fcx.infcx().next_ty_var();
    bot = check_expr_with(fcx, discrim, pattern_ty);
    let is_lvalue = ty::expr_is_lval(tcx, fcx.ccx.method_map, discrim);

    // Typecheck the patterns first, so that we get types for all the
    // bindings.
    for arms.each |arm| {
        let pcx = {
            fcx: fcx,
            map: pat_id_map(tcx.def_map, arm.pats[0]),
            alt_region: ty::re_scope(expr.id),
            block_region: ty::re_scope(arm.body.node.id)
        };

        for arm.pats.each |p| { check_pat(pcx, *p, pattern_ty);}
        check_legality_of_move_bindings(fcx,
                                        is_lvalue,
                                        arm.guard.is_some(),
                                        arm.pats);
    }

    // Now typecheck the blocks.
    let mut result_ty = fcx.infcx().next_ty_var();
    let mut arm_non_bot = false;
    for arms.each |arm| {
        match arm.guard {
          Some(e) => { check_expr_with(fcx, e, ty::mk_bool(tcx)); },
          None => ()
        }
        if !check_block(fcx, arm.body) { arm_non_bot = true; }
        let bty = fcx.node_ty(arm.body.node.id);
        demand::suptype(fcx, arm.body.span, result_ty, bty);
    }
    bot |= !arm_non_bot;
    if !arm_non_bot { result_ty = ty::mk_bot(tcx); }
    fcx.write_ty(expr.id, result_ty);
    return bot;
}

fn check_legality_of_move_bindings(fcx: @fn_ctxt,
                                   is_lvalue: bool,
                                   has_guard: bool,
                                   pats: &[@ast::pat])
{
    let tcx = fcx.tcx();
    let def_map = tcx.def_map;
    let mut by_ref = None;
    let mut any_by_move = false;
    for pats.each |pat| {
        do pat_util::pat_bindings(def_map, *pat) |bm, _id, span, _path| {
            match bm {
                ast::bind_by_ref(_) | ast::bind_by_implicit_ref => {
                    by_ref = Some(span);
                }
                ast::bind_by_move => {
                    any_by_move = true;
                }
                _ => { }
            }
        }
    }

    if !any_by_move { return; } // pointless micro-optimization
    for pats.each |pat| {
        do walk_pat(*pat) |p| {
            if !pat_is_variant(def_map, p) {
                match p.node {
                    ast::pat_ident(ast::bind_by_move, _, sub) => {
                        // check legality of moving out of the enum
                        if sub.is_some() {
                            tcx.sess.span_err(
                                p.span,
                                ~"cannot bind by-move with sub-bindings");
                        } else if has_guard {
                            tcx.sess.span_err(
                                p.span,
                                ~"cannot bind by-move into a pattern guard");
                        } else if by_ref.is_some() {
                            tcx.sess.span_err(
                                p.span,
                                ~"cannot bind by-move and by-ref \
                                  in the same pattern");
                            tcx.sess.span_note(
                                by_ref.get(),
                                ~"by-ref binding occurs here");
                        } else if is_lvalue {
                            tcx.sess.span_err(
                                p.span,
                                ~"cannot bind by-move when \
                                  matching an lvalue");
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}


type pat_ctxt = {
    fcx: @fn_ctxt,
    map: PatIdMap,
    alt_region: ty::Region,   // Region for the alt as a whole
    block_region: ty::Region, // Region for the block of the arm
};

fn check_pat_variant(pcx: pat_ctxt, pat: @ast::pat, path: @ast::path,
                     subpats: Option<~[@ast::pat]>, expected: ty::t) {

    // Typecheck the path.
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    // Lookup the enum and variant def ids:
    let v_def = lookup_def(pcx.fcx, path.span, pat.id);
    let v_def_ids = ast_util::variant_def_ids(v_def);

    // Assign the pattern the type of the *enum*, not the variant.
    let enum_tpt = ty::lookup_item_type(tcx, v_def_ids.enm);
    instantiate_path(pcx.fcx, path, enum_tpt, pat.span, pat.id,
                     pcx.block_region);

    // structure_of requires type variables to be resolved.
    // So when we pass in <expected>, it's an error if it
    // contains type variables.

    // Take the enum type params out of `expected`.
    match structure_of(pcx.fcx, pat.span, expected) {
      ty::ty_enum(_, ref expected_substs) => {
        // check that the type of the value being matched is a subtype
        // of the type of the pattern:
        let pat_ty = fcx.node_ty(pat.id);
        demand::suptype(fcx, pat.span, pat_ty, expected);

        // Get the expected types of the arguments.
        let arg_types = {
            let vinfo =
                ty::enum_variant_with_id(
                    tcx, v_def_ids.enm, v_def_ids.var);
            vinfo.args.map(|t| { ty::subst(tcx, expected_substs, *t) })
        };
        let arg_len = arg_types.len(), subpats_len = match subpats {
            None => arg_len,
            Some(ps) => ps.len()
        };
        if arg_len > 0 {
            // N-ary variant.
            if arg_len != subpats_len {
                let s = fmt!("this pattern has %u field%s, but the \
                              corresponding variant has %u field%s",
                             subpats_len,
                             if subpats_len == 1u { ~"" } else { ~"s" },
                             arg_len,
                             if arg_len == 1u { ~"" } else { ~"s" });
                tcx.sess.span_fatal(pat.span, s);
            }

            do subpats.iter() |pats| {
                for vec::each2(*pats, arg_types) |subpat, arg_ty| {
                  check_pat(pcx, *subpat, *arg_ty);
                }
            };
        } else if subpats_len > 0 {
            tcx.sess.span_fatal
                (pat.span, fmt!("this pattern has %u field%s, \
                                 but the corresponding variant has no fields",
                                subpats_len,
                                if subpats_len == 1u { ~"" }
                                else { ~"s" }));
        }
      }
      _ => {
        tcx.sess.span_fatal
            (pat.span,
             fmt!("mismatched types: expected enum but found `%s`",
                  fcx.infcx().ty_to_str(expected)));
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
fn check_struct_pat_fields(pcx: pat_ctxt,
                           span: span,
                           path: @ast::path,
                           fields: ~[ast::field_pat],
                           class_fields: ~[ty::field_ty],
                           class_id: ast::def_id,
                           substitutions: &ty::substs,
                           etc: bool) {
    let tcx = pcx.fcx.ccx.tcx;

    // Index the class fields.
    let field_map = std::map::HashMap();
    for class_fields.eachi |i, class_field| {
        field_map.insert(class_field.ident, i);
    }

    // Typecheck each field.
    let found_fields = std::map::HashMap();
    for fields.each |field| {
        match field_map.find(field.ident) {
            Some(index) => {
                let class_field = class_fields[index];
                let field_type = ty::lookup_field_type(tcx,
                                                       class_id,
                                                       class_field.id,
                                                       substitutions);
                check_pat(pcx, field.pat, field_type);
                found_fields.insert(index, ());
            }
            None => {
                let name = pprust::path_to_str(path, tcx.sess.intr());
                tcx.sess.span_err(span,
                                  fmt!("struct `%s` does not have a field
                                        named `%s`", name,
                                       tcx.sess.str_of(field.ident)));
            }
        }
    }

    // Report an error if not all the fields were specified.
    if !etc {
        for class_fields.eachi |i, field| {
            if found_fields.contains_key(i) {
                loop;
            }
            tcx.sess.span_err(span,
                              fmt!("pattern does not mention field `%s`",
                                   tcx.sess.str_of(field.ident)));
        }
    }
}

fn check_struct_pat(pcx: pat_ctxt, pat_id: ast::node_id, span: span,
                    expected: ty::t, path: @ast::path,
                    fields: ~[ast::field_pat], etc: bool,
                    class_id: ast::def_id, substitutions: &ty::substs) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    let class_fields = ty::lookup_class_fields(tcx, class_id);

    // Check to ensure that the struct is the one specified.
    match tcx.def_map.find(pat_id) {
        Some(ast::def_class(supplied_def_id))
                if supplied_def_id == class_id => {
            // OK.
        }
        Some(ast::def_class(*)) | Some(ast::def_variant(*)) => {
            let name = pprust::path_to_str(path, tcx.sess.intr());
            tcx.sess.span_err(span,
                              fmt!("mismatched types: expected `%s` but \
                                    found `%s`",
                                   fcx.infcx().ty_to_str(expected),
                                   name));
        }
        _ => {
            tcx.sess.span_bug(span, ~"resolve didn't write in class");
        }
    }

    // Forbid pattern-matching structs with destructors.
    if ty::has_dtor(tcx, class_id) {
        tcx.sess.span_err(span, ~"deconstructing struct not allowed in \
                                  pattern (it has a destructor)");
    }

    check_struct_pat_fields(pcx, span, path, fields, class_fields, class_id,
                            substitutions, etc);
}

fn check_struct_like_enum_variant_pat(pcx: pat_ctxt,
                                      pat_id: ast::node_id,
                                      span: span,
                                      expected: ty::t,
                                      path: @ast::path,
                                      fields: ~[ast::field_pat],
                                      etc: bool,
                                      enum_id: ast::def_id,
                                      substitutions: &ty::substs) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    // Find the variant that was specified.
    match tcx.def_map.find(pat_id) {
        Some(ast::def_variant(found_enum_id, variant_id))
                if found_enum_id == enum_id => {
            // Get the struct fields from this struct-like enum variant.
            let class_fields = ty::lookup_class_fields(tcx, variant_id);

            check_struct_pat_fields(pcx, span, path, fields, class_fields,
                                    variant_id, substitutions, etc);
        }
        Some(ast::def_class(*)) | Some(ast::def_variant(*)) => {
            let name = pprust::path_to_str(path, tcx.sess.intr());
            tcx.sess.span_err(span,
                              fmt!("mismatched types: expected `%s` but \
                                    found `%s`",
                                   fcx.infcx().ty_to_str(expected),
                                   name));
        }
        _ => {
            tcx.sess.span_bug(span, ~"resolve didn't write in variant");
        }
    }
}

// Pattern checking is top-down rather than bottom-up so that bindings get
// their types immediately.
fn check_pat(pcx: pat_ctxt, pat: @ast::pat, expected: ty::t) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    match pat.node {
      ast::pat_wild => {
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_lit(lt) => {
        check_expr_with(fcx, lt, expected);
        fcx.write_ty(pat.id, fcx.expr_ty(lt));
      }
      ast::pat_range(begin, end) => {
        check_expr_with(fcx, begin, expected);
        check_expr_with(fcx, end, expected);
        let b_ty =
            fcx.infcx().resolve_type_vars_if_possible(fcx.expr_ty(begin));
        let e_ty =
            fcx.infcx().resolve_type_vars_if_possible(fcx.expr_ty(end));
        debug!("pat_range beginning type: %?", b_ty);
        debug!("pat_range ending type: %?", e_ty);
        if !require_same_types(
            tcx, Some(fcx.infcx()), false, pat.span, b_ty, e_ty,
            || ~"mismatched types in range")
        {
            // no-op
        } else if !ty::type_is_numeric(b_ty) {
            tcx.sess.span_err(pat.span, ~"non-numeric type used in range");
        } else if !valid_range_bounds(fcx.ccx, begin, end) {
            tcx.sess.span_err(begin.span, ~"lower range bound must be less \
                                           than upper");
        }
        fcx.write_ty(pat.id, b_ty);
      }
      ast::pat_ident(bm, name, sub) if !pat_is_variant(tcx.def_map, pat) => {
        let vid = lookup_local(fcx, pat.span, pat.id);
        let mut typ = ty::mk_var(tcx, vid);

        match bm {
          ast::bind_by_ref(mutbl) => {
            // if the binding is like
            //    ref x | ref const x | ref mut x
            // then the type of x is &M T where M is the mutability
            // and T is the expected type
            let region_var =
                fcx.infcx().next_region_var_with_lb(
                    pat.span, pcx.block_region);
            let mt = {ty: expected, mutbl: mutbl};
            let region_ty = ty::mk_rptr(tcx, region_var, mt);
            demand::eqtype(fcx, pat.span, region_ty, typ);
          }
          // otherwise the type of x is the expected type T
          ast::bind_by_value => {
            demand::eqtype(fcx, pat.span, expected, typ);
          }
          ast::bind_by_move => {
            demand::eqtype(fcx, pat.span, expected, typ);
          }
          ast::bind_by_implicit_ref => {
            demand::eqtype(fcx, pat.span, expected, typ);
          }
        }

        let canon_id = pcx.map.get(ast_util::path_to_ident(name));
        if canon_id != pat.id {
            let tv_id = lookup_local(fcx, pat.span, canon_id);
            let ct = ty::mk_var(tcx, tv_id);
            demand::eqtype(fcx, pat.span, ct, typ);
        }
        fcx.write_ty(pat.id, typ);
        match sub {
          Some(p) => check_pat(pcx, p, expected),
          _ => ()
        }
      }
      ast::pat_ident(_, path, _) => {
        check_pat_variant(pcx, pat, path, Some(~[]), expected);
      }
      ast::pat_enum(path, subpats) => {
        check_pat_variant(pcx, pat, path, subpats, expected);
      }
      ast::pat_rec(fields, etc) => {
        let ex_fields = match structure_of(fcx, pat.span, expected) {
          ty::ty_rec(fields) => fields,
          _ => {
            tcx.sess.span_fatal
                (pat.span,
                fmt!("mismatched types: expected `%s` but found record",
                     fcx.infcx().ty_to_str(expected)));
          }
        };
        let f_count = vec::len(fields);
        let ex_f_count = vec::len(ex_fields);
        if ex_f_count < f_count || !etc && ex_f_count > f_count {
            tcx.sess.span_fatal
                (pat.span, fmt!("mismatched types: expected a record \
                      with %u fields, found one with %u \
                      fields",
                                ex_f_count, f_count));
        }

        for fields.each |f| {
            match vec::find(ex_fields, |a| f.ident == a.ident) {
              Some(field) => {
                check_pat(pcx, f.pat, field.mt.ty);
              }
              None => {
                tcx.sess.span_fatal(pat.span,
                                    fmt!("mismatched types: did not \
                                          expect a record with a field `%s`",
                                          tcx.sess.str_of(f.ident)));
              }
            }
        }
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_struct(path, fields, etc) => {
        // Grab the class data that we care about.
        let structure = structure_of(fcx, pat.span, expected);
        match structure {
            ty::ty_class(cid, ref substs) => {
                check_struct_pat(pcx, pat.id, pat.span, expected, path,
                                 fields, etc, cid, substs);
            }
            ty::ty_enum(eid, ref substs) => {
                check_struct_like_enum_variant_pat(
                    pcx, pat.id, pat.span, expected, path, fields, etc, eid,
                    substs);
            }
            _ => {
                // XXX: This should not be fatal.
                tcx.sess.span_fatal(pat.span,
                                    fmt!("mismatched types: expected `%s` \
                                          but found struct",
                                         fcx.infcx().ty_to_str(expected)));
            }
        }

        // Finally, write in the type.
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_tup(elts) => {
        let ex_elts = match structure_of(fcx, pat.span, expected) {
          ty::ty_tup(elts) => elts,
          _ => {
            tcx.sess.span_fatal
                (pat.span,
                 fmt!("mismatched types: expected `%s`, found tuple",
                      fcx.infcx().ty_to_str(expected)));
          }
        };
        let e_count = vec::len(elts);
        if e_count != vec::len(ex_elts) {
            tcx.sess.span_fatal
                (pat.span, fmt!("mismatched types: expected a tuple \
                      with %u fields, found one with %u \
                      fields", vec::len(ex_elts), e_count));
        }
        let mut i = 0u;
        for elts.each |elt| {
            check_pat(pcx, *elt, ex_elts[i]);
            i += 1u;
        }

        fcx.write_ty(pat.id, expected);
      }
      ast::pat_box(inner) => {
        match structure_of(fcx, pat.span, expected) {
          ty::ty_box(e_inner) => {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ => {
            tcx.sess.span_fatal(
                pat.span,
                ~"mismatched types: expected `" +
                fcx.infcx().ty_to_str(expected) +
                ~"` found box");
          }
        }
      }
      ast::pat_uniq(inner) => {
        match structure_of(fcx, pat.span, expected) {
          ty::ty_uniq(e_inner) => {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ => {
            tcx.sess.span_fatal(
                pat.span,
                ~"mismatched types: expected `" +
                fcx.infcx().ty_to_str(expected) +
                ~"` found uniq");
          }
        }
      }
      ast::pat_region(inner) => {
        match structure_of(fcx, pat.span, expected) {
          ty::ty_rptr(_, e_inner) => {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ => {
            tcx.sess.span_fatal(
                pat.span,
                ~"mismatched types: expected `" +
                fcx.infcx().ty_to_str(expected) +
                ~"` found borrowed pointer");
          }
        }
      }

    }
}

