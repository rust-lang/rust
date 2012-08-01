import middle::typeck::infer::methods; // next_ty_var,
                                       // resolve_type_vars_if_possible
import syntax::print::pprust;

fn check_alt(fcx: @fn_ctxt,
             expr: @ast::expr,
             discrim: @ast::expr,
             arms: ~[ast::arm]) -> bool {
    let tcx = fcx.ccx.tcx;
    let mut bot;

    let pattern_ty = fcx.infcx.next_ty_var();
    bot = check_expr_with(fcx, discrim, pattern_ty);

    // Typecheck the patterns first, so that we get types for all the
    // bindings.
    for arms.each |arm| {
        let pcx = {
            fcx: fcx,
            map: pat_id_map(tcx.def_map, arm.pats[0]),
            alt_region: ty::re_scope(expr.id),
            block_region: ty::re_scope(arm.body.node.id),
            pat_region: ty::re_scope(expr.id)
        };

        for arm.pats.each |p| { check_pat(pcx, p, pattern_ty);}
    }
    // Now typecheck the blocks.
    let mut result_ty = fcx.infcx.next_ty_var();
    let mut arm_non_bot = false;
    for arms.each |arm| {
        alt arm.guard {
          some(e) { check_expr_with(fcx, e, ty::mk_bool(tcx)); }
          none { }
        }
        if !check_block(fcx, arm.body) { arm_non_bot = true; }
        let bty = fcx.node_ty(arm.body.node.id);
        demand::suptype(fcx, arm.body.span, result_ty, bty);
    }
    bot |= !arm_non_bot;
    if !arm_non_bot { result_ty = ty::mk_bot(tcx); }
    fcx.write_ty(expr.id, result_ty);
    ret bot;
}

type pat_ctxt = {
    fcx: @fn_ctxt,
    map: pat_id_map,
    alt_region: ty::region,
    block_region: ty::region,
    /* Equal to either alt_region or block_region. */
    pat_region: ty::region
};

fn check_pat_variant(pcx: pat_ctxt, pat: @ast::pat, path: @ast::path,
                     subpats: option<~[@ast::pat]>, expected: ty::t) {

    // Typecheck the path.
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    // Lookup the enum and variant def ids:
    let v_def = lookup_def(pcx.fcx, path.span, pat.id);
    let v_def_ids = ast_util::variant_def_ids(v_def);

    // Assign the pattern the type of the *enum*, not the variant.
    let enum_tpt = ty::lookup_item_type(tcx, v_def_ids.enm);
    instantiate_path(pcx.fcx, path, enum_tpt, pat.span, pat.id);

    // Take the enum type params out of `expected`.
    alt structure_of(pcx.fcx, pat.span, expected) {
      ty::ty_enum(_, expected_substs) {
        // check that the type of the value being matched is a subtype
        // of the type of the pattern:
        let pat_ty = fcx.node_ty(pat.id);
        demand::suptype(fcx, pat.span, pat_ty, expected);

        // Get the expected types of the arguments.
        let arg_types = {
            let vinfo =
                ty::enum_variant_with_id(
                    tcx, v_def_ids.enm, v_def_ids.var);
            vinfo.args.map(|t| { ty::subst(tcx, expected_substs, t) })
        };
        let arg_len = arg_types.len(), subpats_len = alt subpats {
            none { arg_len }
            some(ps) { ps.len() }};
        if arg_len > 0u {
            // N-ary variant.
            if arg_len != subpats_len {
                let s = fmt!{"this pattern has %u field%s, but the \
                              corresponding variant has %u field%s",
                             subpats_len,
                             if subpats_len == 1u { ~"" } else { ~"s" },
                             arg_len,
                             if arg_len == 1u { ~"" } else { ~"s" }};
                tcx.sess.span_fatal(pat.span, s);
            }

            do option::iter(subpats) |pats| {
                do vec::iter2(pats, arg_types) |subpat, arg_ty| {
                  check_pat(pcx, subpat, arg_ty);
                }
            };
        } else if subpats_len > 0u {
            tcx.sess.span_fatal
                (pat.span, fmt!{"this pattern has %u field%s, \
                                 but the corresponding variant has no fields",
                                subpats_len,
                                if subpats_len == 1u { ~"" }
                                else { ~"s" }});
        }
      }
      _ {
        tcx.sess.span_fatal
            (pat.span,
             fmt!{"mismatched types: expected enum but found `%s`",
                  fcx.infcx.ty_to_str(expected)});
      }
    }
}

// Pattern checking is top-down rather than bottom-up so that bindings get
// their types immediately.
fn check_pat(pcx: pat_ctxt, pat: @ast::pat, expected: ty::t) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;

    alt pat.node {
      ast::pat_wild {
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_lit(lt) {
        check_expr_with(fcx, lt, expected);
        fcx.write_ty(pat.id, fcx.expr_ty(lt));
      }
      ast::pat_range(begin, end) {
        check_expr_with(fcx, begin, expected);
        check_expr_with(fcx, end, expected);
        let b_ty =
            fcx.infcx.resolve_type_vars_if_possible(fcx.expr_ty(begin));
        let e_ty =
            fcx.infcx.resolve_type_vars_if_possible(fcx.expr_ty(end));
        debug!{"pat_range beginning type: %?", b_ty};
        debug!{"pat_range ending type: %?", e_ty};
        if !require_same_types(
            tcx, some(fcx.infcx), pat.span, b_ty, e_ty,
            || ~"mismatched types in range") {
            // no-op
        } else if !ty::type_is_numeric(b_ty) {
            tcx.sess.span_err(pat.span, ~"non-numeric type used in range");
        } else if !valid_range_bounds(fcx.ccx, begin, end) {
            tcx.sess.span_err(begin.span, ~"lower range bound must be less \
                                           than upper");
        }
        fcx.write_ty(pat.id, b_ty);
      }
      ast::pat_ident(_, name, sub) if !pat_is_variant(tcx.def_map, pat) {
        let vid = lookup_local(fcx, pat.span, pat.id);
        let mut typ = ty::mk_var(tcx, vid);
        demand::suptype(fcx, pat.span, expected, typ);
        let canon_id = pcx.map.get(ast_util::path_to_ident(name));
        if canon_id != pat.id {
            let tv_id = lookup_local(fcx, pat.span, canon_id);
            let ct = ty::mk_var(tcx, tv_id);
            demand::suptype(fcx, pat.span, ct, typ);
        }
        fcx.write_ty(pat.id, typ);
        alt sub {
          some(p) { check_pat(pcx, p, expected); }
          _ {}
        }
      }
      ast::pat_ident(_, path, c) {
        check_pat_variant(pcx, pat, path, some(~[]), expected);
      }
      ast::pat_enum(path, subpats) {
        check_pat_variant(pcx, pat, path, subpats, expected);
      }
      ast::pat_rec(fields, etc) {
        let ex_fields = alt structure_of(fcx, pat.span, expected) {
          ty::ty_rec(fields) { fields }
          _ {
            tcx.sess.span_fatal
                (pat.span,
                fmt!{"mismatched types: expected `%s` but found record",
                     fcx.infcx.ty_to_str(expected)});
          }
        };
        let f_count = vec::len(fields);
        let ex_f_count = vec::len(ex_fields);
        if ex_f_count < f_count || !etc && ex_f_count > f_count {
            tcx.sess.span_fatal
                (pat.span, fmt!{"mismatched types: expected a record \
                      with %u fields, found one with %u \
                      fields",
                                ex_f_count, f_count});
        }
        fn matches(name: ast::ident, f: ty::field) -> bool {
            ret str::eq(*name, *f.ident);
        }
        for fields.each |f| {
            alt vec::find(ex_fields, |a| matches(f.ident, a)) {
              some(field) {
                check_pat(pcx, f.pat, field.mt.ty);
              }
              none {
                tcx.sess.span_fatal(pat.span,
                                    fmt!{"mismatched types: did not \
                                          expect a record with a field `%s`",
                                         *f.ident});
              }
            }
        }
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_tup(elts) {
        let ex_elts = alt structure_of(fcx, pat.span, expected) {
          ty::ty_tup(elts) { elts }
          _ {
            tcx.sess.span_fatal
                (pat.span,
                 fmt!{"mismatched types: expected `%s`, found tuple",
                      fcx.infcx.ty_to_str(expected)});
          }
        };
        let e_count = vec::len(elts);
        if e_count != vec::len(ex_elts) {
            tcx.sess.span_fatal
                (pat.span, fmt!{"mismatched types: expected a tuple \
                      with %u fields, found one with %u \
                      fields", vec::len(ex_elts), e_count});
        }
        let mut i = 0u;
        for elts.each |elt| {
            check_pat(pcx, elt, ex_elts[i]);
            i += 1u;
        }

        fcx.write_ty(pat.id, expected);
      }
      ast::pat_box(inner) {
        alt structure_of(fcx, pat.span, expected) {
          ty::ty_box(e_inner) {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ {
            tcx.sess.span_fatal(
                pat.span,
                ~"mismatched types: expected `" +
                fcx.infcx.ty_to_str(expected) +
                ~"` found box");
          }
        }
      }
      ast::pat_uniq(inner) {
        alt structure_of(fcx, pat.span, expected) {
          ty::ty_uniq(e_inner) {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ {
            tcx.sess.span_fatal(
                pat.span,
                ~"mismatched types: expected `" +
                fcx.infcx.ty_to_str(expected) +
                ~"` found uniq");
          }
        }
      }
    }
}

