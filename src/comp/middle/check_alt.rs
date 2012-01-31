
import syntax::ast::*;
import syntax::ast_util::{variant_def_ids, dummy_sp, compare_lit_exprs,
        lit_expr_eq, unguarded_pat};
import syntax::codemap::span;
import pat_util::*;
import syntax::visit;
import option::{some, none};
import driver::session::session;
import middle::ty;
import middle::ty::*;

fn check_crate(tcx: ty::ctxt, crate: @crate) {
    let v =
        @{visit_expr: bind check_expr(tcx, _, _, _),
          visit_local: bind check_local(tcx, _, _, _)
             with *visit::default_visitor::<()>()};
    visit::visit_crate(*crate, (), visit::mk_vt(v));
    tcx.sess.abort_if_errors();
}

fn check_expr(tcx: ty::ctxt, ex: @expr, &&s: (), v: visit::vt<()>) {
    visit::visit_expr(ex, s, v);
    alt ex.node {
        expr_alt(scrut, arms) {
            check_arms(tcx, ex.span, scrut,
                       pat_util::normalize_arms(tcx, arms));
        }
        _ { }
    }
}

fn check_arms(tcx: ty::ctxt, sp:span, scrut: @expr, arms: [arm]) {
    let i = 0;
    let scrut_ty = expr_ty(tcx, scrut);
    /* (Could both checks be done in a single pass?) */

    /* Check for unreachable patterns */
    for arm: arm in arms {
        for arm_pat: @pat in arm.pats {
            let reachable = true;
            let j = 0;
            while j < i {
                if option::is_none(arms[j].guard) {
                    for prev_pat: @pat in arms[j].pats {
                        if pattern_supersedes(tcx, prev_pat, arm_pat) {
                            reachable = false;
                        }
                    }
                }
                j += 1;
            }
            if !reachable {
                tcx.sess.span_err(arm_pat.span, "unreachable pattern");
            }
        }
        i += 1;
    }

    /* Check for exhaustiveness */

    check_exhaustive(tcx, sp, scrut_ty,
       vec::concat(vec::filter_map(arms, unguarded_pat)));
}

// Precondition: patterns have been normalized
// (not checked statically yet)
fn check_exhaustive(tcx: ty::ctxt, sp:span, scrut_ty:ty::t, pats:[@pat]) {
    let represented : [def_id] = [];
    /* Determine the type of the scrutinee */
    /* If it's not an enum, exit (bailing out on checking non-enum alts
       for now) */
    /* Otherwise, get the list of variants and make sure each one is
     represented. Then recurse on the columns. */

    let ty_def_id = alt ty::struct(tcx, scrut_ty) {
            ty_enum(id, _) { id }
            _ { ret; } };

    let variants = *enum_variants(tcx, ty_def_id);
    for pat in pats {
        if !is_refutable(tcx, pat) {
                /* automatically makes this alt complete */ ret;
        }
        alt pat.node {
                // want the def_id for the constructor
            pat_enum(id,_) {
                alt tcx.def_map.find(pat.id) {
                    some(def_variant(_, variant_def_id)) {
                        represented += [variant_def_id];
                    }
                    _ { tcx.sess.span_bug(pat.span, "check_exhaustive:
                          pat_tag not bound to a variant"); }
                }
            }
            _ { tcx.sess.span_bug(pat.span, "check_exhaustive: ill-typed \
                  pattern");   // we know this has enum type,
            }                  // so anything else should be impossible
         }
    }
    fn not_represented(v: [def_id], &&vinfo: variant_info) -> bool {
        !vec::member(vinfo.id, v)
    }
    // Could be more efficient (bitvectors?)
    alt vec::find(variants, bind not_represented(represented,_)) {
        some(bad) {
        // complain
        // TODO: give examples of cases that aren't covered
            tcx.sess.note("Patterns not covered include:");
            tcx.sess.note(bad.name);
            tcx.sess.span_err(sp, "Non-exhaustive pattern");
        }
        _ {}
    }
    // Otherwise, check subpatterns
    // inefficient
    for variant in variants {
        // rows consists of the argument list for each pat that's an enum
        let rows : [[@pat]] = [];
        for pat in pats {
            alt pat.node {
               pat_enum(id, args) {
                  alt tcx.def_map.find(pat.id) {
                      some(def_variant(_,variant_id))
                        if variant_id == variant.id { rows += [args]; }
                      _ { }
                  }
               }
               _ {}
            }
        }
        if check vec::is_not_empty(rows) {
             let i = 0u;
             for it in rows[0] {
                let column = [it];
                // Annoying -- see comment in
                // tstate::states::find_pre_post_state_loop
                check vec::is_not_empty(rows);
                for row in vec::tail(rows) {
                  column += [row[i]];
                }
                check_exhaustive(tcx, sp, pat_ty(tcx, it), column);
                i += 1u;
             }
        }
        // This shouldn't actually happen, since there were no
        // irrefutable patterns if we got here.
        else { cont; }
    }
}

fn pattern_supersedes(tcx: ty::ctxt, a: @pat, b: @pat) -> bool {
    fn patterns_supersede(tcx: ty::ctxt, as: [@pat], bs: [@pat]) -> bool {
        let i = 0;
        for a: @pat in as {
            if !pattern_supersedes(tcx, a, bs[i]) { ret false; }
            i += 1;
        }
        ret true;
    }
    fn field_patterns_supersede(tcx: ty::ctxt, fas: [field_pat],
                                fbs: [field_pat]) -> bool {
        let wild = @{id: 0, node: pat_wild, span: dummy_sp()};
        for fa: field_pat in fas {
            let pb = wild;
            for fb: field_pat in fbs {
                if fa.ident == fb.ident { pb = fb.pat; }
            }
            if !pattern_supersedes(tcx, fa.pat, pb) { ret false; }
        }
        ret true;
    }

    alt a.node {
      pat_ident(_, some(p)) { pattern_supersedes(tcx, p, b) }
      pat_wild | pat_ident(_, none) { true }
      pat_lit(la) {
        alt b.node {
          pat_lit(lb) { lit_expr_eq(la, lb) }
          _ { false }
        }
      }
      pat_enum(va, suba) {
        alt b.node {
          pat_enum(vb, subb) {
            tcx.def_map.get(a.id) == tcx.def_map.get(b.id) &&
                patterns_supersede(tcx, suba, subb)
          }
          _ { false }
        }
      }
      pat_rec(suba, _) {
        alt b.node {
          pat_rec(subb, _) { field_patterns_supersede(tcx, suba, subb) }
          _ { false }
        }
      }
      pat_tup(suba) {
        alt b.node {
          pat_tup(subb) { patterns_supersede(tcx, suba, subb) }
          _ { false }
        }
      }
      pat_box(suba) {
        alt b.node {
          pat_box(subb) { pattern_supersedes(tcx, suba, subb) }
          _ { pattern_supersedes(tcx, suba, b) }
        }
      }
      pat_uniq(suba) {
        alt b.node {
          pat_uniq(subb) { pattern_supersedes(tcx, suba, subb) }
          _ { pattern_supersedes(tcx, suba, b) }
        }
      }
      pat_range(begina, enda) {
        alt b.node {
          pat_lit(lb) {
            compare_lit_exprs(begina, lb) <= 0 &&
            compare_lit_exprs(enda, lb) >= 0
          }
          pat_range(beginb, endb) {
            compare_lit_exprs(begina, beginb) <= 0 &&
            compare_lit_exprs(enda, endb) >= 0
          }
          _ { false }
        }
      }
    }
}

fn check_local(tcx: ty::ctxt, loc: @local, &&s: (), v: visit::vt<()>) {
    visit::visit_local(loc, s, v);
    if is_refutable(tcx, loc.node.pat) {
        tcx.sess.span_err(loc.node.pat.span,
                          "refutable pattern in local binding");
    }
}

fn is_refutable(tcx: ty::ctxt, pat: @pat) -> bool {
    alt normalize_pat(tcx, pat).node {
      pat_box(sub) | pat_uniq(sub) | pat_ident(_, some(sub)) {
        is_refutable(tcx, sub)
      }
      pat_wild | pat_ident(_, none) { false }
      pat_lit(_) { true }
      pat_rec(fields, _) {
        for it: field_pat in fields {
            if is_refutable(tcx, it.pat) { ret true; }
        }
        false
      }
      pat_tup(elts) {
        for elt in elts { if is_refutable(tcx, elt) { ret true; } }
        false
      }
      pat_enum(_, args) {
        let vdef = variant_def_ids(tcx.def_map.get(pat.id));
        if vec::len(*ty::enum_variants(tcx, vdef.enm)) != 1u { ret true; }
        for p: @pat in args { if is_refutable(tcx, p) { ret true; } }
        false
      }
      pat_range(_, _) { true }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
