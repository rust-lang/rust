import syntax::ast::*;
import syntax::ast_util::{variant_def_ids, dummy_sp, compare_lit_exprs,
                          lit_expr_eq};
import syntax::visit;
import option::{some, none};

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
    alt ex.node { expr_alt(_, arms) { check_arms(tcx, arms); } _ { } }
}

fn check_arms(tcx: ty::ctxt, arms: [arm]) {
    let i = 0;
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
      pat_bind(_, some(p)) { pattern_supersedes(tcx, p, b) }
      pat_wild. | pat_bind(_, none.) { true }
      pat_lit(la) {
        alt b.node {
          pat_lit(lb) { lit_expr_eq(la, lb) }
          _ { false }
        }
      }
      pat_tag(va, suba) {
        alt b.node {
          pat_tag(vb, subb) {
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
    alt pat.node {
      pat_box(sub) | pat_uniq(sub) | pat_bind(_, some(sub)) {
        is_refutable(tcx, sub)
      }
      pat_wild. | pat_bind(_, none.) { false }
      pat_lit(_) { true }
      pat_rec(fields, _) {
        for field: field_pat in fields {
            if is_refutable(tcx, field.pat) { ret true; }
        }
        false
      }
      pat_tup(elts) {
        for elt in elts { if is_refutable(tcx, elt) { ret true; } }
        false
      }
      pat_tag(_, args) {
        let vdef = variant_def_ids(tcx.def_map.get(pat.id));
        if vec::len(*ty::tag_variants(tcx, vdef.tg)) != 1u { ret true; }
        for p: @pat in args { if is_refutable(tcx, p) { ret true; } }
        false
      }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
