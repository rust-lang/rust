import syntax::ast::*;
import syntax::visit;

fn check_crate(tcx: &ty::ctxt, crate: &@crate) {
    let v =
        @{visit_expr: bind check_expr(tcx, _, _, _),
          visit_local: bind check_local(tcx, _, _, _)
             with *visit::default_visitor::<()>()};
    visit::visit_crate(*crate, (), visit::mk_vt(v));
    tcx.sess.abort_if_errors();
}

fn check_expr(tcx: &ty::ctxt, ex: &@expr, s: &(), v: &visit::vt<()>) {
    visit::visit_expr(ex, s, v);
    alt ex.node { expr_alt(_, arms) { check_arms(tcx, arms); } _ { } }
}

fn check_arms(tcx: &ty::ctxt, arms: &[arm]) {
    let i = 0;
    for arm: arm in arms {
        for arm_pat: @pat in arm.pats {
            let reachable = true;
            let j = 0;
            while j < i {
                if std::option::is_none(arms[j].guard) {
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

fn pattern_supersedes(tcx: &ty::ctxt, a: &@pat, b: &@pat) -> bool {
    fn patterns_supersede(tcx: &ty::ctxt, as: &[@pat], bs: &[@pat]) -> bool {
        let i = 0;
        for a: @pat in as {
            if !pattern_supersedes(tcx, a, bs[i]) { ret false; }
            i += 1;
        }
        ret true;
    }
    fn field_patterns_supersede(tcx: &ty::ctxt, fas: &[field_pat],
                                fbs: &[field_pat]) -> bool {
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
      pat_wild. | pat_bind(_) { ret true; }
      pat_lit(la) {
        alt b.node {
          pat_lit(lb) { ret util::common::lit_eq(la, lb); }
          _ { ret false; }
        }
      }
      pat_tag(va, suba) {
        alt b.node {
          pat_tag(vb, subb) {
            ret tcx.def_map.get(a.id) == tcx.def_map.get(b.id) &&
                    patterns_supersede(tcx, suba, subb);
          }
          _ { ret false; }
        }
      }
      pat_rec(suba, _) {
        alt b.node {
          pat_rec(subb, _) { ret field_patterns_supersede(tcx, suba, subb); }
          _ { ret false; }
        }
      }
      pat_tup(suba) {
        alt b.node {
          pat_tup(subb) { ret patterns_supersede(tcx, suba, subb); }
          _ { ret false; }
        }
      }
      pat_box(suba) {
        alt b.node {
          pat_box(subb) { ret pattern_supersedes(tcx, suba, subb); }
          _ { ret pattern_supersedes(tcx, suba, b); }
        }
      }
    }
}

fn check_local(tcx: &ty::ctxt, loc: &@local, s: &(), v: &visit::vt<()>) {
    visit::visit_local(loc, s, v);
    if is_refutable(tcx, loc.node.pat) {
        tcx.sess.span_err(loc.node.pat.span,
                          "refutable pattern in local binding");
    }
}

fn is_refutable(tcx: &ty::ctxt, pat: &@pat) -> bool {
    alt pat.node {
      pat_wild. | pat_bind(_) { ret false; }
      pat_lit(_) { ret true; }
      pat_box(sub) { ret is_refutable(tcx, sub); }
      pat_rec(fields, _) {
        for field: field_pat in fields {
            if is_refutable(tcx, field.pat) { ret true; }
        }
        ret false;
      }
      pat_tup(elts) {
        for elt in elts { if is_refutable(tcx, elt) { ret true; } }
        ret false;
      }
      pat_tag(_, args) {
        let vdef = variant_def_ids(tcx.def_map.get(pat.id));
        if std::vec::len(ty::tag_variants(tcx, vdef.tg)) != 1u { ret true; }
        for p: @pat in args { if is_refutable(tcx, p) { ret true; } }
        ret false;
      }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
