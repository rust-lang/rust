import syntax::ast::*;
import syntax::visit;

fn check_crate(&ty::ctxt tcx, &@crate crate) {
    auto v = @rec(visit_expr=bind check_expr(tcx, _, _, _)
                  with *visit::default_visitor[()]());
    visit::visit_crate(*crate, (), visit::mk_vt(v));
    tcx.sess.abort_if_errors();
}

fn check_expr(&ty::ctxt tcx, &@expr ex, &() s, &visit::vt[()] v) {
    visit::visit_expr(ex, s, v);
    alt ex.node {
      expr_alt(_, ?arms) { check_arms(tcx, arms); }
      _ {}
    }
}

fn check_arms(&ty::ctxt tcx, &arm[] arms) {
    auto i = 0;
    for (arm arm in arms) {
        for (@pat arm_pat in arm.pats) {
            auto reachable = true;
            auto j = 0;
            while j < i {
                for (@pat prev_pat in arms.(j).pats) {
                    if pattern_supersedes(tcx, prev_pat, arm_pat) {
                        reachable = false;
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

fn pattern_supersedes(&ty::ctxt tcx, &@pat a, &@pat b) -> bool {
    fn patterns_supersede(&ty::ctxt tcx, &(@pat)[] as, &(@pat)[] bs) -> bool {
        auto i = 0;
        for (@pat a in as) {
            if !pattern_supersedes(tcx, a, bs.(i)) { ret false; }
            i += 1;
        }
        ret true;
    }
    fn field_patterns_supersede(&ty::ctxt tcx, &field_pat[] fas,
                                &field_pat[] fbs) -> bool {
        auto wild = @rec(id=0, node=pat_wild, span=rec(lo=0u, hi=0u));
        for (field_pat fa in fas) {
            auto pb = wild;
            for (field_pat fb in fbs) {
                if fa.ident == fb.ident { pb = fb.pat; }
            }
            if !pattern_supersedes(tcx, fa.pat, pb) { ret false; }
        }
        ret true;
    }

    alt a.node {
      pat_wild | pat_bind(_) { ret true; }
      pat_lit(?la) {
        alt b.node {
          pat_lit(?lb) { ret util::common::lit_eq(la, lb); }
          _ { ret false; }
        }
      }
      pat_tag(?va, ?suba) {
        alt b.node {
          pat_tag(?vb, ?subb) {
            ret tcx.def_map.get(a.id) == tcx.def_map.get(b.id) &&
                patterns_supersede(tcx, suba, subb);
          }
          _ { ret false; }
        }
      }
      pat_rec(?suba, _) {
        alt b.node {
          pat_rec(?subb, _) { ret field_patterns_supersede(tcx, suba, subb); }
          _ { ret false; }
        }
      }
      pat_box(?suba) {
        alt b.node {
          pat_box(?subb) { ret pattern_supersedes(tcx, suba, subb); }
          _ { ret pattern_supersedes(tcx, suba, b); }
        }
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
