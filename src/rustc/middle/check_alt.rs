
import syntax::ast::*;
import syntax::ast_util::{variant_def_ids, dummy_sp, compare_lit_exprs,
        lit_expr_eq, unguarded_pat};
import syntax::codemap::span;
import pat_util::*;
import syntax::visit;
import driver::session::session;
import middle::ty;
import middle::ty::*;

fn check_crate(tcx: ty::ctxt, crate: @crate) {
    visit::visit_crate(*crate, (), visit::mk_vt(@{
        visit_expr: bind check_expr(tcx, _, _, _),
        visit_local: bind check_local(tcx, _, _, _)
        with *visit::default_visitor::<()>()
    }));
    tcx.sess.abort_if_errors();
}

fn check_expr(tcx: ty::ctxt, ex: @expr, &&s: (), v: visit::vt<()>) {
    visit::visit_expr(ex, s, v);
    alt ex.node {
      expr_alt(scrut, arms, mode) {
        check_arms(tcx, arms);
        /* Check for exhaustiveness */
        if mode == alt_exhaustive {
            let arms = vec::concat(vec::filter_map(arms, unguarded_pat));
            check_exhaustive(tcx, ex.span, arms);
        }
      }
      _ { }
    }
}

fn check_arms(tcx: ty::ctxt, arms: [arm]) {
    let i = 0;
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
}

fn raw_pat(p: @pat) -> @pat {
    alt p.node {
      pat_ident(_, some(s)) { raw_pat(s) }
      _ { p }
    }
}

fn check_exhaustive(tcx: ty::ctxt, sp: span, pats: [@pat]) {
    if pats.len() == 0u {
        tcx.sess.span_err(sp, "non-exhaustive patterns");
        ret;
    }
    // If there a non-refutable pattern in the set, we're okay.
    for pat in pats { if !is_refutable(tcx, pat) { ret; } }

    alt ty::get(ty::node_id_to_type(tcx, pats[0].id)).struct {
      ty::ty_enum(id, _) {
        check_exhaustive_enum(tcx, id, sp, pats);
      }
      ty::ty_box(_) {
        check_exhaustive(tcx, sp, vec::filter_map(pats, {|p|
            alt raw_pat(p).node { pat_box(sub) { some(sub) } _ { none } }
        }));
      }
      ty::ty_uniq(_) {
        check_exhaustive(tcx, sp, vec::filter_map(pats, {|p|
            alt raw_pat(p).node { pat_uniq(sub) { some(sub) } _ { none } }
        }));
      }
      ty::ty_tup(ts) {
        let cols = vec::to_mut(vec::init_elt(ts.len(), []));
        for p in pats {
            alt raw_pat(p).node {
              pat_tup(sub) {
                vec::iteri(sub) {|i, sp| cols[i] += [sp];}
              }
              _ {}
            }
        }
        vec::iter(cols) {|col| check_exhaustive(tcx, sp, col); }
      }
      ty::ty_rec(fs) {
        let cols = vec::init_elt(fs.len(), {mutable wild: false,
                                            mutable pats: []});
        for p in pats {
            alt raw_pat(p).node {
              pat_rec(sub, _) {
                vec::iteri(fs) {|i, field|
                    alt vec::find(sub, {|pf| pf.ident == field.ident }) {
                      some(pf) { cols[i].pats += [pf.pat]; }
                      none { cols[i].wild = true; }
                    }
                }
              }
              _ {}
            }
        }
        vec::iter(cols) {|col|
            if !col.wild { check_exhaustive(tcx, sp, copy col.pats); }
        }
      }
      ty::ty_bool {
        let saw_true = false, saw_false = false;
        for p in pats {
            alt raw_pat(p).node {
              pat_lit(@{node: expr_lit(@{node: lit_bool(b), _}), _}) {
                if b { saw_true = true; }
                else { saw_false = true; }
              }
              _ {}
            }
        }
        if !saw_true { tcx.sess.span_err(
            sp, "non-exhaustive bool patterns: true not covered"); }
        if !saw_false { tcx.sess.span_err(
            sp, "non-exhaustive bool patterns: false not covered"); }
      }
      ty::ty_nil {
        let seen = vec::any(pats, {|p|
            alt raw_pat(p).node {
              pat_lit(@{node: expr_lit(@{node: lit_nil, _}), _}) { true }
              _ { false }
            }
        });
        if !seen { tcx.sess.span_err(sp, "non-exhaustive patterns"); }
      }
      // Literal patterns are always considered non-exhaustive
      _ {
        tcx.sess.span_err(sp, "non-exhaustive literal patterns");
      }
    }
}

fn check_exhaustive_enum(tcx: ty::ctxt, enum_id: def_id, sp: span,
                         pats: [@pat]) {
    let variants = enum_variants(tcx, enum_id);
    let columns_by_variant = vec::map(*variants, {|v|
        {mutable seen: false,
         cols: vec::to_mut(vec::init_elt(v.args.len(), []))}
    });

    for pat in pats {
        let pat = raw_pat(pat);
        alt tcx.def_map.get(pat.id) {
          def_variant(_, id) {
            let variant_idx =
                option::get(vec::position(*variants, {|v| v.id == id}));
            columns_by_variant[variant_idx].seen = true;
            alt pat.node {
              pat_enum(_, args) {
                vec::iteri(args) {|i, p|
                    columns_by_variant[variant_idx].cols[i] += [p];
                }
              }
              _ {}
            }
          }
          _ {}
        }
    }

    vec::iteri(columns_by_variant) {|i, cv|
        if !cv.seen {
            tcx.sess.span_err(sp, "non-exhaustive patterns: variant `" +
                              variants[i].name + "` not covered");
        } else {
            vec::iter(cv.cols) {|col| check_exhaustive(tcx, sp, col); }
        }
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
      pat_wild { true }
      pat_ident(_, none) {
        let opt_def_a = tcx.def_map.find(a.id);
        alt opt_def_a {
          some(def_variant(_, _)) { opt_def_a == tcx.def_map.find(b.id) }
          // This is a binding
          _ { true }
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
      pat_lit(la) {
        alt b.node {
          pat_lit(lb) { lit_expr_eq(la, lb) }
          _ { false }
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
    alt tcx.def_map.find(pat.id) {
      some(def_variant(enum_id, var_id)) {
        if vec::len(*ty::enum_variants(tcx, enum_id)) != 1u { ret true; }
      }
      _ {}
    }

    alt pat.node {
      pat_box(sub) | pat_uniq(sub) | pat_ident(_, some(sub)) {
        is_refutable(tcx, sub)
      }
      pat_wild | pat_ident(_, none) { false }
      pat_lit(_) | pat_range(_, _) { true }
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
