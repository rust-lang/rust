import syntax::ast::*;
import syntax::ast_util::{variant_def_ids, dummy_sp, unguarded_pat};
import const_eval::{eval_const_expr, const_val, const_int,
                    compare_const_vals};
import syntax::codemap::span;
import syntax::print::pprust::pat_to_str;
import pat_util::*;
import syntax::visit;
import driver::session::session;
import middle::ty;
import middle::ty::*;
import std::map::hashmap;

fn check_crate(tcx: ty::ctxt, crate: @crate) {
    visit::visit_crate(*crate, (), visit::mk_vt(@{
        visit_expr: {|a,b,c|check_expr(tcx, a, b, c)},
        visit_local: {|a,b,c|check_local(tcx, a, b, c)}
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

// Check for unreachable patterns
fn check_arms(tcx: ty::ctxt, arms: [arm]/~) {
    let mut seen = []/~;
    for arms.each {|arm|
        for arm.pats.each {|pat|
            let v = [pat]/~;
            alt is_useful(tcx, seen, v) {
              not_useful {
                tcx.sess.span_err(pat.span, "unreachable pattern");
              }
              _ {}
            }
            if option::is_none(arm.guard) { seen += [v]/~; }
        }
    }
}

fn raw_pat(p: @pat) -> @pat {
    alt p.node {
      pat_ident(_, some(s)) { raw_pat(s) }
      _ { p }
    }
}

fn check_exhaustive(tcx: ty::ctxt, sp: span, pats: [@pat]/~) {
    let ext = alt is_useful(tcx, vec::map(pats, {|p| [p]/~}), [wild()]/~) {
      not_useful { ret; } // This is good, wildcard pattern isn't reachable
      useful_ { none }
      useful(ty, ctor) {
        alt ty::get(ty).struct {
          ty::ty_bool {
            alt check ctor {
              val(const_int(1i64)) { some(@"true") }
              val(const_int(0i64)) { some(@"false") }
            }
          }
          ty::ty_enum(id, _) {
            let vid = alt check ctor { variant(id) { id } };
            alt check vec::find(*ty::enum_variants(tcx, id),
                                {|v| v.id == vid}) {
              some(v) { some(v.name) }
            }
          }
          _ { none }
        }
      }
    };
    let msg = "non-exhaustive patterns" + alt ext {
      some(s) { ": " + *s + " not covered" }
      none { "" }
    };
    tcx.sess.span_err(sp, msg);
}

type matrix = [[@pat]/~]/~;

enum useful { useful(ty::t, ctor), useful_, not_useful }

enum ctor {
    single,
    variant(def_id),
    val(const_val),
    range(const_val, const_val),
}

// Algorithm from http://moscova.inria.fr/~maranget/papers/warn/index.html
//
// Whether a vector `v` of patterns is 'useful' in relation to a set of such
// vectors `m` is defined as there being a set of inputs that will match `v`
// but not any of the sets in `m`.
//
// This is used both for reachability checking (if a pattern isn't useful in
// relation to preceding patterns, it is not reachable) and exhaustiveness
// checking (if a wildcard pattern is useful in relation to a matrix, the
// matrix isn't exhaustive).

fn is_useful(tcx: ty::ctxt, m: matrix, v: [@pat]/~) -> useful {
    if m.len() == 0u { ret useful_; }
    if m[0].len() == 0u { ret not_useful; }
    let real_pat = alt vec::find(m, {|r| r[0].id != 0}) {
      some(r) { r[0] } none { v[0] }
    };
    let left_ty = if real_pat.id == 0 { ty::mk_nil(tcx) }
                  else { ty::node_id_to_type(tcx, real_pat.id) };

    alt pat_ctor_id(tcx, v[0]) {
      none {
        alt missing_ctor(tcx, m, left_ty) {
          none {
            alt ty::get(left_ty).struct {
              ty::ty_bool {
                alt is_useful_specialized(tcx, m, v, val(const_int(1i64)),
                                          0u, left_ty){
                  not_useful {
                    is_useful_specialized(tcx, m, v, val(const_int(0i64)),
                                          0u, left_ty)
                  }
                  u { u }
                }
              }
              ty::ty_enum(eid, _) {
                for (*ty::enum_variants(tcx, eid)).each {|va|
                    alt is_useful_specialized(tcx, m, v, variant(va.id),
                                              va.args.len(), left_ty) {
                      not_useful {}
                      u { ret u; }
                    }
                }
                not_useful
              }
              _ {
                let arity = ctor_arity(tcx, single, left_ty);
                is_useful_specialized(tcx, m, v, single, arity, left_ty)
              }
            }
          }
          some(ctor) {
            alt is_useful(tcx, vec::filter_map(m, {|r| default(tcx, r)}),
                          vec::tail(v)) {
              useful_ { useful(left_ty, ctor) }
              u { u }
            }
          }
        }
      }
      some(v0_ctor) {
        let arity = ctor_arity(tcx, v0_ctor, left_ty);
        is_useful_specialized(tcx, m, v, v0_ctor, arity, left_ty)
      }
    }
}

fn is_useful_specialized(tcx: ty::ctxt, m: matrix, v: [@pat]/~, ctor: ctor,
                          arity: uint, lty: ty::t) -> useful {
    let ms = vec::filter_map(m, {|r| specialize(tcx, r, ctor, arity, lty)});
    alt is_useful(tcx, ms, option::get(specialize(tcx, v, ctor, arity, lty))){
      useful_ { useful(lty, ctor) }
      u { u }
    }
}

fn pat_ctor_id(tcx: ty::ctxt, p: @pat) -> option<ctor> {
    let pat = raw_pat(p);
    alt pat.node {
      pat_wild { none }
      pat_ident(_, _) | pat_enum(_, _) {
        alt tcx.def_map.find(pat.id) {
          some(def_variant(_, id)) { some(variant(id)) }
          _ { none }
        }
      }
      pat_lit(expr) { some(val(eval_const_expr(tcx, expr))) }
      pat_range(lo, hi) {
        some(range(eval_const_expr(tcx, lo), eval_const_expr(tcx, hi)))
      }
      pat_box(_) | pat_uniq(_) | pat_rec(_, _) | pat_tup(_) { some(single) }
    }
}

fn is_wild(tcx: ty::ctxt, p: @pat) -> bool {
    let pat = raw_pat(p);
    alt pat.node {
      pat_wild { true }
      pat_ident(_, _) {
        alt tcx.def_map.find(pat.id) {
          some(def_variant(_, _)) { false }
          _ { true }
        }
      }
      _ { false }
    }
}

fn missing_ctor(tcx: ty::ctxt, m: matrix, left_ty: ty::t) -> option<ctor> {
    alt ty::get(left_ty).struct {
      ty::ty_box(_) | ty::ty_uniq(_) | ty::ty_tup(_) | ty::ty_rec(_) {
        for m.each {|r|
            if !is_wild(tcx, r[0]) { ret none; }
        }
        ret some(single);
      }
      ty::ty_enum(eid, _) {
        let mut found = []/~;
        for m.each {|r|
            option::iter(pat_ctor_id(tcx, r[0])) {|id|
                if !vec::contains(found, id) { found += [id]/~; }
            }
        }
        let variants = ty::enum_variants(tcx, eid);
        if found.len() != (*variants).len() {
            for vec::each(*variants) {|v|
                if !found.contains(variant(v.id)) {
                    ret some(variant(v.id));
                }
            }
            fail;
        } else { none }
      }
      ty::ty_nil { none }
      ty::ty_bool {
        let mut true_found = false, false_found = false;
        for m.each {|r|
            alt check pat_ctor_id(tcx, r[0]) {
              none {}
              some(val(const_int(1i64))) { true_found = true; }
              some(val(const_int(0i64))) { false_found = true; }
            }
        }
        if true_found && false_found { none }
        else if true_found { some(val(const_int(0i64))) }
        else { some(val(const_int(1i64))) }
      }
      _ { some(single) }
    }
}

fn ctor_arity(tcx: ty::ctxt, ctor: ctor, ty: ty::t) -> uint {
    alt ty::get(ty).struct {
      ty::ty_tup(fs) { fs.len() }
      ty::ty_rec(fs) { fs.len() }
      ty::ty_box(_) | ty::ty_uniq(_) { 1u }
      ty::ty_enum(eid, _) {
        let id = alt check ctor { variant(id) { id } };
        alt check vec::find(*ty::enum_variants(tcx, eid), {|v| v.id == id}) {
          some(v) { v.args.len() }
        }
      }
      _ { 0u }
    }
}

fn wild() -> @pat {
    @{id: 0, node: pat_wild, span: syntax::ast_util::dummy_sp()}
}

fn specialize(tcx: ty::ctxt, r: [@pat]/~, ctor_id: ctor, arity: uint,
              left_ty: ty::t) -> option<[@pat]/~> {
    let r0 = raw_pat(r[0]);
    alt r0.node {
      pat_wild { some(vec::from_elem(arity, wild()) + vec::tail(r)) }
      pat_ident(_, _) {
        alt tcx.def_map.find(r0.id) {
          some(def_variant(_, id)) {
            if variant(id) == ctor_id { some(vec::tail(r)) }
            else { none }
          }
          _ { some(vec::from_elem(arity, wild()) + vec::tail(r)) }
        }
      }
      pat_enum(_, args) {
        alt check tcx.def_map.get(r0.id) {
          def_variant(_, id) if variant(id) == ctor_id {
            let args = alt args {
              some(args) { args }
              none { vec::from_elem(arity, wild()) }
            };
            some(args + vec::tail(r))
          }
          def_variant(_, _) { none }
        }
      }
      pat_rec(flds, _) {
        let ty_flds = alt check ty::get(left_ty).struct {
          ty::ty_rec(flds) { flds }
        };
        let args = vec::map(ty_flds, {|ty_f|
            alt vec::find(flds, {|f| f.ident == ty_f.ident}) {
              some(f) { f.pat } _ { wild() }
            }
        });
        some(args + vec::tail(r))
      }
      pat_tup(args) { some(args + vec::tail(r)) }
      pat_box(a) | pat_uniq(a) { some([a]/~ + vec::tail(r)) }
      pat_lit(expr) {
        let e_v = eval_const_expr(tcx, expr);
        let match = alt check ctor_id {
          val(v) { compare_const_vals(e_v, v) == 0 }
          range(c_lo, c_hi) { compare_const_vals(c_lo, e_v) >= 0 &&
                              compare_const_vals(c_hi, e_v) <= 0 }
          single { true }
        };
        if match { some(vec::tail(r)) } else { none }
      }
      pat_range(lo, hi) {
        let (c_lo, c_hi) = alt check ctor_id {
          val(v) { (v, v) }
          range(lo, hi) { (lo, hi) }
          single { ret some(vec::tail(r)); }
        };
        let v_lo = eval_const_expr(tcx, lo),
            v_hi = eval_const_expr(tcx, hi);
        let match = compare_const_vals(c_lo, v_lo) >= 0 &&
                    compare_const_vals(c_hi, v_hi) <= 0;
        if match { some(vec::tail(r)) } else { none }
      }
    }
}

fn default(tcx: ty::ctxt, r: [@pat]/~) -> option<[@pat]/~> {
    if is_wild(tcx, r[0]) { some(vec::tail(r)) }
    else { none }
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
        for fields.each {|it|
            if is_refutable(tcx, it.pat) { ret true; }
        }
        false
      }
      pat_tup(elts) {
        for elts.each {|elt| if is_refutable(tcx, elt) { ret true; } }
        false
      }
      pat_enum(_, some(args)) {
        for args.each {|p| if is_refutable(tcx, p) { ret true; } };
        false
      }
      pat_enum(_,_) { false }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
