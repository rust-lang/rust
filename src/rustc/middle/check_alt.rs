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
        visit_expr: |a,b,c| check_expr(tcx, a, b, c),
        visit_local: |a,b,c| check_local(tcx, a, b, c)
        with *visit::default_visitor::<()>()
    }));
    tcx.sess.abort_if_errors();
}

fn check_expr(tcx: ty::ctxt, ex: @expr, &&s: (), v: visit::vt<()>) {
    visit::visit_expr(ex, s, v);
    match ex.node {
      expr_alt(scrut, arms, mode) => {
        check_arms(tcx, arms);
        /* Check for exhaustiveness */
         // Check for empty enum, because is_useful only works on inhabited
         // types.
       let pat_ty = node_id_to_type(tcx, scrut.id);
       if type_is_empty(tcx, pat_ty) && arms.is_empty() {
               // Vacuously exhaustive
               return;
           }
       match ty::get(pat_ty).struct {
          ty_enum(did, _) => {
              if (*enum_variants(tcx, did)).is_empty() && arms.is_empty() {

               return;
            }
          }
          _ => { /* We assume only enum types can be uninhabited */ }
       }

        if mode == alt_exhaustive {
            let arms = vec::concat(vec::filter_map(arms, unguarded_pat));
            check_exhaustive(tcx, ex.span, arms);
        }
      }
      _ => ()
    }
}

// Check for unreachable patterns
fn check_arms(tcx: ty::ctxt, arms: ~[arm]) {
    let mut seen = ~[];
    for arms.each |arm| {
        for arm.pats.each |pat| {
            let v = ~[pat];
            match is_useful(tcx, seen, v) {
              not_useful => {
                tcx.sess.span_err(pat.span, ~"unreachable pattern");
              }
              _ => ()
            }
            if option::is_none(arm.guard) { vec::push(seen, v); }
        }
    }
}

fn raw_pat(p: @pat) -> @pat {
    match p.node {
      pat_ident(_, _, some(s)) => { raw_pat(s) }
      _ => { p }
    }
}

fn check_exhaustive(tcx: ty::ctxt, sp: span, pats: ~[@pat]) {
    assert(pats.is_not_empty());
    let ext = match is_useful(tcx, vec::map(pats, |p| ~[p]), ~[wild()]) {
      not_useful => return, // This is good, wildcard pattern isn't reachable
      useful_ => none,
      useful(ty, ctor) => {
        match ty::get(ty).struct {
          ty::ty_bool => {
            match check ctor {
              val(const_int(1i64)) => some(@~"true"),
              val(const_int(0i64)) => some(@~"false")
            }
          }
          ty::ty_enum(id, _) => {
            let vid = match check ctor { variant(id) => id };
            match check vec::find(*ty::enum_variants(tcx, id),
                                |v| v.id == vid) {
              some(v) => some(v.name)
            }
          }
          _ => none
        }
      }
    };
    let msg = ~"non-exhaustive patterns" + match ext {
      some(s) => ~": " + *s + ~" not covered",
      none => ~""
    };
    tcx.sess.span_err(sp, msg);
}

type matrix = ~[~[@pat]];

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

// Note: is_useful doesn't work on empty types, as the paper notes.
// So it assumes that v is non-empty.
fn is_useful(tcx: ty::ctxt, m: matrix, v: ~[@pat]) -> useful {
    if m.len() == 0u { return useful_; }
    if m[0].len() == 0u { return not_useful; }
    let real_pat = match vec::find(m, |r| r[0].id != 0) {
      some(r) => r[0], none => v[0]
    };
    let left_ty = if real_pat.id == 0 { ty::mk_nil(tcx) }
                  else { ty::node_id_to_type(tcx, real_pat.id) };

    match pat_ctor_id(tcx, v[0]) {
      none => {
        match missing_ctor(tcx, m, left_ty) {
          none => {
            match ty::get(left_ty).struct {
              ty::ty_bool => {
                match is_useful_specialized(tcx, m, v, val(const_int(1i64)),
                                          0u, left_ty){
                  not_useful => {
                    is_useful_specialized(tcx, m, v, val(const_int(0i64)),
                                          0u, left_ty)
                  }
                  u => u
                }
              }
              ty::ty_enum(eid, _) => {
                for (*ty::enum_variants(tcx, eid)).each |va| {
                    match is_useful_specialized(tcx, m, v, variant(va.id),
                                              va.args.len(), left_ty) {
                      not_useful => (),
                      u => return u
                    }
                }
                not_useful
              }
              _ => {
                let arity = ctor_arity(tcx, single, left_ty);
                is_useful_specialized(tcx, m, v, single, arity, left_ty)
              }
            }
          }
          some(ctor) => {
            match is_useful(tcx, vec::filter_map(m, |r| default(tcx, r) ),
                          vec::tail(v)) {
              useful_ => useful(left_ty, ctor),
              u => u
            }
          }
        }
      }
      some(v0_ctor) => {
        let arity = ctor_arity(tcx, v0_ctor, left_ty);
        is_useful_specialized(tcx, m, v, v0_ctor, arity, left_ty)
      }
    }
}

fn is_useful_specialized(tcx: ty::ctxt, m: matrix, v: ~[@pat], ctor: ctor,
                          arity: uint, lty: ty::t) -> useful {
    let ms = vec::filter_map(m, |r| specialize(tcx, r, ctor, arity, lty) );
    let could_be_useful = is_useful(
        tcx, ms, option::get(specialize(tcx, v, ctor, arity, lty)));
    match could_be_useful {
      useful_ => useful(lty, ctor),
      u => u
    }
}

fn pat_ctor_id(tcx: ty::ctxt, p: @pat) -> option<ctor> {
    let pat = raw_pat(p);
    match pat.node {
      pat_wild => { none }
      pat_ident(_, _, _) | pat_enum(_, _) => {
        match tcx.def_map.find(pat.id) {
          some(def_variant(_, id)) => some(variant(id)),
          _ => none
        }
      }
      pat_lit(expr) => { some(val(eval_const_expr(tcx, expr))) }
      pat_range(lo, hi) => {
        some(range(eval_const_expr(tcx, lo), eval_const_expr(tcx, hi)))
      }
      pat_box(_) | pat_uniq(_) | pat_rec(_, _) | pat_tup(_) |
      pat_struct(*) => {
        some(single)
      }
    }
}

fn is_wild(tcx: ty::ctxt, p: @pat) -> bool {
    let pat = raw_pat(p);
    match pat.node {
      pat_wild => { true }
      pat_ident(_, _, _) => {
        match tcx.def_map.find(pat.id) {
          some(def_variant(_, _)) => { false }
          _ => { true }
        }
      }
      _ => { false }
    }
}

fn missing_ctor(tcx: ty::ctxt, m: matrix, left_ty: ty::t) -> option<ctor> {
    match ty::get(left_ty).struct {
      ty::ty_box(_) | ty::ty_uniq(_) | ty::ty_tup(_) | ty::ty_rec(_) |
      ty::ty_class(*) => {
        for m.each |r| {
            if !is_wild(tcx, r[0]) { return none; }
        }
        return some(single);
      }
      ty::ty_enum(eid, _) => {
        let mut found = ~[];
        for m.each |r| {
            do option::iter(pat_ctor_id(tcx, r[0])) |id| {
                if !vec::contains(found, id) { vec::push(found, id); }
            }
        }
        let variants = ty::enum_variants(tcx, eid);
        if found.len() != (*variants).len() {
            for vec::each(*variants) |v| {
                if !found.contains(variant(v.id)) {
                    return some(variant(v.id));
                }
            }
            fail;
        } else { none }
      }
      ty::ty_nil => none,
      ty::ty_bool => {
        let mut true_found = false, false_found = false;
        for m.each |r| {
            match check pat_ctor_id(tcx, r[0]) {
              none => (),
              some(val(const_int(1i64))) => true_found = true,
              some(val(const_int(0i64))) => false_found = true
            }
        }
        if true_found && false_found { none }
        else if true_found { some(val(const_int(0i64))) }
        else { some(val(const_int(1i64))) }
      }
      _ => some(single)
    }
}

fn ctor_arity(tcx: ty::ctxt, ctor: ctor, ty: ty::t) -> uint {
    match ty::get(ty).struct {
      ty::ty_tup(fs) => fs.len(),
      ty::ty_rec(fs) => fs.len(),
      ty::ty_box(_) | ty::ty_uniq(_) => 1u,
      ty::ty_enum(eid, _) => {
        let id = match check ctor { variant(id) => id };
        match check vec::find(*ty::enum_variants(tcx, eid), |v| v.id == id ) {
          some(v) => v.args.len()
        }
      }
      ty::ty_class(cid, _) => ty::lookup_class_fields(tcx, cid).len(),
      _ => 0u
    }
}

fn wild() -> @pat {
    @{id: 0, node: pat_wild, span: syntax::ast_util::dummy_sp()}
}

fn specialize(tcx: ty::ctxt, r: ~[@pat], ctor_id: ctor, arity: uint,
              left_ty: ty::t) -> option<~[@pat]> {
    let r0 = raw_pat(r[0]);
    match r0.node {
      pat_wild => some(vec::append(vec::from_elem(arity, wild()),
                                   vec::tail(r))),
      pat_ident(_, _, _) => {
        match tcx.def_map.find(r0.id) {
          some(def_variant(_, id)) => {
            if variant(id) == ctor_id { some(vec::tail(r)) }
            else { none }
          }
          _ => some(vec::append(vec::from_elem(arity, wild()), vec::tail(r)))
        }
      }
      pat_enum(_, args) => {
        match check tcx.def_map.get(r0.id) {
          def_variant(_, id) if variant(id) == ctor_id => {
            let args = match args {
              some(args) => args,
              none => vec::from_elem(arity, wild())
            };
            some(vec::append(args, vec::tail(r)))
          }
          def_variant(_, _) => none
        }
      }
      pat_rec(flds, _) => {
        let ty_flds = match check ty::get(left_ty).struct {
          ty::ty_rec(flds) => flds
        };
        let args = vec::map(ty_flds, |ty_f| {
            match vec::find(flds, |f| f.ident == ty_f.ident ) {
              some(f) => f.pat,
              _ => wild()
            }
        });
        some(vec::append(args, vec::tail(r)))
      }
      pat_struct(_, flds, _) => {
        // Grab the class data that we care about.
        let class_fields, class_id;
        match ty::get(left_ty).struct {
            ty::ty_class(cid, substs) => {
                class_id = cid;
                class_fields = ty::lookup_class_fields(tcx, class_id);
            }
            _ => {
                tcx.sess.span_bug(r0.span, ~"struct pattern didn't resolve \
                                             to a struct");
            }
        }
        let args = vec::map(class_fields, |class_field| {
            match vec::find(flds, |f| f.ident == class_field.ident ) {
              some(f) => f.pat,
              _ => wild()
            }
        });
        some(vec::append(args, vec::tail(r)))
      }
      pat_tup(args) => some(vec::append(args, vec::tail(r))),
      pat_box(a) | pat_uniq(a) => some(vec::append(~[a], vec::tail(r))),
      pat_lit(expr) => {
        let e_v = eval_const_expr(tcx, expr);
        let match_ = match check ctor_id {
          val(v) => compare_const_vals(e_v, v) == 0,
          range(c_lo, c_hi) => {
            compare_const_vals(c_lo, e_v) >= 0 &&
                compare_const_vals(c_hi, e_v) <= 0
          }
          single => true
        };
        if match_ { some(vec::tail(r)) } else { none }
      }
      pat_range(lo, hi) => {
        let (c_lo, c_hi) = match check ctor_id {
          val(v) => (v, v),
          range(lo, hi) => (lo, hi),
          single => return some(vec::tail(r)),
        };
        let v_lo = eval_const_expr(tcx, lo),
            v_hi = eval_const_expr(tcx, hi);
        let match_ = compare_const_vals(c_lo, v_lo) >= 0 &&
                    compare_const_vals(c_hi, v_hi) <= 0;
        if match_ { some(vec::tail(r)) } else { none }
      }
    }
}

fn default(tcx: ty::ctxt, r: ~[@pat]) -> option<~[@pat]> {
    if is_wild(tcx, r[0]) { some(vec::tail(r)) }
    else { none }
}

fn check_local(tcx: ty::ctxt, loc: @local, &&s: (), v: visit::vt<()>) {
    visit::visit_local(loc, s, v);
    if is_refutable(tcx, loc.node.pat) {
        tcx.sess.span_err(loc.node.pat.span,
                          ~"refutable pattern in local binding");
    }
}

fn is_refutable(tcx: ty::ctxt, pat: @pat) -> bool {
    match tcx.def_map.find(pat.id) {
      some(def_variant(enum_id, var_id)) => {
        if vec::len(*ty::enum_variants(tcx, enum_id)) != 1u {
            return true;
        }
      }
      _ => ()
    }

    match pat.node {
      pat_box(sub) | pat_uniq(sub) | pat_ident(_, _, some(sub)) => {
        is_refutable(tcx, sub)
      }
      pat_wild | pat_ident(_, _, none) => { false }
      pat_lit(_) | pat_range(_, _) => { true }
      pat_rec(fields, _) => {
        for fields.each |it| {
            if is_refutable(tcx, it.pat) { return true; }
        }
        false
      }
      pat_struct(_, fields, _) => {
        for fields.each |it| {
            if is_refutable(tcx, it.pat) { return true; }
        }
        false
      }
      pat_tup(elts) => {
        for elts.each |elt| { if is_refutable(tcx, elt) { return true; } }
        false
      }
      pat_enum(_, some(args)) => {
        for args.each |p| { if is_refutable(tcx, p) { return true; } };
        false
      }
      pat_enum(_,_) => { false }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
