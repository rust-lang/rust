// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use metadata::encoder;
use middle::ty::{ReSkolemized, ReVar};
use middle::ty::{bound_region, br_anon, br_named, br_self, br_cap_avoid};
use middle::ty::{br_fresh, ctxt, field, method};
use middle::ty::{mt, t, param_ty};
use middle::ty::{re_bound, re_free, re_scope, re_infer, re_static, Region,
                 re_empty};
use middle::ty::{ty_bool, ty_bot, ty_box, ty_struct, ty_enum};
use middle::ty::{ty_err, ty_estr, ty_evec, ty_float, ty_bare_fn, ty_closure};
use middle::ty::{ty_nil, ty_opaque_box, ty_opaque_closure_ptr, ty_param};
use middle::ty::{ty_ptr, ty_rptr, ty_self, ty_tup, ty_type, ty_uniq};
use middle::ty::{ty_trait, ty_int};
use middle::ty::{ty_uint, ty_unboxed_vec, ty_infer};
use middle::ty;
use middle::typeck;
use syntax::abi::AbiSet;
use syntax::ast_map;
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::{ast, ast_util};

/// Produces a string suitable for debugging output.
pub trait Repr {
    fn repr(&self, tcx: ctxt) -> ~str;
}

/// Produces a string suitable for showing to the user.
pub trait UserString {
    fn user_string(&self, tcx: ctxt) -> ~str;
}

pub fn note_and_explain_region(cx: ctxt,
                               prefix: &str,
                               region: ty::Region,
                               suffix: &str) {
    match explain_region_and_span(cx, region) {
      (ref str, Some(span)) => {
        cx.sess.span_note(
            span,
            fmt!("%s%s%s", prefix, (*str), suffix));
      }
      (ref str, None) => {
        cx.sess.note(
            fmt!("%s%s%s", prefix, (*str), suffix));
      }
    }
}

/// Returns a string like "the block at 27:31" that attempts to explain a
/// lifetime in a way it might plausibly be understood.
pub fn explain_region(cx: ctxt, region: ty::Region) -> ~str {
  let (res, _) = explain_region_and_span(cx, region);
  return res;
}


pub fn explain_region_and_span(cx: ctxt, region: ty::Region)
                            -> (~str, Option<span>) {
    return match region {
      re_scope(node_id) => {
        match cx.items.find(&node_id) {
          Some(&ast_map::node_block(ref blk)) => {
            explain_span(cx, "block", blk.span)
          }
          Some(&ast_map::node_callee_scope(expr)) => {
              explain_span(cx, "callee", expr.span)
          }
          Some(&ast_map::node_expr(expr)) => {
            match expr.node {
              ast::expr_call(*) => explain_span(cx, "call", expr.span),
              ast::expr_method_call(*) => {
                explain_span(cx, "method call", expr.span)
              },
              ast::expr_match(*) => explain_span(cx, "match", expr.span),
              _ => explain_span(cx, "expression", expr.span)
            }
          }
          Some(&ast_map::node_stmt(stmt)) => {
              explain_span(cx, "statement", stmt.span)
          }
          Some(&ast_map::node_item(it, _)) if (match it.node {
                ast::item_fn(*) => true, _ => false}) => {
              explain_span(cx, "function body", it.span)
          }
          Some(_) | None => {
            // this really should not happen
            (fmt!("unknown scope: %d.  Please report a bug.", node_id),
             None)
          }
        }
      }

      re_free(ref fr) => {
        let prefix = match fr.bound_region {
          br_anon(idx) => fmt!("the anonymous lifetime #%u defined on",
                               idx + 1),
          br_fresh(_) => fmt!("an anonymous lifetime defined on"),
          _ => fmt!("the lifetime %s as defined on",
                    bound_region_to_str(cx, fr.bound_region))
        };

        match cx.items.find(&fr.scope_id) {
          Some(&ast_map::node_block(ref blk)) => {
            let (msg, opt_span) = explain_span(cx, "block", blk.span);
            (fmt!("%s %s", prefix, msg), opt_span)
          }
          Some(_) | None => {
            // this really should not happen
            (fmt!("%s node %d", prefix, fr.scope_id), None)
          }
        }
      }

      re_static => { (~"the static lifetime", None) }

      re_empty => { (~"the empty lifetime", None) }

      // I believe these cases should not occur (except when debugging,
      // perhaps)
      re_infer(_) | re_bound(_) => {
        (fmt!("lifetime %?", region), None)
      }
    };

    fn explain_span(cx: ctxt, heading: &str, span: span)
        -> (~str, Option<span>)
    {
        let lo = cx.sess.codemap.lookup_char_pos_adj(span.lo);
        (fmt!("the %s at %u:%u", heading,
              lo.line, lo.col.to_uint()), Some(span))
    }
}

pub fn bound_region_to_str(cx: ctxt, br: bound_region) -> ~str {
    bound_region_to_str_space(cx, "&", br)
}

pub fn bound_region_to_str_space(cx: ctxt,
                                 prefix: &str,
                                 br: bound_region)
                              -> ~str {
    if cx.sess.verbose() { return fmt!("%s%? ", prefix, br); }

    match br {
      br_named(id)         => fmt!("%s'%s ", prefix, *cx.sess.str_of(id)),
      br_self              => fmt!("%s'self ", prefix),
      br_anon(_)           => prefix.to_str(),
      br_fresh(_)          => prefix.to_str(),
      br_cap_avoid(_, br)  => bound_region_to_str_space(cx, prefix, *br)
    }
}

pub fn re_scope_id_to_str(cx: ctxt, node_id: ast::node_id) -> ~str {
    match cx.items.find(&node_id) {
      Some(&ast_map::node_block(ref blk)) => {
        fmt!("<block at %s>",
             cx.sess.codemap.span_to_str(blk.span))
      }
      Some(&ast_map::node_expr(expr)) => {
        match expr.node {
          ast::expr_call(*) => {
            fmt!("<call at %s>",
                 cx.sess.codemap.span_to_str(expr.span))
          }
          ast::expr_match(*) => {
            fmt!("<match at %s>",
                 cx.sess.codemap.span_to_str(expr.span))
          }
          ast::expr_assign_op(*) |
          ast::expr_field(*) |
          ast::expr_unary(*) |
          ast::expr_binary(*) |
          ast::expr_index(*) => {
            fmt!("<method at %s>",
                 cx.sess.codemap.span_to_str(expr.span))
          }
          _ => {
            fmt!("<expression at %s>",
                 cx.sess.codemap.span_to_str(expr.span))
          }
        }
      }
      None => {
        fmt!("<unknown-%d>", node_id)
      }
      _ => { cx.sess.bug(
          fmt!("re_scope refers to %s",
               ast_map::node_id_to_str(cx.items, node_id,
                                       cx.sess.parse_sess.interner))) }
    }
}

// In general, if you are giving a region error message,
// you should use `explain_region()` or, better yet,
// `note_and_explain_region()`
pub fn region_to_str(cx: ctxt, region: Region) -> ~str {
    region_to_str_space(cx, "&", region)
}

pub fn region_to_str_space(cx: ctxt, prefix: &str, region: Region) -> ~str {
    if cx.sess.verbose() {
        return fmt!("%s%? ", prefix, region);
    }

    // These printouts are concise.  They do not contain all the information
    // the user might want to diagnose an error, but there is basically no way
    // to fit that into a short string.  Hence the recommendation to use
    // `explain_region()` or `note_and_explain_region()`.
    match region {
        re_scope(_) => prefix.to_str(),
        re_bound(br) => bound_region_to_str_space(cx, prefix, br),
        re_free(ref fr) => bound_region_to_str_space(cx, prefix, fr.bound_region),
        re_infer(ReSkolemized(_, br)) => {
            bound_region_to_str_space(cx, prefix, br)
        }
        re_infer(ReVar(_)) => prefix.to_str(),
        re_static => fmt!("%s'static ", prefix),
        re_empty => fmt!("%s'<empty> ", prefix)
    }
}

fn mutability_to_str(m: ast::mutability) -> ~str {
    match m {
        ast::m_mutbl => ~"mut ",
        ast::m_imm => ~"",
        ast::m_const => ~"const "
    }
}

pub fn mt_to_str(cx: ctxt, m: &mt) -> ~str {
    mt_to_str_wrapped(cx, "", m, "")
}

pub fn mt_to_str_wrapped(cx: ctxt, before: &str, m: &mt, after: &str) -> ~str {
    let mstr = mutability_to_str(m.mutbl);
    return fmt!("%s%s%s%s", mstr, before, ty_to_str(cx, m.ty), after);
}

pub fn vstore_to_str(cx: ctxt, vs: ty::vstore) -> ~str {
    match vs {
      ty::vstore_fixed(n) => fmt!("%u", n),
      ty::vstore_uniq => ~"~",
      ty::vstore_box => ~"@",
      ty::vstore_slice(r) => region_to_str_space(cx, "&", r)
    }
}

pub fn trait_store_to_str(cx: ctxt, s: ty::TraitStore) -> ~str {
    match s {
      ty::UniqTraitStore => ~"~",
      ty::BoxTraitStore => ~"@",
      ty::RegionTraitStore(r) => region_to_str_space(cx, "&", r)
    }
}

pub fn vstore_ty_to_str(cx: ctxt, mt: &mt, vs: ty::vstore) -> ~str {
    match vs {
        ty::vstore_fixed(_) => {
            fmt!("[%s, .. %s]", mt_to_str(cx, mt), vstore_to_str(cx, vs))
        }
        _ => {
            fmt!("%s%s", vstore_to_str(cx, vs), mt_to_str_wrapped(cx, "[", mt, "]"))
        }
    }
}

pub fn tys_to_str(cx: ctxt, ts: &[t]) -> ~str {
    let tstrs = ts.map(|t| ty_to_str(cx, *t));
    fmt!("(%s)", str::connect(tstrs, ", "))
}

pub fn fn_sig_to_str(cx: ctxt, typ: &ty::FnSig) -> ~str {
    fmt!("fn%s -> %s",
         tys_to_str(cx, typ.inputs.map(|a| a.ty)),
         ty_to_str(cx, typ.output))
}

pub fn trait_ref_to_str(cx: ctxt, trait_ref: &ty::TraitRef) -> ~str {
    trait_ref.user_string(cx)
}

pub fn ty_to_str(cx: ctxt, typ: t) -> ~str {
    fn fn_input_to_str(cx: ctxt, input: ty::arg) -> ~str {
        ty_to_str(cx, input.ty)
    }
    fn bare_fn_to_str(cx: ctxt,
                      purity: ast::purity,
                      abis: AbiSet,
                      ident: Option<ast::ident>,
                      sig: &ty::FnSig)
                      -> ~str {
        let mut s = ~"extern ";

        s.push_str(abis.to_str());
        s.push_char(' ');

        match purity {
            ast::impure_fn => {}
            _ => {
                s.push_str(purity.to_str());
                s.push_char(' ');
            }
        };

        s.push_str("fn");

        match ident {
          Some(i) => {
              s.push_char(' ');
              s.push_str(*cx.sess.str_of(i));
          }
          _ => { }
        }

        push_sig_to_str(cx, &mut s, sig);

        return s;
    }
    fn closure_to_str(cx: ctxt, cty: &ty::ClosureTy) -> ~str
    {
        let mut s = cty.sigil.to_str();

        match (cty.sigil, cty.region) {
            (ast::ManagedSigil, ty::re_static) |
            (ast::OwnedSigil, ty::re_static) => {}

            (_, region) => {
                s.push_str(region_to_str_space(cx, "", region));
            }
        }

        match cty.purity {
            ast::impure_fn => {}
            _ => {
                s.push_str(cty.purity.to_str());
                s.push_char(' ');
            }
        };

        match cty.onceness {
            ast::Many => {}
            ast::Once => {
                s.push_str(cty.onceness.to_str());
                s.push_char(' ');
            }
        };

        s.push_str("fn");

        push_sig_to_str(cx, &mut s, &cty.sig);

        return s;
    }
    fn push_sig_to_str(cx: ctxt, s: &mut ~str, sig: &ty::FnSig) {
        s.push_char('(');
        let strs = sig.inputs.map(|a| fn_input_to_str(cx, *a));
        s.push_str(str::connect(strs, ", "));
        s.push_char(')');
        if ty::get(sig.output).sty != ty_nil {
            s.push_str(" -> ");
            if ty::type_is_bot(sig.output) {
                s.push_char('!');
            } else {
                s.push_str(ty_to_str(cx, sig.output));
            }
        }
    }
    fn method_to_str(cx: ctxt, m: method) -> ~str {
        bare_fn_to_str(cx,
                       m.fty.purity,
                       m.fty.abis,
                       Some(m.ident),
                       &m.fty.sig) + ~";"
    }
    fn field_to_str(cx: ctxt, f: field) -> ~str {
        return *cx.sess.str_of(f.ident) + ~": " + mt_to_str(cx, &f.mt);
    }

    // if there is an id, print that instead of the structural type:
    /*for ty::type_def_id(typ).each |def_id| {
        // note that this typedef cannot have type parameters
        return ast_map::path_to_str(ty::item_path(cx, *def_id),
                                    cx.sess.intr());
    }*/

    // pretty print the structural type representation:
    return match ty::get(typ).sty {
      ty_nil => ~"()",
      ty_bot => ~"!",
      ty_bool => ~"bool",
      ty_int(ast::ty_i) => ~"int",
      ty_int(ast::ty_char) => ~"char",
      ty_int(t) => ast_util::int_ty_to_str(t),
      ty_uint(ast::ty_u) => ~"uint",
      ty_uint(t) => ast_util::uint_ty_to_str(t),
      ty_float(ast::ty_f) => ~"float",
      ty_float(t) => ast_util::float_ty_to_str(t),
      ty_box(ref tm) => ~"@" + mt_to_str(cx, tm),
      ty_uniq(ref tm) => ~"~" + mt_to_str(cx, tm),
      ty_ptr(ref tm) => ~"*" + mt_to_str(cx, tm),
      ty_rptr(r, ref tm) => {
        region_to_str_space(cx, ~"&", r) + mt_to_str(cx, tm)
      }
      ty_unboxed_vec(ref tm) => { ~"unboxed_vec<" + mt_to_str(cx, tm) + ~">" }
      ty_type => ~"type",
      ty_tup(ref elems) => {
        let strs = elems.map(|elem| ty_to_str(cx, *elem));
        ~"(" + str::connect(strs, ~",") + ~")"
      }
      ty_closure(ref f) => {
          closure_to_str(cx, f)
      }
      ty_bare_fn(ref f) => {
          bare_fn_to_str(cx, f.purity, f.abis, None, &f.sig)
      }
      ty_infer(infer_ty) => infer_ty.to_str(),
      ty_err => ~"[type error]",
      ty_param(param_ty {idx: id, def_id: did}) => {
          if cx.sess.verbose() {
              fmt!("'%s:%?",
                   str::from_bytes(~[('a' as u8) + (id as u8)]),
                   did)
          } else {
              fmt!("'%s",
                   str::from_bytes(~[('a' as u8) + (id as u8)]))
          }
      }
      ty_self(*) => ~"Self",
      ty_enum(did, ref substs) | ty_struct(did, ref substs) => {
        let path = ty::item_path(cx, did);
        let base = ast_map::path_to_str(path, cx.sess.intr());
        parameterized(cx, base, substs.self_r, substs.tps)
      }
      ty_trait(did, ref substs, s, mutbl) => {
        let path = ty::item_path(cx, did);
        let base = ast_map::path_to_str(path, cx.sess.intr());
        let ty = parameterized(cx, base, substs.self_r, substs.tps);
        fmt!("%s%s%s", trait_store_to_str(cx, s), mutability_to_str(mutbl), ty)
      }
      ty_evec(ref mt, vs) => {
        vstore_ty_to_str(cx, mt, vs)
      }
      ty_estr(vs) => fmt!("%s%s", vstore_to_str(cx, vs), ~"str"),
      ty_opaque_box => ~"@?",
      ty_opaque_closure_ptr(ast::BorrowedSigil) => ~"closure&",
      ty_opaque_closure_ptr(ast::ManagedSigil) => ~"closure@",
      ty_opaque_closure_ptr(ast::OwnedSigil) => ~"closure~",
    }
}

pub fn parameterized(cx: ctxt,
                     base: &str,
                     self_r: Option<ty::Region>,
                     tps: &[ty::t]) -> ~str {

    let r_str = match self_r {
      None => ~"",
      Some(r) => {
        fmt!("/%s", region_to_str(cx, r))
      }
    };

    if vec::len(tps) > 0u {
        let strs = vec::map(tps, |t| ty_to_str(cx, *t));
        fmt!("%s%s<%s>", base, r_str, str::connect(strs, ","))
    } else {
        fmt!("%s%s", base, r_str)
    }
}

pub fn ty_to_short_str(cx: ctxt, typ: t) -> ~str {
    let mut s = encoder::encoded_ty(cx, typ);
    if str::len(s) >= 32u { s = str::slice(s, 0u, 32u).to_owned(); }
    return s;
}

impl<T:Repr> Repr for Option<T> {
    fn repr(&self, tcx: ctxt) -> ~str {
        match self {
            &None => ~"None",
            &Some(ref t) => fmt!("Some(%s)", t.repr(tcx))
        }
    }
}

/*
Annoyingly, these conflict with @ast::expr.

impl<T:Repr> Repr for @T {
    fn repr(&self, tcx: ctxt) -> ~str {
        (&**self).repr(tcx)
    }
}

impl<T:Repr> Repr for ~T {
    fn repr(&self, tcx: ctxt) -> ~str {
        (&**self).repr(tcx)
    }
}
*/

fn repr_vec<T:Repr>(tcx: ctxt, v: &[T]) -> ~str {
    fmt!("[%s]", str::connect(v.map(|t| t.repr(tcx)), ","))
}

impl<'self, T:Repr> Repr for &'self [T] {
    fn repr(&self, tcx: ctxt) -> ~str {
        repr_vec(tcx, *self)
    }
}

// This is necessary to handle types like Option<@~[T]>, for which
// autoderef cannot convert the &[T] handler
impl<T:Repr> Repr for @~[T] {
    fn repr(&self, tcx: ctxt) -> ~str {
        repr_vec(tcx, **self)
    }
}

impl Repr for ty::TypeParameterDef {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("TypeParameterDef {%?, bounds: %s}",
             self.def_id, self.bounds.repr(tcx))
    }
}

impl Repr for ty::t {
    fn repr(&self, tcx: ctxt) -> ~str {
        ty_to_str(tcx, *self)
    }
}

impl Repr for ty::substs {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("substs(self_r=%s, self_ty=%s, tps=%s)",
             self.self_r.repr(tcx),
             self.self_ty.repr(tcx),
             self.tps.repr(tcx))
    }
}

impl Repr for ty::ParamBounds {
    fn repr(&self, tcx: ctxt) -> ~str {
        let mut res = ~[];
        for self.builtin_bounds.each |b| {
            res.push(match b {
                ty::BoundCopy => ~"Copy",
                ty::BoundStatic => ~"'static",
                ty::BoundOwned => ~"Owned",
                ty::BoundConst => ~"Const",
            });
        }
        for self.trait_bounds.each |t| {
            res.push(t.repr(tcx));
        }
        str::connect(res, "+")
    }
}

impl Repr for ty::TraitRef {
    fn repr(&self, tcx: ctxt) -> ~str {
        trait_ref_to_str(tcx, self)
    }
}

impl Repr for @ast::expr {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("expr(%d: %s)",
             self.id,
             pprust::expr_to_str(*self, tcx.sess.intr()))
    }
}

impl Repr for @ast::pat {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("pat(%d: %s)",
             self.id,
             pprust::pat_to_str(*self, tcx.sess.intr()))
    }
}

impl Repr for ty::Region {
    fn repr(&self, tcx: ctxt) -> ~str {
        region_to_str(tcx, *self)
    }
}

impl Repr for ast::def_id {
    fn repr(&self, tcx: ctxt) -> ~str {
        // Unfortunately, there seems to be no way to attempt to print
        // a path for a def-id, so I'll just make a best effort for now
        // and otherwise fallback to just printing the crate/node pair
        if self.crate == ast::local_crate {
            match tcx.items.find(&self.node) {
                Some(&ast_map::node_item(*)) |
                Some(&ast_map::node_foreign_item(*)) |
                Some(&ast_map::node_method(*)) |
                Some(&ast_map::node_trait_method(*)) |
                Some(&ast_map::node_variant(*)) |
                Some(&ast_map::node_struct_ctor(*)) => {
                    return fmt!("%?:%s", *self, ty::item_path_str(tcx, *self));
                }
                _ => {}
            }
        }
        return fmt!("%?", *self);
    }
}

impl Repr for ty::ty_param_bounds_and_ty {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("ty_param_bounds_and_ty {generics: %s, ty: %s}",
             self.generics.repr(tcx),
             self.ty.repr(tcx))
    }
}

impl Repr for ty::Generics {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("Generics {type_param_defs: %s, region_param: %?}",
             self.type_param_defs.repr(tcx),
             self.region_param)
    }
}

impl Repr for ty::method {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("method {ident: %s, generics: %s, transformed_self_ty: %s, \
              fty: %s, self_ty: %s, vis: %s, def_id: %s}",
             self.ident.repr(tcx),
             self.generics.repr(tcx),
             self.transformed_self_ty.repr(tcx),
             self.fty.repr(tcx),
             self.self_ty.repr(tcx),
             self.vis.repr(tcx),
             self.def_id.repr(tcx))
    }
}

impl Repr for ast::ident {
    fn repr(&self, tcx: ctxt) -> ~str {
        copy *tcx.sess.intr().get(*self)
    }
}

impl Repr for ast::self_ty_ {
    fn repr(&self, _tcx: ctxt) -> ~str {
        fmt!("%?", *self)
    }
}

impl Repr for ast::visibility {
    fn repr(&self, _tcx: ctxt) -> ~str {
        fmt!("%?", *self)
    }
}

impl Repr for ty::BareFnTy {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("BareFnTy {purity: %?, abis: %s, sig: %s}",
             self.purity,
             self.abis.to_str(),
             self.sig.repr(tcx))
    }
}

impl Repr for ty::FnSig {
    fn repr(&self, tcx: ctxt) -> ~str {
        fn_sig_to_str(tcx, self)
    }
}

impl Repr for typeck::method_map_entry {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("method_map_entry {self_arg: %s, \
              explicit_self: %s, \
              origin: %s}",
             self.self_arg.repr(tcx),
             self.explicit_self.repr(tcx),
             self.origin.repr(tcx))
    }
}

impl Repr for ty::arg {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("(%s)", self.ty.repr(tcx))
    }
}

impl Repr for typeck::method_origin {
    fn repr(&self, tcx: ctxt) -> ~str {
        match self {
            &typeck::method_super(def_id, n) => {
                fmt!("method_super(%s, %?)",
                     def_id.repr(tcx), n)
            }
            &typeck::method_static(def_id) => {
                fmt!("method_static(%s)", def_id.repr(tcx))
            }
            &typeck::method_param(ref p) => {
                p.repr(tcx)
            }
            &typeck::method_trait(def_id, n, st) => {
                fmt!("method_trait(%s, %?, %s)", def_id.repr(tcx), n,
                     st.repr(tcx))
            }
            &typeck::method_self(def_id, n) => {
                fmt!("method_self(%s, %?)", def_id.repr(tcx), n)
            }
        }
    }
}

impl Repr for typeck::method_param {
    fn repr(&self, tcx: ctxt) -> ~str {
        fmt!("method_param(%s,%?,%?,%?)",
             self.trait_id.repr(tcx),
             self.method_num,
             self.param_num,
             self.bound_num)
    }
}

impl Repr for ty::TraitStore {
    fn repr(&self, tcx: ctxt) -> ~str {
        match self {
            &ty::BoxTraitStore => ~"@Trait",
            &ty::UniqTraitStore => ~"~Trait",
            &ty::RegionTraitStore(r) => fmt!("&%s Trait", r.repr(tcx))
        }
    }
}

impl Repr for ty::vstore {
    fn repr(&self, tcx: ctxt) -> ~str {
        vstore_to_str(tcx, *self)
    }
}

impl Repr for ast_map::path_elt {
    fn repr(&self, tcx: ctxt) -> ~str {
        match *self {
            ast_map::path_mod(id) => id.repr(tcx),
            ast_map::path_name(id) => id.repr(tcx)
        }
    }
}

impl Repr for ty::BuiltinBound {
    fn repr(&self, _tcx: ctxt) -> ~str {
        fmt!("%?", *self)
    }
}

impl UserString for ty::BuiltinBound {
    fn user_string(&self, _tcx: ctxt) -> ~str {
        match *self {
            ty::BoundCopy => ~"Copy",
            ty::BoundStatic => ~"'static",
            ty::BoundOwned => ~"Owned",
            ty::BoundConst => ~"Const"
        }
    }
}

impl Repr for ty::BuiltinBounds {
    fn repr(&self, tcx: ctxt) -> ~str {
        self.user_string(tcx)
    }
}

impl UserString for ty::BuiltinBounds {
    fn user_string(&self, tcx: ctxt) -> ~str {
        if self.is_empty() { ~"<no-bounds>" } else {
            let mut result = ~[];
            for self.each |bb| {
                result.push(bb.user_string(tcx));
            }
            str::connect(result, "+")
        }
    }
}

impl UserString for ty::TraitRef {
    fn user_string(&self, tcx: ctxt) -> ~str {
        let path = ty::item_path(tcx, self.def_id);
        let base = ast_map::path_to_str(path, tcx.sess.intr());
        if tcx.sess.verbose() && self.substs.self_ty.is_some() {
            let mut all_tps = copy self.substs.tps;
            for self.substs.self_ty.each |&t| { all_tps.push(t); }
            parameterized(tcx, base, self.substs.self_r, all_tps)
        } else {
            parameterized(tcx, base, self.substs.self_r,
                          self.substs.tps)
        }
    }
}
