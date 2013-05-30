// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Type encoding

use core::prelude::*;

use middle::ty::param_ty;
use middle::ty;

use core::hashmap::HashMap;
use core::io::WriterUtil;
use core::io;
use core::uint;
use syntax::abi::AbiSet;
use syntax::ast;
use syntax::ast::*;
use syntax::diagnostic::span_handler;
use syntax::print::pprust::*;

pub struct ctxt {
    diag: @span_handler,
    // Def -> str Callback:
    ds: @fn(def_id) -> ~str,
    // The type context.
    tcx: ty::ctxt,
    reachable: @fn(node_id) -> bool,
    abbrevs: abbrev_ctxt
}

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.
pub struct ty_abbrev {
    pos: uint,
    len: uint,
    s: @~str
}

pub enum abbrev_ctxt {
    ac_no_abbrevs,
    ac_use_abbrevs(@mut HashMap<ty::t, ty_abbrev>),
}

fn cx_uses_abbrevs(cx: @ctxt) -> bool {
    match cx.abbrevs {
      ac_no_abbrevs => return false,
      ac_use_abbrevs(_) => return true
    }
}

pub fn enc_ty(w: @io::Writer, cx: @ctxt, t: ty::t) {
    match cx.abbrevs {
      ac_no_abbrevs => {
        let result_str = match cx.tcx.short_names_cache.find(&t) {
            Some(&s) => /*bad*/copy *s,
            None => {
                let s = do io::with_str_writer |wr| {
                    enc_sty(wr, cx, /*bad*/copy ty::get(t).sty);
                };
                cx.tcx.short_names_cache.insert(t, @copy s);
                s
          }
        };
        w.write_str(result_str);
      }
      ac_use_abbrevs(abbrevs) => {
          match abbrevs.find(&t) {
              Some(a) => { w.write_str(*a.s); return; }
              None => {}
          }
          let pos = w.tell();
          enc_sty(w, cx, /*bad*/copy ty::get(t).sty);
          let end = w.tell();
          let len = end - pos;
          fn estimate_sz(u: uint) -> uint {
              let mut n = u;
              let mut len = 0u;
              while n != 0u { len += 1u; n = n >> 4u; }
              return len;
          }
          let abbrev_len = 3u + estimate_sz(pos) + estimate_sz(len);
          if abbrev_len < len {
              // I.e. it's actually an abbreviation.
              let s = fmt!("#%x:%x#", pos, len);
              let a = ty_abbrev { pos: pos, len: len, s: @s };
              abbrevs.insert(t, a);
          }
          return;
      }
    }
}

fn enc_mutability(w: @io::Writer, mt: ast::mutability) {
    match mt {
      m_imm => (),
      m_mutbl => w.write_char('m'),
      m_const => w.write_char('?')
    }
}

fn enc_mt(w: @io::Writer, cx: @ctxt, mt: ty::mt) {
    enc_mutability(w, mt.mutbl);
    enc_ty(w, cx, mt.ty);
}

fn enc_opt<T>(w: @io::Writer, t: Option<T>, enc_f: &fn(T)) {
    match t {
      None => w.write_char('n'),
      Some(v) => {
        w.write_char('s');
        enc_f(v);
      }
    }
}

fn enc_substs(w: @io::Writer, cx: @ctxt, substs: &ty::substs) {
    do enc_opt(w, substs.self_r) |r| { enc_region(w, cx, r) }
    do enc_opt(w, substs.self_ty) |t| { enc_ty(w, cx, t) }
    w.write_char('[');
    for substs.tps.each |t| { enc_ty(w, cx, *t); }
    w.write_char(']');
}

fn enc_region(w: @io::Writer, cx: @ctxt, r: ty::Region) {
    match r {
      ty::re_bound(br) => {
        w.write_char('b');
        enc_bound_region(w, cx, br);
      }
      ty::re_free(ref fr) => {
        w.write_char('f');
        w.write_char('[');
        w.write_int(fr.scope_id);
        w.write_char('|');
        enc_bound_region(w, cx, fr.bound_region);
        w.write_char(']');
      }
      ty::re_scope(nid) => {
        w.write_char('s');
        w.write_int(nid);
        w.write_char('|');
      }
      ty::re_static => {
        w.write_char('t');
      }
      ty::re_empty => {
        w.write_char('e');
      }
      ty::re_infer(_) => {
        // these should not crop up after typeck
        cx.diag.handler().bug("Cannot encode region variables");
      }
    }
}

fn enc_bound_region(w: @io::Writer, cx: @ctxt, br: ty::bound_region) {
    match br {
      ty::br_self => w.write_char('s'),
      ty::br_anon(idx) => {
        w.write_char('a');
        w.write_uint(idx);
        w.write_char('|');
      }
      ty::br_named(s) => {
        w.write_char('[');
        w.write_str(*cx.tcx.sess.str_of(s));
        w.write_char(']')
      }
      ty::br_cap_avoid(id, br) => {
        w.write_char('c');
        w.write_int(id);
        w.write_char('|');
        enc_bound_region(w, cx, *br);
      }
      ty::br_fresh(id) => {
        w.write_uint(id);
      }
    }
}

pub fn enc_vstore(w: @io::Writer, cx: @ctxt, v: ty::vstore) {
    w.write_char('/');
    match v {
      ty::vstore_fixed(u) => {
        w.write_uint(u);
        w.write_char('|');
      }
      ty::vstore_uniq => {
        w.write_char('~');
      }
      ty::vstore_box => {
        w.write_char('@');
      }
      ty::vstore_slice(r) => {
        w.write_char('&');
        enc_region(w, cx, r);
      }
    }
}

pub fn enc_trait_ref(w: @io::Writer, cx: @ctxt, s: &ty::TraitRef) {
    w.write_str((cx.ds)(s.def_id));
    w.write_char('|');
    enc_substs(w, cx, &s.substs);
}

pub fn enc_trait_store(w: @io::Writer, cx: @ctxt, s: ty::TraitStore) {
    match s {
        ty::UniqTraitStore => w.write_char('~'),
        ty::BoxTraitStore => w.write_char('@'),
        ty::RegionTraitStore(re) => {
            w.write_char('&');
            enc_region(w, cx, re);
        }
    }
}

fn enc_sty(w: @io::Writer, cx: @ctxt, st: ty::sty) {
    match st {
      ty::ty_nil => w.write_char('n'),
      ty::ty_bot => w.write_char('z'),
      ty::ty_bool => w.write_char('b'),
      ty::ty_int(t) => {
        match t {
          ty_i => w.write_char('i'),
          ty_char => w.write_char('c'),
          ty_i8 => w.write_str(&"MB"),
          ty_i16 => w.write_str(&"MW"),
          ty_i32 => w.write_str(&"ML"),
          ty_i64 => w.write_str(&"MD")
        }
      }
      ty::ty_uint(t) => {
        match t {
          ty_u => w.write_char('u'),
          ty_u8 => w.write_str(&"Mb"),
          ty_u16 => w.write_str(&"Mw"),
          ty_u32 => w.write_str(&"Ml"),
          ty_u64 => w.write_str(&"Md")
        }
      }
      ty::ty_float(t) => {
        match t {
          ty_f => w.write_char('l'),
          ty_f32 => w.write_str(&"Mf"),
          ty_f64 => w.write_str(&"MF"),
        }
      }
      ty::ty_enum(def, ref substs) => {
        w.write_str(&"t[");
        w.write_str((cx.ds)(def));
        w.write_char('|');
        enc_substs(w, cx, substs);
        w.write_char(']');
      }
      ty::ty_trait(def, ref substs, store, mt) => {
        w.write_str(&"x[");
        w.write_str((cx.ds)(def));
        w.write_char('|');
        enc_substs(w, cx, substs);
        enc_trait_store(w, cx, store);
        enc_mutability(w, mt);
        w.write_char(']');
      }
      ty::ty_tup(ts) => {
        w.write_str(&"T[");
        for ts.each |t| { enc_ty(w, cx, *t); }
        w.write_char(']');
      }
      ty::ty_box(mt) => { w.write_char('@'); enc_mt(w, cx, mt); }
      ty::ty_uniq(mt) => { w.write_char('~'); enc_mt(w, cx, mt); }
      ty::ty_ptr(mt) => { w.write_char('*'); enc_mt(w, cx, mt); }
      ty::ty_rptr(r, mt) => {
        w.write_char('&');
        enc_region(w, cx, r);
        enc_mt(w, cx, mt);
      }
      ty::ty_evec(mt, v) => {
        w.write_char('V');
        enc_mt(w, cx, mt);
        enc_vstore(w, cx, v);
      }
      ty::ty_estr(v) => {
        w.write_char('v');
        enc_vstore(w, cx, v);
      }
      ty::ty_unboxed_vec(mt) => { w.write_char('U'); enc_mt(w, cx, mt); }
      ty::ty_closure(ref f) => {
        w.write_char('f');
        enc_closure_ty(w, cx, f);
      }
      ty::ty_bare_fn(ref f) => {
        w.write_char('F');
        enc_bare_fn_ty(w, cx, f);
      }
      ty::ty_infer(_) => {
        cx.diag.handler().bug("Cannot encode inference variable types");
      }
      ty::ty_param(param_ty {idx: id, def_id: did}) => {
        w.write_char('p');
        w.write_str((cx.ds)(did));
        w.write_char('|');
        w.write_str(uint::to_str(id));
      }
      ty::ty_self(did) => {
        w.write_char('s');
        w.write_str((cx.ds)(did));
        w.write_char('|');
      }
      ty::ty_type => w.write_char('Y'),
      ty::ty_opaque_closure_ptr(p) => {
          w.write_str(&"C&");
          enc_sigil(w, p);
      }
      ty::ty_opaque_box => w.write_char('B'),
      ty::ty_struct(def, ref substs) => {
          debug!("~~~~ %s", "a[");
          w.write_str(&"a[");
          let s = (cx.ds)(def);
          debug!("~~~~ %s", s);
          w.write_str(s);
          debug!("~~~~ %s", "|");
          w.write_char('|');
          enc_substs(w, cx, substs);
          debug!("~~~~ %s", "]");
          w.write_char(']');
      }
      ty::ty_err => fail!("Shouldn't encode error type")
    }
}

fn enc_sigil(w: @io::Writer, sigil: Sigil) {
    match sigil {
        ManagedSigil => w.write_str("@"),
        OwnedSigil => w.write_str("~"),
        BorrowedSigil => w.write_str("&"),
    }
}

fn enc_purity(w: @io::Writer, p: purity) {
    match p {
      pure_fn => w.write_char('p'),
      impure_fn => w.write_char('i'),
      unsafe_fn => w.write_char('u'),
      extern_fn => w.write_char('c')
    }
}

fn enc_abi_set(w: @io::Writer, abis: AbiSet) {
    w.write_char('[');
    for abis.each |abi| {
        w.write_str(abi.name());
        w.write_char(',');
    }
    w.write_char(']')
}

fn enc_onceness(w: @io::Writer, o: Onceness) {
    match o {
        Once => w.write_char('o'),
        Many => w.write_char('m')
    }
}

pub fn enc_bare_fn_ty(w: @io::Writer, cx: @ctxt, ft: &ty::BareFnTy) {
    enc_purity(w, ft.purity);
    enc_abi_set(w, ft.abis);
    enc_fn_sig(w, cx, &ft.sig);
}

fn enc_closure_ty(w: @io::Writer, cx: @ctxt, ft: &ty::ClosureTy) {
    enc_sigil(w, ft.sigil);
    enc_purity(w, ft.purity);
    enc_onceness(w, ft.onceness);
    enc_region(w, cx, ft.region);
    let bounds = ty::ParamBounds {builtin_bounds: ft.bounds,
                                  trait_bounds: ~[]};
    enc_bounds(w, cx, &bounds);
    enc_fn_sig(w, cx, &ft.sig);
}

fn enc_fn_sig(w: @io::Writer, cx: @ctxt, fsig: &ty::FnSig) {
    w.write_char('[');
    for fsig.inputs.each |ty| {
        enc_ty(w, cx, *ty);
    }
    w.write_char(']');
    enc_ty(w, cx, fsig.output);
}

fn enc_bounds(w: @io::Writer, cx: @ctxt, bs: &ty::ParamBounds) {
    for bs.builtin_bounds.each |bound| {
        match bound {
            ty::BoundOwned => w.write_char('S'),
            ty::BoundCopy => w.write_char('C'),
            ty::BoundConst => w.write_char('K'),
            ty::BoundStatic => w.write_char('O'),
        }
    }

    for bs.trait_bounds.each |&tp| {
        w.write_char('I');
        enc_trait_ref(w, cx, tp);
    }

    w.write_char('.');
}

pub fn enc_type_param_def(w: @io::Writer, cx: @ctxt, v: &ty::TypeParameterDef) {
    w.write_str((cx.ds)(v.def_id));
    w.write_char('|');
    enc_bounds(w, cx, v.bounds);
}
