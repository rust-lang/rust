// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Type encoding

#![allow(unused_must_use)] // as with encoding, everything is a no-fail MemWriter
#![allow(non_camel_case_types)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::MemWriter;

use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty::ParamTy;
use middle::ty;

use syntax::abi::Abi;
use syntax::ast;
use syntax::ast::*;
use syntax::diagnostic::SpanHandler;
use syntax::parse::token;

macro_rules! mywrite( ($($arg:tt)*) => ({ write!($($arg)*); }) )

pub struct ctxt<'a> {
    pub diag: &'a SpanHandler,
    // Def -> str Callback:
    pub ds: fn(DefId) -> String,
    // The type context.
    pub tcx: &'a ty::ctxt,
    pub abbrevs: &'a abbrev_map
}

// Compact string representation for ty.t values. API ty_str & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.
pub struct ty_abbrev {
    s: String
}

pub type abbrev_map = RefCell<HashMap<ty::t, ty_abbrev>>;

pub fn enc_ty(w: &mut MemWriter, cx: &ctxt, t: ty::t) {
    match cx.abbrevs.borrow_mut().find(&t) {
        Some(a) => { w.write(a.s.as_bytes()); return; }
        None => {}
    }
    let pos = w.tell().unwrap();
    enc_sty(w, cx, &ty::get(t).sty);
    let end = w.tell().unwrap();
    let len = end - pos;
    fn estimate_sz(u: u64) -> u64 {
        let mut n = u;
        let mut len = 0;
        while n != 0 { len += 1; n = n >> 4; }
        return len;
    }
    let abbrev_len = 3 + estimate_sz(pos) + estimate_sz(len);
    if abbrev_len < len {
        // I.e. it's actually an abbreviation.
        cx.abbrevs.borrow_mut().insert(t, ty_abbrev {
            s: format!("#{:x}:{:x}#", pos, len)
        });
    }
}

fn enc_mutability(w: &mut MemWriter, mt: ast::Mutability) {
    match mt {
        MutImmutable => (),
        MutMutable => mywrite!(w, "m"),
    }
}

fn enc_mt(w: &mut MemWriter, cx: &ctxt, mt: ty::mt) {
    enc_mutability(w, mt.mutbl);
    enc_ty(w, cx, mt.ty);
}

fn enc_opt<T>(w: &mut MemWriter, t: Option<T>, enc_f: |&mut MemWriter, T|) {
    match t {
        None => mywrite!(w, "n"),
        Some(v) => {
            mywrite!(w, "s");
            enc_f(w, v);
        }
    }
}

fn enc_vec_per_param_space<T>(w: &mut MemWriter,
                              cx: &ctxt,
                              v: &VecPerParamSpace<T>,
                              op: |&mut MemWriter, &ctxt, &T|) {
    for &space in subst::ParamSpace::all().iter() {
        mywrite!(w, "[");
        for t in v.get_vec(space).iter() {
            op(w, cx, t);
        }
        mywrite!(w, "]");
    }
}

pub fn enc_substs(w: &mut MemWriter, cx: &ctxt, substs: &subst::Substs) {
    enc_region_substs(w, cx, &substs.regions);
    enc_vec_per_param_space(w, cx, &substs.types,
                            |w, cx, &ty| enc_ty(w, cx, ty));
}

fn enc_region_substs(w: &mut MemWriter, cx: &ctxt, substs: &subst::RegionSubsts) {
    match *substs {
        subst::ErasedRegions => {
            mywrite!(w, "e");
        }
        subst::NonerasedRegions(ref regions) => {
            mywrite!(w, "n");
            enc_vec_per_param_space(w, cx, regions,
                                    |w, cx, &r| enc_region(w, cx, r));
        }
    }
}

fn enc_region(w: &mut MemWriter, cx: &ctxt, r: ty::Region) {
    match r {
        ty::ReLateBound(id, br) => {
            mywrite!(w, "b[{}|", id);
            enc_bound_region(w, cx, br);
            mywrite!(w, "]");
        }
        ty::ReEarlyBound(node_id, space, index, name) => {
            mywrite!(w, "B[{}|{}|{}|{}]",
                     node_id,
                     space.to_uint(),
                     index,
                     token::get_name(name));
        }
        ty::ReFree(ref fr) => {
            mywrite!(w, "f[{}|", fr.scope_id);
            enc_bound_region(w, cx, fr.bound_region);
            mywrite!(w, "]");
        }
        ty::ReScope(nid) => {
            mywrite!(w, "s{}|", nid);
        }
        ty::ReStatic => {
            mywrite!(w, "t");
        }
        ty::ReEmpty => {
            mywrite!(w, "e");
        }
        ty::ReInfer(_) => {
            // these should not crop up after typeck
            cx.diag.handler().bug("cannot encode region variables");
        }
    }
}

fn enc_bound_region(w: &mut MemWriter, cx: &ctxt, br: ty::BoundRegion) {
    match br {
        ty::BrAnon(idx) => {
            mywrite!(w, "a{}|", idx);
        }
        ty::BrNamed(d, name) => {
            mywrite!(w, "[{}|{}]",
                     (cx.ds)(d),
                     token::get_name(name));
        }
        ty::BrFresh(id) => {
            mywrite!(w, "f{}|", id);
        }
    }
}

pub fn enc_trait_ref(w: &mut MemWriter, cx: &ctxt, s: &ty::TraitRef) {
    mywrite!(w, "{}|", (cx.ds)(s.def_id));
    enc_substs(w, cx, &s.substs);
}

pub fn enc_trait_store(w: &mut MemWriter, cx: &ctxt, s: ty::TraitStore) {
    match s {
        ty::UniqTraitStore => mywrite!(w, "~"),
        ty::RegionTraitStore(re, m) => {
            mywrite!(w, "&");
            enc_region(w, cx, re);
            enc_mutability(w, m);
        }
    }
}

fn enc_sty(w: &mut MemWriter, cx: &ctxt, st: &ty::sty) {
    match *st {
        ty::ty_nil => mywrite!(w, "n"),
        ty::ty_bot => mywrite!(w, "z"),
        ty::ty_bool => mywrite!(w, "b"),
        ty::ty_char => mywrite!(w, "c"),
        ty::ty_int(t) => {
            match t {
                TyI => mywrite!(w, "i"),
                TyI8 => mywrite!(w, "MB"),
                TyI16 => mywrite!(w, "MW"),
                TyI32 => mywrite!(w, "ML"),
                TyI64 => mywrite!(w, "MD")
            }
        }
        ty::ty_uint(t) => {
            match t {
                TyU => mywrite!(w, "u"),
                TyU8 => mywrite!(w, "Mb"),
                TyU16 => mywrite!(w, "Mw"),
                TyU32 => mywrite!(w, "Ml"),
                TyU64 => mywrite!(w, "Md")
            }
        }
        ty::ty_float(t) => {
            match t {
                TyF32 => mywrite!(w, "Mf"),
                TyF64 => mywrite!(w, "MF"),
                TyF128 => mywrite!(w, "MQ")
            }
        }
        ty::ty_enum(def, ref substs) => {
            mywrite!(w, "t[{}|", (cx.ds)(def));
            enc_substs(w, cx, substs);
            mywrite!(w, "]");
        }
        ty::ty_trait(box ty::TyTrait {
                def_id,
                ref substs,
                store,
                bounds
            }) => {
            mywrite!(w, "x[{}|", (cx.ds)(def_id));
            enc_substs(w, cx, substs);
            enc_trait_store(w, cx, store);
            let bounds = ty::ParamBounds {builtin_bounds: bounds,
                                          trait_bounds: Vec::new()};
            enc_bounds(w, cx, &bounds);
            mywrite!(w, "]");
        }
        ty::ty_tup(ref ts) => {
            mywrite!(w, "T[");
            for t in ts.iter() { enc_ty(w, cx, *t); }
            mywrite!(w, "]");
        }
        ty::ty_box(typ) => { mywrite!(w, "@"); enc_ty(w, cx, typ); }
        ty::ty_uniq(typ) => { mywrite!(w, "~"); enc_ty(w, cx, typ); }
        ty::ty_ptr(mt) => { mywrite!(w, "*"); enc_mt(w, cx, mt); }
        ty::ty_rptr(r, mt) => {
            mywrite!(w, "&");
            enc_region(w, cx, r);
            enc_mt(w, cx, mt);
        }
        ty::ty_vec(mt, sz) => {
            mywrite!(w, "V");
            enc_mt(w, cx, mt);
            mywrite!(w, "/");
            match sz {
                Some(n) => mywrite!(w, "{}|", n),
                None => mywrite!(w, "|"),
            }
        }
        ty::ty_str => {
            mywrite!(w, "v");
        }
        ty::ty_closure(ref f) => {
            mywrite!(w, "f");
            enc_closure_ty(w, cx, *f);
        }
        ty::ty_bare_fn(ref f) => {
            mywrite!(w, "F");
            enc_bare_fn_ty(w, cx, f);
        }
        ty::ty_infer(_) => {
            cx.diag.handler().bug("cannot encode inference variable types");
        }
        ty::ty_param(ParamTy {space, idx: id, def_id: did}) => {
            mywrite!(w, "p{}|{}|{}|", (cx.ds)(did), id, space.to_uint())
        }
        ty::ty_struct(def, ref substs) => {
            mywrite!(w, "a[{}|", (cx.ds)(def));
            enc_substs(w, cx, substs);
            mywrite!(w, "]");
        }
        ty::ty_err => {
            mywrite!(w, "e");
        }
    }
}

fn enc_fn_style(w: &mut MemWriter, p: FnStyle) {
    match p {
        NormalFn => mywrite!(w, "n"),
        UnsafeFn => mywrite!(w, "u"),
    }
}

fn enc_abi(w: &mut MemWriter, abi: Abi) {
    mywrite!(w, "[");
    mywrite!(w, "{}", abi.name());
    mywrite!(w, "]")
}

fn enc_onceness(w: &mut MemWriter, o: Onceness) {
    match o {
        Once => mywrite!(w, "o"),
        Many => mywrite!(w, "m")
    }
}

pub fn enc_bare_fn_ty(w: &mut MemWriter, cx: &ctxt, ft: &ty::BareFnTy) {
    enc_fn_style(w, ft.fn_style);
    enc_abi(w, ft.abi);
    enc_fn_sig(w, cx, &ft.sig);
}

fn enc_closure_ty(w: &mut MemWriter, cx: &ctxt, ft: &ty::ClosureTy) {
    enc_fn_style(w, ft.fn_style);
    enc_onceness(w, ft.onceness);
    enc_trait_store(w, cx, ft.store);
    let bounds = ty::ParamBounds {builtin_bounds: ft.bounds,
                                  trait_bounds: Vec::new()};
    enc_bounds(w, cx, &bounds);
    enc_fn_sig(w, cx, &ft.sig);
}

fn enc_fn_sig(w: &mut MemWriter, cx: &ctxt, fsig: &ty::FnSig) {
    mywrite!(w, "[{}|", fsig.binder_id);
    for ty in fsig.inputs.iter() {
        enc_ty(w, cx, *ty);
    }
    mywrite!(w, "]");
    if fsig.variadic {
        mywrite!(w, "V");
    } else {
        mywrite!(w, "N");
    }
    enc_ty(w, cx, fsig.output);
}

fn enc_bounds(w: &mut MemWriter, cx: &ctxt, bs: &ty::ParamBounds) {
    for bound in bs.builtin_bounds.iter() {
        match bound {
            ty::BoundSend => mywrite!(w, "S"),
            ty::BoundStatic => mywrite!(w, "O"),
            ty::BoundSized => mywrite!(w, "Z"),
            ty::BoundCopy => mywrite!(w, "P"),
            ty::BoundShare => mywrite!(w, "T"),
        }
    }

    for tp in bs.trait_bounds.iter() {
        mywrite!(w, "I");
        enc_trait_ref(w, cx, &**tp);
    }

    mywrite!(w, ".");
}

pub fn enc_type_param_def(w: &mut MemWriter, cx: &ctxt, v: &ty::TypeParameterDef) {
    mywrite!(w, "{}:{}|{}|{}|",
             token::get_ident(v.ident), (cx.ds)(v.def_id),
             v.space.to_uint(), v.index);
    enc_bounds(w, cx, &*v.bounds);
    enc_opt(w, v.default, |w, t| enc_ty(w, cx, t));
}
