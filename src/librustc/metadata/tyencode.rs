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

use middle::region;
use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty::ParamTy;
use middle::ty::{self, Ty};
use util::nodemap::FnvHashMap;

use syntax::abi::Abi;
use syntax::ast;
use syntax::diagnostic::SpanHandler;
use syntax::parse::token;

use rbml::io::SeekableMemWriter;

macro_rules! mywrite { ($($arg:tt)*) => ({ write!($($arg)*); }) }

pub struct ctxt<'a, 'tcx: 'a> {
    pub diag: &'a SpanHandler,
    // Def -> str Callback:
    pub ds: fn(ast::DefId) -> String,
    // The type context.
    pub tcx: &'a ty::ctxt<'tcx>,
    pub abbrevs: &'a abbrev_map<'tcx>
}

// Compact string representation for Ty values. API ty_str & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.
pub struct ty_abbrev {
    s: String
}

pub type abbrev_map<'tcx> = RefCell<FnvHashMap<Ty<'tcx>, ty_abbrev>>;

pub fn enc_ty<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>, t: Ty<'tcx>) {
    match cx.abbrevs.borrow_mut().get(&t) {
        Some(a) => { w.write(a.s.as_bytes()); return; }
        None => {}
    }
    let pos = w.tell().unwrap();

    match t.sty {
        ty::ty_bool => mywrite!(w, "b"),
        ty::ty_char => mywrite!(w, "c"),
        ty::ty_int(t) => {
            match t {
                ast::TyIs(_) => mywrite!(w, "is"),
                ast::TyI8 => mywrite!(w, "MB"),
                ast::TyI16 => mywrite!(w, "MW"),
                ast::TyI32 => mywrite!(w, "ML"),
                ast::TyI64 => mywrite!(w, "MD")
            }
        }
        ty::ty_uint(t) => {
            match t {
                ast::TyUs(_) => mywrite!(w, "us"),
                ast::TyU8 => mywrite!(w, "Mb"),
                ast::TyU16 => mywrite!(w, "Mw"),
                ast::TyU32 => mywrite!(w, "Ml"),
                ast::TyU64 => mywrite!(w, "Md")
            }
        }
        ty::ty_float(t) => {
            match t {
                ast::TyF32 => mywrite!(w, "Mf"),
                ast::TyF64 => mywrite!(w, "MF"),
            }
        }
        ty::ty_enum(def, substs) => {
            mywrite!(w, "t[{}|", (cx.ds)(def));
            enc_substs(w, cx, substs);
            mywrite!(w, "]");
        }
        ty::ty_trait(box ty::TyTrait { ref principal,
                                       ref bounds }) => {
            mywrite!(w, "x[");
            enc_trait_ref(w, cx, &*principal.0);
            enc_existential_bounds(w, cx, bounds);
            mywrite!(w, "]");
        }
        ty::ty_tup(ref ts) => {
            mywrite!(w, "T[");
            for t in ts.iter() { enc_ty(w, cx, *t); }
            mywrite!(w, "]");
        }
        ty::ty_uniq(typ) => { mywrite!(w, "~"); enc_ty(w, cx, typ); }
        ty::ty_ptr(mt) => { mywrite!(w, "*"); enc_mt(w, cx, mt); }
        ty::ty_rptr(r, mt) => {
            mywrite!(w, "&");
            enc_region(w, cx, *r);
            enc_mt(w, cx, mt);
        }
        ty::ty_vec(t, sz) => {
            mywrite!(w, "V");
            enc_ty(w, cx, t);
            mywrite!(w, "/");
            match sz {
                Some(n) => mywrite!(w, "{}|", n),
                None => mywrite!(w, "|"),
            }
        }
        ty::ty_str => {
            mywrite!(w, "v");
        }
        ty::ty_bare_fn(Some(def_id), f) => {
            mywrite!(w, "F");
            mywrite!(w, "{}|", (cx.ds)(def_id));
            enc_bare_fn_ty(w, cx, f);
        }
        ty::ty_bare_fn(None, f) => {
            mywrite!(w, "G");
            enc_bare_fn_ty(w, cx, f);
        }
        ty::ty_infer(_) => {
            cx.diag.handler().bug("cannot encode inference variable types");
        }
        ty::ty_param(ParamTy {space, idx, name}) => {
            mywrite!(w, "p[{}|{}|{}]", idx, space.to_uint(), token::get_name(name))
        }
        ty::ty_struct(def, substs) => {
            mywrite!(w, "a[{}|", (cx.ds)(def));
            enc_substs(w, cx, substs);
            mywrite!(w, "]");
        }
        ty::ty_unboxed_closure(def, region, substs) => {
            mywrite!(w, "k[{}|", (cx.ds)(def));
            enc_region(w, cx, *region);
            enc_substs(w, cx, substs);
            mywrite!(w, "]");
        }
        ty::ty_projection(ref data) => {
            mywrite!(w, "P[");
            enc_trait_ref(w, cx, &*data.trait_ref);
            mywrite!(w, "{}]", token::get_name(data.item_name));
        }
        ty::ty_err => {
            mywrite!(w, "e");
        }
        ty::ty_open(_) => {
            cx.diag.handler().bug("unexpected type in enc_sty (ty_open)");
        }
    }

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

fn enc_mutability(w: &mut SeekableMemWriter, mt: ast::Mutability) {
    match mt {
        ast::MutImmutable => (),
        ast::MutMutable => mywrite!(w, "m"),
    }
}

fn enc_mt<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>,
                    mt: ty::mt<'tcx>) {
    enc_mutability(w, mt.mutbl);
    enc_ty(w, cx, mt.ty);
}

fn enc_opt<T, F>(w: &mut SeekableMemWriter, t: Option<T>, enc_f: F) where
    F: FnOnce(&mut SeekableMemWriter, T),
{
    match t {
        None => mywrite!(w, "n"),
        Some(v) => {
            mywrite!(w, "s");
            enc_f(w, v);
        }
    }
}

fn enc_vec_per_param_space<'a, 'tcx, T, F>(w: &mut SeekableMemWriter,
                                           cx: &ctxt<'a, 'tcx>,
                                           v: &VecPerParamSpace<T>,
                                           mut op: F) where
    F: FnMut(&mut SeekableMemWriter, &ctxt<'a, 'tcx>, &T),
{
    for &space in subst::ParamSpace::all().iter() {
        mywrite!(w, "[");
        for t in v.get_slice(space).iter() {
            op(w, cx, t);
        }
        mywrite!(w, "]");
    }
}

pub fn enc_substs<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>,
                            substs: &subst::Substs<'tcx>) {
    enc_region_substs(w, cx, &substs.regions);
    enc_vec_per_param_space(w, cx, &substs.types,
                            |w, cx, &ty| enc_ty(w, cx, ty));
}

fn enc_region_substs(w: &mut SeekableMemWriter, cx: &ctxt, substs: &subst::RegionSubsts) {
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

pub fn enc_region(w: &mut SeekableMemWriter, cx: &ctxt, r: ty::Region) {
    match r {
        ty::ReLateBound(id, br) => {
            mywrite!(w, "b[{}|", id.depth);
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
            mywrite!(w, "f[");
            enc_scope(w, cx, fr.scope);
            mywrite!(w, "|");
            enc_bound_region(w, cx, fr.bound_region);
            mywrite!(w, "]");
        }
        ty::ReScope(scope) => {
            mywrite!(w, "s");
            enc_scope(w, cx, scope);
            mywrite!(w, "|");
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

fn enc_scope(w: &mut SeekableMemWriter, _cx: &ctxt, scope: region::CodeExtent) {
    match scope {
        region::CodeExtent::Misc(node_id) => mywrite!(w, "M{}", node_id)
    }
}

fn enc_bound_region(w: &mut SeekableMemWriter, cx: &ctxt, br: ty::BoundRegion) {
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
        ty::BrEnv => {
            mywrite!(w, "e|");
        }
    }
}

pub fn enc_trait_ref<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>,
                               s: &ty::TraitRef<'tcx>) {
    mywrite!(w, "{}|", (cx.ds)(s.def_id));
    enc_substs(w, cx, s.substs);
}

pub fn enc_trait_store(w: &mut SeekableMemWriter, cx: &ctxt, s: ty::TraitStore) {
    match s {
        ty::UniqTraitStore => mywrite!(w, "~"),
        ty::RegionTraitStore(re, m) => {
            mywrite!(w, "&");
            enc_region(w, cx, re);
            enc_mutability(w, m);
        }
    }
}

fn enc_unsafety(w: &mut SeekableMemWriter, p: ast::Unsafety) {
    match p {
        ast::Unsafety::Normal => mywrite!(w, "n"),
        ast::Unsafety::Unsafe => mywrite!(w, "u"),
    }
}

fn enc_abi(w: &mut SeekableMemWriter, abi: Abi) {
    mywrite!(w, "[");
    mywrite!(w, "{}", abi.name());
    mywrite!(w, "]")
}

fn enc_onceness(w: &mut SeekableMemWriter, o: ast::Onceness) {
    match o {
        ast::Once => mywrite!(w, "o"),
        ast::Many => mywrite!(w, "m")
    }
}

pub fn enc_bare_fn_ty<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>,
                                ft: &ty::BareFnTy<'tcx>) {
    enc_unsafety(w, ft.unsafety);
    enc_abi(w, ft.abi);
    enc_fn_sig(w, cx, &ft.sig);
}

pub fn enc_closure_ty<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>,
                                ft: &ty::ClosureTy<'tcx>) {
    enc_unsafety(w, ft.unsafety);
    enc_onceness(w, ft.onceness);
    enc_trait_store(w, cx, ft.store);
    enc_existential_bounds(w, cx, &ft.bounds);
    enc_fn_sig(w, cx, &ft.sig);
    enc_abi(w, ft.abi);
}

fn enc_fn_sig<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>,
                        fsig: &ty::PolyFnSig<'tcx>) {
    mywrite!(w, "[");
    for ty in fsig.0.inputs.iter() {
        enc_ty(w, cx, *ty);
    }
    mywrite!(w, "]");
    if fsig.0.variadic {
        mywrite!(w, "V");
    } else {
        mywrite!(w, "N");
    }
    match fsig.0.output {
        ty::FnConverging(result_type) => {
            enc_ty(w, cx, result_type);
        }
        ty::FnDiverging => {
            mywrite!(w, "z");
        }
    }
}

pub fn enc_builtin_bounds(w: &mut SeekableMemWriter, _cx: &ctxt, bs: &ty::BuiltinBounds) {
    for bound in bs.iter() {
        match bound {
            ty::BoundSend => mywrite!(w, "S"),
            ty::BoundSized => mywrite!(w, "Z"),
            ty::BoundCopy => mywrite!(w, "P"),
            ty::BoundSync => mywrite!(w, "T"),
        }
    }

    mywrite!(w, ".");
}

pub fn enc_existential_bounds<'a,'tcx>(w: &mut SeekableMemWriter,
                                       cx: &ctxt<'a,'tcx>,
                                       bs: &ty::ExistentialBounds<'tcx>) {
    let param_bounds = ty::ParamBounds { trait_bounds: vec!(),
                                         region_bounds: vec!(bs.region_bound),
                                         builtin_bounds: bs.builtin_bounds,
                                         projection_bounds: bs.projection_bounds.clone() };
    enc_bounds(w, cx, &param_bounds);
}

pub fn enc_bounds<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>,
                            bs: &ty::ParamBounds<'tcx>) {
    enc_builtin_bounds(w, cx, &bs.builtin_bounds);

    for &r in bs.region_bounds.iter() {
        mywrite!(w, "R");
        enc_region(w, cx, r);
    }

    for tp in bs.trait_bounds.iter() {
        mywrite!(w, "I");
        enc_trait_ref(w, cx, &*tp.0);
    }

    for tp in bs.projection_bounds.iter() {
        mywrite!(w, "P");
        enc_projection_predicate(w, cx, &tp.0);
    }

    mywrite!(w, ".");
}

pub fn enc_type_param_def<'a, 'tcx>(w: &mut SeekableMemWriter, cx: &ctxt<'a, 'tcx>,
                                    v: &ty::TypeParameterDef<'tcx>) {
    mywrite!(w, "{}:{}|{}|{}|",
             token::get_name(v.name), (cx.ds)(v.def_id),
             v.space.to_uint(), v.index);
    enc_bounds(w, cx, &v.bounds);
    enc_opt(w, v.default, |w, t| enc_ty(w, cx, t));
}

pub fn enc_predicate<'a, 'tcx>(w: &mut SeekableMemWriter,
                               cx: &ctxt<'a, 'tcx>,
                               p: &ty::Predicate<'tcx>)
{
    match *p {
        ty::Predicate::Trait(ref trait_ref) => {
            mywrite!(w, "t");
            enc_trait_ref(w, cx, &*trait_ref.0.trait_ref);
        }
        ty::Predicate::Equate(ty::Binder(ty::EquatePredicate(a, b))) => {
            mywrite!(w, "e");
            enc_ty(w, cx, a);
            enc_ty(w, cx, b);
        }
        ty::Predicate::RegionOutlives(ty::Binder(ty::OutlivesPredicate(a, b))) => {
            mywrite!(w, "r");
            enc_region(w, cx, a);
            enc_region(w, cx, b);
        }
        ty::Predicate::TypeOutlives(ty::Binder(ty::OutlivesPredicate(a, b))) => {
            mywrite!(w, "o");
            enc_ty(w, cx, a);
            enc_region(w, cx, b);
        }
        ty::Predicate::Projection(ty::Binder(ref data)) => {
            mywrite!(w, "p");
            enc_projection_predicate(w, cx, data)
        }
    }
}

fn enc_projection_predicate<'a, 'tcx>(w: &mut SeekableMemWriter,
                                      cx: &ctxt<'a, 'tcx>,
                                      data: &ty::ProjectionPredicate<'tcx>) {
    enc_trait_ref(w, cx, &*data.projection_ty.trait_ref);
    mywrite!(w, "{}|", token::get_name(data.projection_ty.item_name));
    enc_ty(w, cx, data.ty);
}
