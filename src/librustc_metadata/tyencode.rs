// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
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
use std::io::Cursor;
use std::io::prelude::*;

use middle::def_id::DefId;
use middle::region;
use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty::ParamTy;
use middle::ty::{self, Ty};
use rustc::util::nodemap::FnvHashMap;

use rustc_front::hir;

use syntax::abi::Abi;
use syntax::ast;
use syntax::errors::Handler;

use rbml::leb128;
use encoder;

pub struct ctxt<'a, 'tcx: 'a> {
    pub diag: &'a Handler,
    // Def -> str Callback:
    pub ds: fn(DefId) -> String,
    // The type context.
    pub tcx: &'a ty::ctxt<'tcx>,
    pub abbrevs: &'a abbrev_map<'tcx>
}

impl<'a, 'tcx> encoder::EncodeContext<'a, 'tcx> {
    pub fn ty_str_ctxt<'b>(&'b self) -> ctxt<'b, 'tcx> {
        ctxt {
            diag: self.tcx.sess.diagnostic(),
            ds: encoder::def_to_string,
            tcx: self.tcx,
            abbrevs: &self.type_abbrevs
        }
    }
}

// Compact string representation for Ty values. API TyStr & parse_from_str.
// Extra parameters are for converting to/from def_ids in the string rep.
// Whatever format you choose should not contain pipe characters.
pub struct ty_abbrev {
    s: Vec<u8>
}

pub type abbrev_map<'tcx> = RefCell<FnvHashMap<Ty<'tcx>, ty_abbrev>>;

pub fn enc_ty<'a, 'tcx>(w: &mut Cursor<Vec<u8>>, cx: &ctxt<'a, 'tcx>, t: Ty<'tcx>) {
    match cx.abbrevs.borrow_mut().get(&t) {
        Some(a) => { w.write_all(&a.s); return; }
        None => {}
    }

    let pos = w.position();

    match t.sty {
        ty::TyBool => { write!(w, "b"); }
        ty::TyChar => { write!(w, "c"); }
        ty::TyInt(t) => {
            match t {
                ast::TyIs => write!(w, "is"),
                ast::TyI8 => write!(w, "MB"),
                ast::TyI16 => write!(w, "MW"),
                ast::TyI32 => write!(w, "ML"),
                ast::TyI64 => write!(w, "MD")
            };
        }
        ty::TyUint(t) => {
            match t {
                ast::TyUs => write!(w, "us"),
                ast::TyU8 => write!(w, "Mb"),
                ast::TyU16 => write!(w, "Mw"),
                ast::TyU32 => write!(w, "Ml"),
                ast::TyU64 => write!(w, "Md")
            };
        }
        ty::TyFloat(t) => {
            match t {
                ast::TyF32 => write!(w, "Mf"),
                ast::TyF64 => write!(w, "MF"),
            };
        }
        ty::TyEnum(def, substs) => {
            write!(w, "t[{}|", (cx.ds)(def.did));
            enc_substs(w, cx, substs);
            write!(w, "]");
        }
        ty::TyTrait(box ty::TraitTy { ref principal,
                                       ref bounds }) => {
            write!(w, "x[");
            enc_trait_ref(w, cx, principal.0);
            enc_existential_bounds(w, cx, bounds);
            write!(w, "]");
        }
        ty::TyTuple(ref ts) => {
            write!(w, "T[");
            for t in ts { enc_ty(w, cx, *t); }
            write!(w, "]");
        }
        ty::TyBox(typ) => { write!(w, "~"); enc_ty(w, cx, typ); }
        ty::TyRawPtr(mt) => { write!(w, "*"); enc_mt(w, cx, mt); }
        ty::TyRef(r, mt) => {
            write!(w, "&");
            enc_region(w, cx, *r);
            enc_mt(w, cx, mt);
        }
        ty::TyArray(t, sz) => {
            write!(w, "V");
            enc_ty(w, cx, t);
            write!(w, "/{}|", sz);
        }
        ty::TySlice(t) => {
            write!(w, "V");
            enc_ty(w, cx, t);
            write!(w, "/|");
        }
        ty::TyStr => {
            write!(w, "v");
        }
        ty::TyBareFn(Some(def_id), f) => {
            write!(w, "F");
            write!(w, "{}|", (cx.ds)(def_id));
            enc_bare_fn_ty(w, cx, f);
        }
        ty::TyBareFn(None, f) => {
            write!(w, "G");
            enc_bare_fn_ty(w, cx, f);
        }
        ty::TyInfer(_) => {
            cx.diag.bug("cannot encode inference variable types");
        }
        ty::TyParam(ParamTy {space, idx, name}) => {
            write!(w, "p[{}|{}|{}]", idx, space.to_uint(), name);
        }
        ty::TyStruct(def, substs) => {
            write!(w, "a[{}|", (cx.ds)(def.did));
            enc_substs(w, cx, substs);
            write!(w, "]");
        }
        ty::TyClosure(def, ref substs) => {
            write!(w, "k[{}|", (cx.ds)(def));
            enc_substs(w, cx, &substs.func_substs);
            for ty in &substs.upvar_tys {
                enc_ty(w, cx, ty);
            }
            write!(w, ".");
            write!(w, "]");
        }
        ty::TyProjection(ref data) => {
            write!(w, "P[");
            enc_trait_ref(w, cx, data.trait_ref);
            write!(w, "{}]", data.item_name);
        }
        ty::TyError => {
            write!(w, "e");
        }
    }

    let end = w.position();
    let len = end - pos;

    let mut abbrev = Cursor::new(Vec::with_capacity(16));
    abbrev.write_all(b"#");
    {
        let start_position = abbrev.position() as usize;
        let bytes_written = leb128::write_unsigned_leb128(abbrev.get_mut(),
                                                          start_position,
                                                          pos);
        abbrev.set_position((start_position + bytes_written) as u64);
    }

    cx.abbrevs.borrow_mut().insert(t, ty_abbrev {
        s: if abbrev.position() < len {
            abbrev.get_ref()[..abbrev.position() as usize].to_owned()
        } else {
            // if the abbreviation is longer than the real type,
            // don't use #-notation. However, insert it here so
            // other won't have to `mark_stable_position`
            w.get_ref()[pos as usize .. end as usize].to_owned()
        }
    });
}

fn enc_mutability(w: &mut Cursor<Vec<u8>>, mt: hir::Mutability) {
    match mt {
        hir::MutImmutable => (),
        hir::MutMutable => {
            write!(w, "m");
        }
    };
}

fn enc_mt<'a, 'tcx>(w: &mut Cursor<Vec<u8>>, cx: &ctxt<'a, 'tcx>,
                    mt: ty::TypeAndMut<'tcx>) {
    enc_mutability(w, mt.mutbl);
    enc_ty(w, cx, mt.ty);
}

fn enc_opt<T, F>(w: &mut Cursor<Vec<u8>>, t: Option<T>, enc_f: F) where
    F: FnOnce(&mut Cursor<Vec<u8>>, T),
{
    match t {
        None => {
            write!(w, "n");
        }
        Some(v) => {
            write!(w, "s");
            enc_f(w, v);
        }
    }
}

fn enc_vec_per_param_space<'a, 'tcx, T, F>(w: &mut Cursor<Vec<u8>>,
                                           cx: &ctxt<'a, 'tcx>,
                                           v: &VecPerParamSpace<T>,
                                           mut op: F) where
    F: FnMut(&mut Cursor<Vec<u8>>, &ctxt<'a, 'tcx>, &T),
{
    for &space in &subst::ParamSpace::all() {
        write!(w, "[");
        for t in v.get_slice(space) {
            op(w, cx, t);
        }
        write!(w, "]");
    }
}

pub fn enc_substs<'a, 'tcx>(w: &mut Cursor<Vec<u8>>, cx: &ctxt<'a, 'tcx>,
                            substs: &subst::Substs<'tcx>) {
    enc_region_substs(w, cx, &substs.regions);
    enc_vec_per_param_space(w, cx, &substs.types,
                            |w, cx, &ty| enc_ty(w, cx, ty));
}

fn enc_region_substs(w: &mut Cursor<Vec<u8>>, cx: &ctxt, substs: &subst::RegionSubsts) {
    match *substs {
        subst::ErasedRegions => {
            write!(w, "e");
        }
        subst::NonerasedRegions(ref regions) => {
            write!(w, "n");
            enc_vec_per_param_space(w, cx, regions,
                                    |w, cx, &r| enc_region(w, cx, r));
        }
    }
}

pub fn enc_region(w: &mut Cursor<Vec<u8>>, cx: &ctxt, r: ty::Region) {
    match r {
        ty::ReLateBound(id, br) => {
            write!(w, "b[{}|", id.depth);
            enc_bound_region(w, cx, br);
            write!(w, "]");
        }
        ty::ReEarlyBound(ref data) => {
            write!(w, "B[{}|{}|{}]",
                   data.space.to_uint(),
                   data.index,
                   data.name);
        }
        ty::ReFree(ref fr) => {
            write!(w, "f[");
            enc_scope(w, cx, fr.scope);
            write!(w, "|");
            enc_bound_region(w, cx, fr.bound_region);
            write!(w, "]");
        }
        ty::ReScope(scope) => {
            write!(w, "s");
            enc_scope(w, cx, scope);
            write!(w, "|");
        }
        ty::ReStatic => {
            write!(w, "t");
        }
        ty::ReEmpty => {
            write!(w, "e");
        }
        ty::ReVar(_) | ty::ReSkolemized(..) => {
            // these should not crop up after typeck
            cx.diag.bug("cannot encode region variables");
        }
    }
}

fn enc_scope(w: &mut Cursor<Vec<u8>>, cx: &ctxt, scope: region::CodeExtent) {
    match cx.tcx.region_maps.code_extent_data(scope) {
        region::CodeExtentData::CallSiteScope {
            fn_id, body_id } => write!(w, "C[{}|{}]", fn_id, body_id),
        region::CodeExtentData::ParameterScope {
            fn_id, body_id } => write!(w, "P[{}|{}]", fn_id, body_id),
        region::CodeExtentData::Misc(node_id) => write!(w, "M{}", node_id),
        region::CodeExtentData::Remainder(region::BlockRemainder {
            block: b, first_statement_index: i }) => write!(w, "B[{}|{}]", b, i),
        region::CodeExtentData::DestructionScope(node_id) => write!(w, "D{}", node_id),
    };
}

fn enc_bound_region(w: &mut Cursor<Vec<u8>>, cx: &ctxt, br: ty::BoundRegion) {
    match br {
        ty::BrAnon(idx) => {
            write!(w, "a{}|", idx);
        }
        ty::BrNamed(d, name) => {
            write!(w, "[{}|{}]",
                     (cx.ds)(d),
                     name);
        }
        ty::BrFresh(id) => {
            write!(w, "f{}|", id);
        }
        ty::BrEnv => {
            write!(w, "e|");
        }
    }
}

pub fn enc_trait_ref<'a, 'tcx>(w: &mut Cursor<Vec<u8>>, cx: &ctxt<'a, 'tcx>,
                               s: ty::TraitRef<'tcx>) {
    write!(w, "{}|", (cx.ds)(s.def_id));
    enc_substs(w, cx, s.substs);
}

fn enc_unsafety(w: &mut Cursor<Vec<u8>>, p: hir::Unsafety) {
    match p {
        hir::Unsafety::Normal => write!(w, "n"),
        hir::Unsafety::Unsafe => write!(w, "u"),
    };
}

fn enc_abi(w: &mut Cursor<Vec<u8>>, abi: Abi) {
    write!(w, "[");
    write!(w, "{}", abi.name());
    write!(w, "]");
}

pub fn enc_bare_fn_ty<'a, 'tcx>(w: &mut Cursor<Vec<u8>>, cx: &ctxt<'a, 'tcx>,
                                ft: &ty::BareFnTy<'tcx>) {
    enc_unsafety(w, ft.unsafety);
    enc_abi(w, ft.abi);
    enc_fn_sig(w, cx, &ft.sig);
}

pub fn enc_closure_ty<'a, 'tcx>(w: &mut Cursor<Vec<u8>>, cx: &ctxt<'a, 'tcx>,
                                ft: &ty::ClosureTy<'tcx>) {
    enc_unsafety(w, ft.unsafety);
    enc_fn_sig(w, cx, &ft.sig);
    enc_abi(w, ft.abi);
}

fn enc_fn_sig<'a, 'tcx>(w: &mut Cursor<Vec<u8>>, cx: &ctxt<'a, 'tcx>,
                        fsig: &ty::PolyFnSig<'tcx>) {
    write!(w, "[");
    for ty in &fsig.0.inputs {
        enc_ty(w, cx, *ty);
    }
    write!(w, "]");
    if fsig.0.variadic {
        write!(w, "V");
    } else {
        write!(w, "N");
    }
    match fsig.0.output {
        ty::FnConverging(result_type) => {
            enc_ty(w, cx, result_type);
        }
        ty::FnDiverging => {
            write!(w, "z");
        }
    }
}

pub fn enc_builtin_bounds(w: &mut Cursor<Vec<u8>>, _cx: &ctxt, bs: &ty::BuiltinBounds) {
    for bound in bs {
        match bound {
            ty::BoundSend => write!(w, "S"),
            ty::BoundSized => write!(w, "Z"),
            ty::BoundCopy => write!(w, "P"),
            ty::BoundSync => write!(w, "T"),
        };
    }

    write!(w, ".");
}

pub fn enc_existential_bounds<'a,'tcx>(w: &mut Cursor<Vec<u8>>,
                                       cx: &ctxt<'a,'tcx>,
                                       bs: &ty::ExistentialBounds<'tcx>) {
    enc_builtin_bounds(w, cx, &bs.builtin_bounds);

    enc_region(w, cx, bs.region_bound);

    for tp in &bs.projection_bounds {
        write!(w, "P");
        enc_projection_predicate(w, cx, &tp.0);
    }

    write!(w, ".");
}

pub fn enc_type_param_def<'a, 'tcx>(w: &mut Cursor<Vec<u8>>, cx: &ctxt<'a, 'tcx>,
                                    v: &ty::TypeParameterDef<'tcx>) {
    write!(w, "{}:{}|{}|{}|{}|",
             v.name, (cx.ds)(v.def_id),
             v.space.to_uint(), v.index, (cx.ds)(v.default_def_id));
    enc_opt(w, v.default, |w, t| enc_ty(w, cx, t));
    enc_object_lifetime_default(w, cx, v.object_lifetime_default);
}

pub fn enc_region_param_def(w: &mut Cursor<Vec<u8>>, cx: &ctxt,
                            v: &ty::RegionParameterDef) {
    write!(w, "{}:{}|{}|{}|",
             v.name, (cx.ds)(v.def_id),
             v.space.to_uint(), v.index);
    for &r in &v.bounds {
        write!(w, "R");
        enc_region(w, cx, r);
    }
    write!(w, ".");
}

fn enc_object_lifetime_default<'a, 'tcx>(w: &mut Cursor<Vec<u8>>,
                                         cx: &ctxt<'a, 'tcx>,
                                         default: ty::ObjectLifetimeDefault)
{
    match default {
        ty::ObjectLifetimeDefault::Ambiguous => {
            write!(w, "a");
        }
        ty::ObjectLifetimeDefault::BaseDefault => {
            write!(w, "b");
        }
        ty::ObjectLifetimeDefault::Specific(r) => {
            write!(w, "s");
            enc_region(w, cx, r);
        }
    }
}

pub fn enc_predicate<'a, 'tcx>(w: &mut Cursor<Vec<u8>>,
                               cx: &ctxt<'a, 'tcx>,
                               p: &ty::Predicate<'tcx>)
{
    match *p {
        ty::Predicate::Trait(ref trait_ref) => {
            write!(w, "t");
            enc_trait_ref(w, cx, trait_ref.0.trait_ref);
        }
        ty::Predicate::Equate(ty::Binder(ty::EquatePredicate(a, b))) => {
            write!(w, "e");
            enc_ty(w, cx, a);
            enc_ty(w, cx, b);
        }
        ty::Predicate::RegionOutlives(ty::Binder(ty::OutlivesPredicate(a, b))) => {
            write!(w, "r");
            enc_region(w, cx, a);
            enc_region(w, cx, b);
        }
        ty::Predicate::TypeOutlives(ty::Binder(ty::OutlivesPredicate(a, b))) => {
            write!(w, "o");
            enc_ty(w, cx, a);
            enc_region(w, cx, b);
        }
        ty::Predicate::Projection(ty::Binder(ref data)) => {
            write!(w, "p");
            enc_projection_predicate(w, cx, data);
        }
        ty::Predicate::WellFormed(data) => {
            write!(w, "w");
            enc_ty(w, cx, data);
        }
        ty::Predicate::ObjectSafe(trait_def_id) => {
            write!(w, "O{}|", (cx.ds)(trait_def_id));
        }
    }
}

fn enc_projection_predicate<'a, 'tcx>(w: &mut Cursor<Vec<u8>>,
                                      cx: &ctxt<'a, 'tcx>,
                                      data: &ty::ProjectionPredicate<'tcx>) {
    enc_trait_ref(w, cx, data.projection_ty.trait_ref);
    write!(w, "{}|", data.projection_ty.item_name);
    enc_ty(w, cx, data.ty);
}
