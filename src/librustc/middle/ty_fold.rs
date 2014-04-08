// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Generalized type folding mechanism.

use middle::ty;
use util::ppaux::Repr;

pub trait TypeFolder {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt;

    fn fold_ty(&mut self, t: ty::t) -> ty::t {
        super_fold_ty(self, t)
    }

    fn fold_mt(&mut self, t: &ty::mt) -> ty::mt {
        super_fold_mt(self, t)
    }

    fn fold_trait_ref(&mut self, t: &ty::TraitRef) -> ty::TraitRef {
        super_fold_trait_ref(self, t)
    }

    fn fold_sty(&mut self, sty: &ty::sty) -> ty::sty {
        super_fold_sty(self, sty)
    }

    fn fold_substs(&mut self,
                   substs: &ty::substs)
                   -> ty::substs {
        super_fold_substs(self, substs)
    }

    fn fold_sig(&mut self,
                sig: &ty::FnSig)
                -> ty::FnSig {
        super_fold_sig(self, sig)
    }

    fn fold_bare_fn_ty(&mut self,
                       fty: &ty::BareFnTy)
                       -> ty::BareFnTy {
        ty::BareFnTy { sig: self.fold_sig(&fty.sig),
                       abi: fty.abi,
                       purity: fty.purity }
    }

    fn fold_closure_ty(&mut self,
                       fty: &ty::ClosureTy)
                       -> ty::ClosureTy {
        ty::ClosureTy {
            region: self.fold_region(fty.region),
            sig: self.fold_sig(&fty.sig),
            purity: fty.purity,
            sigil: fty.sigil,
            onceness: fty.onceness,
            bounds: fty.bounds,
        }
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        r
    }

    fn fold_vstore(&mut self, vstore: ty::vstore) -> ty::vstore {
        super_fold_vstore(self, vstore)
    }

    fn fold_trait_store(&mut self, s: ty::TraitStore) -> ty::TraitStore {
        super_fold_trait_store(self, s)
    }
}

pub fn fold_opt_ty<T:TypeFolder>(this: &mut T,
                                 t: Option<ty::t>)
                                 -> Option<ty::t> {
    t.map(|t| this.fold_ty(t))
}

pub fn fold_ty_vec<T:TypeFolder>(this: &mut T, tys: &[ty::t]) -> Vec<ty::t> {
    tys.iter().map(|t| this.fold_ty(*t)).collect()
}

pub fn super_fold_ty<T:TypeFolder>(this: &mut T,
                                   t: ty::t)
                                   -> ty::t {
    let sty = this.fold_sty(&ty::get(t).sty);
    ty::mk_t(this.tcx(), sty)
}

pub fn super_fold_substs<T:TypeFolder>(this: &mut T,
                                       substs: &ty::substs)
                                       -> ty::substs {
    let regions = match substs.regions {
        ty::ErasedRegions => {
            ty::ErasedRegions
        }
        ty::NonerasedRegions(ref regions) => {
            ty::NonerasedRegions(regions.map(|r| this.fold_region(*r)))
        }
    };

    ty::substs { regions: regions,
                 self_ty: fold_opt_ty(this, substs.self_ty),
                 tps: fold_ty_vec(this, substs.tps.as_slice()), }
}

pub fn super_fold_sig<T:TypeFolder>(this: &mut T,
                                    sig: &ty::FnSig)
                                    -> ty::FnSig {
    ty::FnSig { binder_id: sig.binder_id,
                inputs: fold_ty_vec(this, sig.inputs.as_slice()),
                output: this.fold_ty(sig.output),
                variadic: sig.variadic }
}

pub fn super_fold_trait_ref<T:TypeFolder>(this: &mut T,
                                          t: &ty::TraitRef)
                                          -> ty::TraitRef {
    ty::TraitRef {
        def_id: t.def_id,
        substs: this.fold_substs(&t.substs)
    }
}

pub fn super_fold_mt<T:TypeFolder>(this: &mut T,
                                   mt: &ty::mt) -> ty::mt {
    ty::mt {ty: this.fold_ty(mt.ty),
            mutbl: mt.mutbl}
}

pub fn super_fold_sty<T:TypeFolder>(this: &mut T,
                                    sty: &ty::sty) -> ty::sty {
    match *sty {
        ty::ty_box(typ) => {
            ty::ty_box(this.fold_ty(typ))
        }
        ty::ty_uniq(typ) => {
            ty::ty_uniq(this.fold_ty(typ))
        }
        ty::ty_ptr(ref tm) => {
            ty::ty_ptr(this.fold_mt(tm))
        }
        ty::ty_vec(ref tm, vst) => {
            ty::ty_vec(this.fold_mt(tm), this.fold_vstore(vst))
        }
        ty::ty_enum(tid, ref substs) => {
            ty::ty_enum(tid, this.fold_substs(substs))
        }
        ty::ty_trait(~ty::TyTrait { def_id, ref substs, store, mutability, bounds }) => {
            ty::ty_trait(~ty::TyTrait{
                def_id: def_id,
                substs: this.fold_substs(substs),
                store: this.fold_trait_store(store),
                mutability: mutability,
                bounds: bounds
            })
        }
        ty::ty_tup(ref ts) => {
            ty::ty_tup(fold_ty_vec(this, ts.as_slice()))
        }
        ty::ty_bare_fn(ref f) => {
            ty::ty_bare_fn(this.fold_bare_fn_ty(f))
        }
        ty::ty_closure(ref f) => {
            ty::ty_closure(~this.fold_closure_ty(*f))
        }
        ty::ty_rptr(r, ref tm) => {
            ty::ty_rptr(this.fold_region(r),
                        ty::mt {ty: this.fold_ty(tm.ty),
                                mutbl: tm.mutbl})
        }
        ty::ty_struct(did, ref substs) => {
            ty::ty_struct(did,
                          this.fold_substs(substs))
        }
        ty::ty_str(vst) => {
            ty::ty_str(this.fold_vstore(vst))
        }
        ty::ty_nil | ty::ty_bot | ty::ty_bool | ty::ty_char |
        ty::ty_int(_) | ty::ty_uint(_) | ty::ty_float(_) |
        ty::ty_err | ty::ty_infer(_) |
        ty::ty_param(..) | ty::ty_self(_) => {
            (*sty).clone()
        }
    }
}

pub fn super_fold_vstore<T:TypeFolder>(this: &mut T,
                                       vstore: ty::vstore)
                                       -> ty::vstore {
    match vstore {
        ty::vstore_fixed(i) => ty::vstore_fixed(i),
        ty::vstore_uniq => ty::vstore_uniq,
        ty::vstore_slice(r) => ty::vstore_slice(this.fold_region(r)),
    }
}

pub fn super_fold_trait_store<T:TypeFolder>(this: &mut T,
                                            trait_store: ty::TraitStore)
                                            -> ty::TraitStore {
    match trait_store {
        ty::UniqTraitStore      => ty::UniqTraitStore,
        ty::RegionTraitStore(r) => ty::RegionTraitStore(this.fold_region(r)),
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'a> {
    pub tcx: &'a ty::ctxt,
    pub fldop: |ty::t|: 'a -> ty::t,
}

impl<'a> TypeFolder for BottomUpFolder<'a> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt { self.tcx }

    fn fold_ty(&mut self, ty: ty::t) -> ty::t {
        let t1 = super_fold_ty(self, ty);
        (self.fldop)(t1)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

pub struct RegionFolder<'a> {
    tcx: &'a ty::ctxt,
    fld_t: |ty::t|: 'a -> ty::t,
    fld_r: |ty::Region|: 'a -> ty::Region,
}

impl<'a> RegionFolder<'a> {
    pub fn general(tcx: &'a ty::ctxt,
                   fld_r: |ty::Region|: 'a -> ty::Region,
                   fld_t: |ty::t|: 'a -> ty::t)
                   -> RegionFolder<'a> {
        RegionFolder {
            tcx: tcx,
            fld_t: fld_t,
            fld_r: fld_r
        }
    }

    pub fn regions(tcx: &'a ty::ctxt, fld_r: |ty::Region|: 'a -> ty::Region)
                   -> RegionFolder<'a> {
        fn noop(t: ty::t) -> ty::t { t }

        RegionFolder {
            tcx: tcx,
            fld_t: noop,
            fld_r: fld_r
        }
    }
}

impl<'a> TypeFolder for RegionFolder<'a> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt { self.tcx }

    fn fold_ty(&mut self, ty: ty::t) -> ty::t {
        debug!("RegionFolder.fold_ty({})", ty.repr(self.tcx()));
        let t1 = super_fold_ty(self, ty);
        (self.fld_t)(t1)
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        debug!("RegionFolder.fold_region({})", r.repr(self.tcx()));
        (self.fld_r)(r)
    }
}
