// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Type substitutions.

use middle::ty;
use middle::ty_fold;
use middle::ty_fold::TypeFolder;

use std::rc::Rc;
use syntax::opt_vec::OptVec;

///////////////////////////////////////////////////////////////////////////
// Public trait `Subst`
//
// Just call `foo.subst(tcx, substs)` to perform a substitution across
// `foo`.

pub trait Subst {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> Self;
}

///////////////////////////////////////////////////////////////////////////
// Substitution over types
//
// Because this is so common, we make a special optimization to avoid
// doing anything if `substs` is a no-op.  I tried to generalize these
// to all subst methods but ran into trouble due to the limitations of
// our current method/trait matching algorithm. - Niko

impl Subst for ty::t {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::t {
        if ty::substs_is_noop(substs) {
            *self
        } else {
            let mut folder = SubstFolder {tcx: tcx, substs: substs};
            folder.fold_ty(*self)
        }
    }
}

struct SubstFolder<'a> {
    tcx: ty::ctxt,
    substs: &'a ty::substs
}

impl<'a> TypeFolder for SubstFolder<'a> {
    fn tcx(&self) -> ty::ctxt { self.tcx }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        r.subst(self.tcx, self.substs)
    }

    fn fold_ty(&mut self, t: ty::t) -> ty::t {
        if !ty::type_needs_subst(t) {
            return t;
        }

        match ty::get(t).sty {
            ty::ty_param(p) => {
                self.substs.tps[p.idx]
            }
            ty::ty_self(_) => {
                self.substs.self_ty.expect("ty_self not found in substs")
            }
            _ => {
                ty_fold::super_fold_ty(self, t)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Other types

impl<T:Subst> Subst for ~[T] {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ~[T] {
        self.map(|t| t.subst(tcx, substs))
    }
}
impl<T:Subst> Subst for Rc<T> {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> Rc<T> {
        Rc::new(self.borrow().subst(tcx, substs))
    }
}

impl<T:Subst> Subst for OptVec<T> {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> OptVec<T> {
        self.map(|t| t.subst(tcx, substs))
    }
}

impl<T:Subst + 'static> Subst for @T {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> @T {
        match self {
            t => @(**t).subst(tcx, substs)
        }
    }
}

impl<T:Subst> Subst for Option<T> {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> Option<T> {
        self.as_ref().map(|t| t.subst(tcx, substs))
    }
}

impl Subst for ty::TraitRef {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::TraitRef {
        ty::TraitRef {
            def_id: self.def_id,
            substs: self.substs.subst(tcx, substs)
        }
    }
}

impl Subst for ty::substs {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::substs {
        ty::substs {
            regions: self.regions.subst(tcx, substs),
            self_ty: self.self_ty.map(|typ| typ.subst(tcx, substs)),
            tps: self.tps.map(|typ| typ.subst(tcx, substs))
        }
    }
}

impl Subst for ty::RegionSubsts {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::RegionSubsts {
        match *self {
            ty::ErasedRegions => {
                ty::ErasedRegions
            }
            ty::NonerasedRegions(ref regions) => {
                ty::NonerasedRegions(regions.subst(tcx, substs))
            }
        }
    }
}

impl Subst for ty::BareFnTy {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::BareFnTy {
        let mut folder = SubstFolder {tcx: tcx, substs: substs};
        folder.fold_bare_fn_ty(self)
    }
}

impl Subst for ty::ParamBounds {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::ParamBounds {
        ty::ParamBounds {
            builtin_bounds: self.builtin_bounds,
            trait_bounds: self.trait_bounds.subst(tcx, substs)
        }
    }
}

impl Subst for ty::TypeParameterDef {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::TypeParameterDef {
        ty::TypeParameterDef {
            ident: self.ident,
            def_id: self.def_id,
            bounds: self.bounds.subst(tcx, substs),
            default: self.default.map(|x| x.subst(tcx, substs))
        }
    }
}

impl Subst for ty::Generics {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::Generics {
        ty::Generics {
            type_param_defs: self.type_param_defs.subst(tcx, substs),
            region_param_defs: self.region_param_defs.subst(tcx, substs),
        }
    }
}

impl Subst for ty::RegionParameterDef {
    fn subst(&self, _: ty::ctxt, _: &ty::substs) -> ty::RegionParameterDef {
        *self
    }
}

impl Subst for ty::Region {
    fn subst(&self, _tcx: ty::ctxt, substs: &ty::substs) -> ty::Region {
        // Note: This routine only handles regions that are bound on
        // type declarationss and other outer declarations, not those
        // bound in *fn types*. Region substitution of the bound
        // regions that appear in a function signature is done using
        // the specialized routine
        // `middle::typeck::check::regionmanip::replace_bound_regions_in_fn_sig()`.
        match self {
            &ty::ReEarlyBound(_, i, _) => {
                match substs.regions {
                    ty::ErasedRegions => ty::ReStatic,
                    ty::NonerasedRegions(ref regions) => *regions.get(i),
                }
            }
            _ => *self
        }
    }
}

impl Subst for ty::ty_param_bounds_and_ty {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> ty::ty_param_bounds_and_ty {
        ty::ty_param_bounds_and_ty {
            generics: self.generics.subst(tcx, substs),
            ty: self.ty.subst(tcx, substs)
        }
    }
}
