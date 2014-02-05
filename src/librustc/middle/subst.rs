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
use util::ppaux::Repr;

use std::rc::Rc;
use syntax::codemap::Span;
use syntax::opt_vec::OptVec;

///////////////////////////////////////////////////////////////////////////
// Public trait `Subst`
//
// Just call `foo.subst(tcx, substs)` to perform a substitution across
// `foo`.
// Or use `foo.subst_spanned(tcx, substs, Some(span))` when there is more
// information available (for better errors).

pub trait Subst {
    fn subst(&self, tcx: ty::ctxt, substs: &ty::substs) -> Self {
        self.subst_spanned(tcx, substs, None)
    }
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> Self;
}

///////////////////////////////////////////////////////////////////////////
// Substitution over types
//
// Because this is so common, we make a special optimization to avoid
// doing anything if `substs` is a no-op.  I tried to generalize these
// to all subst methods but ran into trouble due to the limitations of
// our current method/trait matching algorithm. - Niko

impl Subst for ty::t {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::t {
        if ty::substs_is_noop(substs) && !ty::type_has_params(*self) {
            *self
        } else {
            let mut folder = SubstFolder {
                tcx: tcx,
                substs: substs,
                span: span,
                root_ty: Some(*self)
            };
            folder.fold_ty(*self)
        }
    }
}

struct SubstFolder<'a> {
    tcx: ty::ctxt,
    substs: &'a ty::substs,

    // The location for which the substitution is performed, if available.
    span: Option<Span>,

    // The root type that is being substituted, if available.
    root_ty: Option<ty::t>
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
                if p.idx < self.substs.tps.len() {
                    self.substs.tps[p.idx]
                } else {
                    let root_msg = match self.root_ty {
                        Some(root) => format!(" in the substitution of `{}`",
                                              root.repr(self.tcx)),
                        None => ~""
                    };
                    let m = format!("missing type param `{}`{}",
                                    t.repr(self.tcx), root_msg);
                    match self.span {
                        Some(span) => self.tcx.sess.span_err(span, m),
                        None => self.tcx.sess.err(m)
                    }
                    ty::mk_err()
                }
            }
            ty::ty_self(_) => {
                match self.substs.self_ty {
                    Some(ty) => ty,
                    None => {
                        let root_msg = match self.root_ty {
                            Some(root) => format!(" in the substitution of `{}`",
                                                  root.repr(self.tcx)),
                            None => ~""
                        };
                        let m = format!("missing `Self` type param{}", root_msg);
                        match self.span {
                            Some(span) => self.tcx.sess.span_err(span, m),
                            None => self.tcx.sess.err(m)
                        }
                        ty::mk_err()
                    }
                }
            }
            _ => ty_fold::super_fold_ty(self, t)
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Other types

impl<T:Subst> Subst for ~[T] {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ~[T] {
        self.map(|t| t.subst_spanned(tcx, substs, span))
    }
}
impl<T:Subst> Subst for Rc<T> {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> Rc<T> {
        Rc::new(self.borrow().subst_spanned(tcx, substs, span))
    }
}

impl<T:Subst> Subst for OptVec<T> {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> OptVec<T> {
        self.map(|t| t.subst_spanned(tcx, substs, span))
    }
}

impl<T:Subst + 'static> Subst for @T {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> @T {
        match self {
            t => @(**t).subst_spanned(tcx, substs, span)
        }
    }
}

impl<T:Subst> Subst for Option<T> {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> Option<T> {
        self.as_ref().map(|t| t.subst_spanned(tcx, substs, span))
    }
}

impl Subst for ty::TraitRef {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::TraitRef {
        ty::TraitRef {
            def_id: self.def_id,
            substs: self.substs.subst_spanned(tcx, substs, span)
        }
    }
}

impl Subst for ty::substs {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::substs {
        ty::substs {
            regions: self.regions.subst_spanned(tcx, substs, span),
            self_ty: self.self_ty.map(|typ| typ.subst_spanned(tcx, substs, span)),
            tps: self.tps.map(|typ| typ.subst_spanned(tcx, substs, span))
        }
    }
}

impl Subst for ty::RegionSubsts {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::RegionSubsts {
        match *self {
            ty::ErasedRegions => {
                ty::ErasedRegions
            }
            ty::NonerasedRegions(ref regions) => {
                ty::NonerasedRegions(regions.subst_spanned(tcx, substs, span))
            }
        }
    }
}

impl Subst for ty::BareFnTy {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::BareFnTy {
        let mut folder = SubstFolder {
            tcx: tcx,
            substs: substs,
            span: span,
            root_ty: None
        };
        folder.fold_bare_fn_ty(self)
    }
}

impl Subst for ty::ParamBounds {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::ParamBounds {
        ty::ParamBounds {
            builtin_bounds: self.builtin_bounds,
            trait_bounds: self.trait_bounds.subst_spanned(tcx, substs, span)
        }
    }
}

impl Subst for ty::TypeParameterDef {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::TypeParameterDef {
        ty::TypeParameterDef {
            ident: self.ident,
            def_id: self.def_id,
            bounds: self.bounds.subst_spanned(tcx, substs, span),
            default: self.default.map(|x| x.subst_spanned(tcx, substs, span))
        }
    }
}

impl Subst for ty::Generics {
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::Generics {
        ty::Generics {
            type_param_defs: self.type_param_defs.subst_spanned(tcx, substs, span),
            region_param_defs: self.region_param_defs.subst_spanned(tcx, substs, span),
        }
    }
}

impl Subst for ty::RegionParameterDef {
    fn subst_spanned(&self, _: ty::ctxt,
                     _: &ty::substs,
                     _: Option<Span>) -> ty::RegionParameterDef {
        *self
    }
}

impl Subst for ty::Region {
    fn subst_spanned(&self, _tcx: ty::ctxt,
                     substs: &ty::substs,
                     _: Option<Span>) -> ty::Region {
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
    fn subst_spanned(&self, tcx: ty::ctxt,
                     substs: &ty::substs,
                     span: Option<Span>) -> ty::ty_param_bounds_and_ty {
        ty::ty_param_bounds_and_ty {
            generics: self.generics.subst_spanned(tcx, substs, span),
            ty: self.ty.subst_spanned(tcx, substs, span)
        }
    }
}
