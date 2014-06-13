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
use middle::ty_fold::{TypeFoldable, TypeFolder};
use util::ppaux::Repr;

use std::iter::Chain;
use std::mem;
use std::raw;
use std::slice::{Items, MutItems};
use std::vec::Vec;
use syntax::codemap::{Span, DUMMY_SP};

///////////////////////////////////////////////////////////////////////////
// HomogeneousTuple3 trait
//
// This could be moved into standard library at some point.

trait HomogeneousTuple3<T> {
    fn len(&self) -> uint;
    fn as_slice<'a>(&'a self) -> &'a [T];
    fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T];
    fn iter<'a>(&'a self) -> Items<'a, T>;
    fn mut_iter<'a>(&'a mut self) -> MutItems<'a, T>;
    fn get<'a>(&'a self, index: uint) -> Option<&'a T>;
    fn get_mut<'a>(&'a mut self, index: uint) -> Option<&'a mut T>;
}

impl<T> HomogeneousTuple3<T> for (T, T, T) {
    fn len(&self) -> uint {
        3
    }

    fn as_slice<'a>(&'a self) -> &'a [T] {
        unsafe {
            let ptr: *T = mem::transmute(self);
            let slice = raw::Slice { data: ptr, len: 3 };
            mem::transmute(slice)
        }
    }

    fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        unsafe {
            let ptr: *T = mem::transmute(self);
            let slice = raw::Slice { data: ptr, len: 3 };
            mem::transmute(slice)
        }
    }

    fn iter<'a>(&'a self) -> Items<'a, T> {
        let slice: &'a [T] = self.as_slice();
        slice.iter()
    }

    fn mut_iter<'a>(&'a mut self) -> MutItems<'a, T> {
        self.as_mut_slice().mut_iter()
    }

    fn get<'a>(&'a self, index: uint) -> Option<&'a T> {
        self.as_slice().get(index)
    }

    fn get_mut<'a>(&'a mut self, index: uint) -> Option<&'a mut T> {
        Some(&mut self.as_mut_slice()[index]) // wrong: fallible
    }
}

///////////////////////////////////////////////////////////////////////////

/**
 * A substitution mapping type/region parameters to new values. We
 * identify each in-scope parameter by an *index* and a *parameter
 * space* (which indices where the parameter is defined; see
 * `ParamSpace`).
 */
#[deriving(Clone, PartialEq, Eq, Hash)]
pub struct Substs {
    pub types: VecPerParamSpace<ty::t>,
    pub regions: RegionSubsts,
}

/**
 * Represents the values to use when substituting lifetime parameters.
 * If the value is `ErasedRegions`, then this subst is occurring during
 * trans, and all region parameters will be replaced with `ty::ReStatic`. */
#[deriving(Clone, PartialEq, Eq, Hash)]
pub enum RegionSubsts {
    ErasedRegions,
    NonerasedRegions(VecPerParamSpace<ty::Region>)
}

impl Substs {
    pub fn new(t: VecPerParamSpace<ty::t>,
               r: VecPerParamSpace<ty::Region>)
               -> Substs
    {
        Substs { types: t, regions: NonerasedRegions(r) }
    }

    pub fn new_type(t: Vec<ty::t>,
                    r: Vec<ty::Region>)
                    -> Substs
    {
        Substs::new(VecPerParamSpace::new(t, Vec::new(), Vec::new()),
                    VecPerParamSpace::new(r, Vec::new(), Vec::new()))
    }

    pub fn new_trait(t: Vec<ty::t>,
                     r: Vec<ty::Region>,
                     s: ty::t)
                    -> Substs
    {
        Substs::new(VecPerParamSpace::new(t, vec!(s), Vec::new()),
                    VecPerParamSpace::new(r, Vec::new(), Vec::new()))
    }

    pub fn erased(t: VecPerParamSpace<ty::t>) -> Substs
    {
        Substs { types: t, regions: ErasedRegions }
    }

    pub fn empty() -> Substs {
        Substs {
            types: VecPerParamSpace::empty(),
            regions: NonerasedRegions(VecPerParamSpace::empty()),
        }
    }

    pub fn trans_empty() -> Substs {
        Substs {
            types: VecPerParamSpace::empty(),
            regions: ErasedRegions
        }
    }

    pub fn is_noop(&self) -> bool {
        let regions_is_noop = match self.regions {
            ErasedRegions => false, // may be used to canonicalize
            NonerasedRegions(ref regions) => regions.is_empty(),
        };

        regions_is_noop && self.types.is_empty()
    }

    pub fn self_ty(&self) -> Option<ty::t> {
        self.types.get_self().map(|&t| t)
    }

    pub fn with_self_ty(&self, self_ty: ty::t) -> Substs {
        assert!(self.self_ty().is_none());
        let mut s = (*self).clone();
        s.types.push(SelfSpace, self_ty);
        s
    }

    pub fn regions<'a>(&'a self) -> &'a VecPerParamSpace<ty::Region> {
        /*!
         * Since ErasedRegions are only to be used in trans, most of
         * the compiler can use this method to easily access the set
         * of region substitutions.
         */

        match self.regions {
            ErasedRegions => fail!("Erased regions only expected in trans"),
            NonerasedRegions(ref r) => r
        }
    }

    pub fn mut_regions<'a>(&'a mut self) -> &'a mut VecPerParamSpace<ty::Region> {
        /*!
         * Since ErasedRegions are only to be used in trans, most of
         * the compiler can use this method to easily access the set
         * of region substitutions.
         */

        match self.regions {
            ErasedRegions => fail!("Erased regions only expected in trans"),
            NonerasedRegions(ref mut r) => r
        }
    }

    pub fn with_method_from(self, substs: &Substs) -> Substs {
        self.with_method((*substs.types.get_vec(FnSpace)).clone(),
                         (*substs.regions().get_vec(FnSpace)).clone())
    }

    pub fn with_method(self,
                       m_types: Vec<ty::t>,
                       m_regions: Vec<ty::Region>)
                       -> Substs
    {
        let Substs { types, regions } = self;
        let types = types.with_vec(FnSpace, m_types);
        let regions = regions.map(m_regions,
                                  |r, m_regions| r.with_vec(FnSpace, m_regions));
        Substs { types: types, regions: regions }
    }
}

impl RegionSubsts {
    fn map<A>(self,
              a: A,
              op: |VecPerParamSpace<ty::Region>, A| -> VecPerParamSpace<ty::Region>)
              -> RegionSubsts {
        match self {
            ErasedRegions => ErasedRegions,
            NonerasedRegions(r) => NonerasedRegions(op(r, a))
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// ParamSpace

#[deriving(PartialOrd, Ord, PartialEq, Eq,
           Clone, Hash, Encodable, Decodable, Show)]
pub enum ParamSpace {
    TypeSpace, // Type parameters attached to a type definition, trait, or impl
    SelfSpace, // Self parameter on a trait
    FnSpace,   // Type parameters attached to a method or fn
}

impl ParamSpace {
    pub fn all() -> [ParamSpace, ..3] {
        [TypeSpace, SelfSpace, FnSpace]
    }

    pub fn to_uint(self) -> uint {
        match self {
            TypeSpace => 0,
            SelfSpace => 1,
            FnSpace => 2,
        }
    }

    pub fn from_uint(u: uint) -> ParamSpace {
        match u {
            0 => TypeSpace,
            1 => SelfSpace,
            2 => FnSpace,
            _ => fail!("Invalid ParamSpace: {}", u)
        }
    }
}

/**
 * Vector of things sorted by param space. Used to keep
 * the set of things declared on the type, self, or method
 * distinct.
 */
#[deriving(PartialEq, Eq, Clone, Hash, Encodable, Decodable)]
pub struct VecPerParamSpace<T> {
    vecs: (Vec<T>, Vec<T>, Vec<T>)
}

impl<T> VecPerParamSpace<T> {
    pub fn empty() -> VecPerParamSpace<T> {
        VecPerParamSpace {
            vecs: (Vec::new(), Vec::new(), Vec::new())
        }
    }

    pub fn params_from_type(types: Vec<T>) -> VecPerParamSpace<T> {
        VecPerParamSpace::empty().with_vec(TypeSpace, types)
    }

    pub fn new(t: Vec<T>, s: Vec<T>, f: Vec<T>) -> VecPerParamSpace<T> {
        VecPerParamSpace {
            vecs: (t, s, f)
        }
    }

    pub fn sort(t: Vec<T>, space: |&T| -> ParamSpace) -> VecPerParamSpace<T> {
        let mut result = VecPerParamSpace::empty();
        for t in t.move_iter() {
            result.push(space(&t), t);
        }
        result
    }

    pub fn push(&mut self, space: ParamSpace, value: T) {
        self.get_mut_vec(space).push(value);
    }

    pub fn get_self<'a>(&'a self) -> Option<&'a T> {
        let v = self.get_vec(SelfSpace);
        assert!(v.len() <= 1);
        if v.len() == 0 { None } else { Some(v.get(0)) }
    }

    pub fn len(&self, space: ParamSpace) -> uint {
        self.get_vec(space).len()
    }

    pub fn get_vec<'a>(&'a self, space: ParamSpace) -> &'a Vec<T> {
        self.vecs.get(space as uint).unwrap()
    }

    pub fn get_mut_vec<'a>(&'a mut self, space: ParamSpace) -> &'a mut Vec<T> {
        self.vecs.get_mut(space as uint).unwrap()
    }

    pub fn opt_get<'a>(&'a self,
                       space: ParamSpace,
                       index: uint)
                       -> Option<&'a T> {
        let v = self.get_vec(space);
        if index < v.len() { Some(v.get(index)) } else { None }
    }

    pub fn get<'a>(&'a self, space: ParamSpace, index: uint) -> &'a T {
        self.get_vec(space).get(index)
    }

    pub fn get_mut<'a>(&'a mut self,
                       space: ParamSpace,
                       index: uint) -> &'a mut T {
        self.get_mut_vec(space).get_mut(index)
    }

    pub fn iter<'a>(&'a self) -> Chain<Items<'a,T>,
                                       Chain<Items<'a,T>,
                                             Items<'a,T>>> {
        let (ref r, ref s, ref f) = self.vecs;
        r.iter().chain(s.iter().chain(f.iter()))
    }

    pub fn all_vecs(&self, pred: |&Vec<T>| -> bool) -> bool {
        self.vecs.iter().all(pred)
    }

    pub fn all(&self, pred: |&T| -> bool) -> bool {
        self.iter().all(pred)
    }

    pub fn any(&self, pred: |&T| -> bool) -> bool {
        self.iter().any(pred)
    }

    pub fn is_empty(&self) -> bool {
        self.all_vecs(|v| v.is_empty())
    }

    pub fn map<U>(&self, pred: |&T| -> U) -> VecPerParamSpace<U> {
        VecPerParamSpace::new(self.vecs.ref0().iter().map(|p| pred(p)).collect(),
                              self.vecs.ref1().iter().map(|p| pred(p)).collect(),
                              self.vecs.ref2().iter().map(|p| pred(p)).collect())
    }

    pub fn map_rev<U>(&self, pred: |&T| -> U) -> VecPerParamSpace<U> {
        /*!
         * Executes the map but in reverse order. For hacky reasons, we rely
         * on this in table.
         *
         * FIXME(#5527) -- order of eval becomes irrelevant with newer
         * trait reform, which features an idempotent algorithm that
         * can be run to a fixed point
         */

        let mut fns: Vec<U> = self.vecs.ref2().iter().rev().map(|p| pred(p)).collect();

        // NB: Calling foo.rev().map().rev() causes the calls to map
        // to occur in the wrong order. This was somewhat surprising
        // to me, though it makes total sense.
        fns.reverse();

        let mut selfs: Vec<U> = self.vecs.ref1().iter().rev().map(|p| pred(p)).collect();
        selfs.reverse();
        let mut tys: Vec<U> = self.vecs.ref0().iter().rev().map(|p| pred(p)).collect();
        tys.reverse();
        VecPerParamSpace::new(tys, selfs, fns)
    }

    pub fn split(self) -> (Vec<T>, Vec<T>, Vec<T>) {
        self.vecs
    }

    pub fn with_vec(mut self, space: ParamSpace, vec: Vec<T>)
                    -> VecPerParamSpace<T>
    {
        assert!(self.get_vec(space).is_empty());
        *self.get_mut_vec(space) = vec;
        self
    }
}

///////////////////////////////////////////////////////////////////////////
// Public trait `Subst`
//
// Just call `foo.subst(tcx, substs)` to perform a substitution across
// `foo`. Or use `foo.subst_spanned(tcx, substs, Some(span))` when
// there is more information available (for better errors).

pub trait Subst {
    fn subst(&self, tcx: &ty::ctxt, substs: &Substs) -> Self {
        self.subst_spanned(tcx, substs, None)
    }

    fn subst_spanned(&self, tcx: &ty::ctxt,
                     substs: &Substs,
                     span: Option<Span>)
                     -> Self;
}

impl<T:TypeFoldable> Subst for T {
    fn subst_spanned(&self,
                     tcx: &ty::ctxt,
                     substs: &Substs,
                     span: Option<Span>)
                     -> T
    {
        let mut folder = SubstFolder { tcx: tcx,
                                       substs: substs,
                                       span: span,
                                       root_ty: None,
                                       ty_stack_depth: 0 };
        (*self).fold_with(&mut folder)
    }
}

///////////////////////////////////////////////////////////////////////////
// The actual substitution engine itself is a type folder.

struct SubstFolder<'a> {
    tcx: &'a ty::ctxt,
    substs: &'a Substs,

    // The location for which the substitution is performed, if available.
    span: Option<Span>,

    // The root type that is being substituted, if available.
    root_ty: Option<ty::t>,

    // Depth of type stack
    ty_stack_depth: uint,
}

impl<'a> TypeFolder for SubstFolder<'a> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt { self.tcx }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region substitution of the bound
        // regions that appear in a function signature is done using
        // the specialized routine
        // `middle::typeck::check::regionmanip::replace_late_regions_in_fn_sig()`.
        match r {
            ty::ReEarlyBound(_, space, i, _) => {
                match self.substs.regions {
                    ErasedRegions => ty::ReStatic,
                    NonerasedRegions(ref regions) => *regions.get(space, i),
                }
            }
            _ => r
        }
    }

    fn fold_ty(&mut self, t: ty::t) -> ty::t {
        if !ty::type_needs_subst(t) {
            return t;
        }

        // track the root type we were asked to substitute
        let depth = self.ty_stack_depth;
        if depth == 0 {
            self.root_ty = Some(t);
        }
        self.ty_stack_depth += 1;

        let t1 = match ty::get(t).sty {
            ty::ty_param(p) => {
                check(self, t, self.substs.types.opt_get(p.space, p.idx))
            }
            _ => {
                ty_fold::super_fold_ty(self, t)
            }
        };

        assert_eq!(depth + 1, self.ty_stack_depth);
        self.ty_stack_depth -= 1;
        if depth == 0 {
            self.root_ty = None;
        }

        return t1;

        fn check(this: &SubstFolder,
                 source_ty: ty::t,
                 opt_ty: Option<&ty::t>)
                 -> ty::t {
            match opt_ty {
                Some(t) => *t,
                None => {
                    let span = this.span.unwrap_or(DUMMY_SP);
                    this.tcx().sess.span_bug(
                        span,
                        format!("Type parameter {} out of range \
                                 when substituting (root type={})",
                                source_ty.repr(this.tcx()),
                                this.root_ty.repr(this.tcx())).as_slice());
                }
            }
        }
    }
}
