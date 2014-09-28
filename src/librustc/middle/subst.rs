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

use std::fmt;
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
    fn iter_mut<'a>(&'a mut self) -> MutItems<'a, T>;
    fn get<'a>(&'a self, index: uint) -> Option<&'a T>;
    fn get_mut<'a>(&'a mut self, index: uint) -> Option<&'a mut T>;
}

impl<T> HomogeneousTuple3<T> for (T, T, T) {
    fn len(&self) -> uint {
        3
    }

    fn as_slice<'a>(&'a self) -> &'a [T] {
        unsafe {
            let ptr: *const T = mem::transmute(self);
            let slice = raw::Slice { data: ptr, len: 3 };
            mem::transmute(slice)
        }
    }

    fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        unsafe {
            let ptr: *const T = mem::transmute(self);
            let slice = raw::Slice { data: ptr, len: 3 };
            mem::transmute(slice)
        }
    }

    fn iter<'a>(&'a self) -> Items<'a, T> {
        let slice: &'a [T] = self.as_slice();
        slice.iter()
    }

    fn iter_mut<'a>(&'a mut self) -> MutItems<'a, T> {
        self.as_mut_slice().iter_mut()
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
#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub struct Substs {
    pub types: VecPerParamSpace<ty::t>,
    pub regions: RegionSubsts,
}

/**
 * Represents the values to use when substituting lifetime parameters.
 * If the value is `ErasedRegions`, then this subst is occurring during
 * trans, and all region parameters will be replaced with `ty::ReStatic`. */
#[deriving(Clone, PartialEq, Eq, Hash, Show)]
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

    pub fn erase_regions(self) -> Substs {
        let Substs { types: types, regions: _ } = self;
        Substs { types: types, regions: ErasedRegions }
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
    // This was originally represented as a tuple with one Vec<T> for
    // each variant of ParamSpace, and that remains the abstraction
    // that it provides to its clients.
    //
    // Here is how the representation corresponds to the abstraction
    // i.e. the "abstraction function" AF:
    //
    // AF(self) = (self.content.slice_to(self.type_limit),
    //             self.content.slice(self.type_limit, self.self_limit),
    //             self.content.slice_from(self.self_limit))
    type_limit: uint,
    self_limit: uint,
    content: Vec<T>,
}

impl<T:fmt::Show> fmt::Show for VecPerParamSpace<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "VecPerParamSpace {{"));
        for space in ParamSpace::all().iter() {
            try!(write!(fmt, "{}: {}, ", *space, self.get_slice(*space)));
        }
        try!(write!(fmt, "}}"));
        Ok(())
    }
}

impl<T> VecPerParamSpace<T> {
    fn limits(&self, space: ParamSpace) -> (uint, uint) {
        match space {
            TypeSpace => (0, self.type_limit),
            SelfSpace => (self.type_limit, self.self_limit),
            FnSpace => (self.self_limit, self.content.len()),
        }
    }

    pub fn empty() -> VecPerParamSpace<T> {
        VecPerParamSpace {
            type_limit: 0,
            self_limit: 0,
            content: Vec::new()
        }
    }

    pub fn params_from_type(types: Vec<T>) -> VecPerParamSpace<T> {
        VecPerParamSpace::empty().with_vec(TypeSpace, types)
    }

    /// `t` is the type space.
    /// `s` is the self space.
    /// `f` is the fn space.
    pub fn new(t: Vec<T>, s: Vec<T>, f: Vec<T>) -> VecPerParamSpace<T> {
        let type_limit = t.len();
        let self_limit = t.len() + s.len();
        let mut content = t;
        content.push_all_move(s);
        content.push_all_move(f);
        VecPerParamSpace {
            type_limit: type_limit,
            self_limit: self_limit,
            content: content,
        }
    }

    fn new_internal(content: Vec<T>, type_limit: uint, self_limit: uint)
                    -> VecPerParamSpace<T>
    {
        VecPerParamSpace {
            type_limit: type_limit,
            self_limit: self_limit,
            content: content,
        }
    }

    /// Appends `value` to the vector associated with `space`.
    ///
    /// Unlike the `push` method in `Vec`, this should not be assumed
    /// to be a cheap operation (even when amortized over many calls).
    pub fn push(&mut self, space: ParamSpace, value: T) {
        let (_, limit) = self.limits(space);
        match space {
            TypeSpace => { self.type_limit += 1; self.self_limit += 1; }
            SelfSpace => { self.self_limit += 1; }
            FnSpace   => {}
        }
        self.content.insert(limit, value);
    }

    pub fn pop(&mut self, space: ParamSpace) -> Option<T> {
        let (start, limit) = self.limits(space);
        if start == limit {
            None
        } else {
            match space {
                TypeSpace => { self.type_limit -= 1; self.self_limit -= 1; }
                SelfSpace => { self.self_limit -= 1; }
                FnSpace   => {}
            }
            self.content.remove(limit - 1)
        }
    }

    pub fn truncate(&mut self, space: ParamSpace, len: uint) {
        // FIXME (#15435): slow; O(n^2); could enhance vec to make it O(n).
        while self.len(space) > len {
            self.pop(space);
        }
    }

    pub fn replace(&mut self, space: ParamSpace, elems: Vec<T>) {
        // FIXME (#15435): slow; O(n^2); could enhance vec to make it O(n).
        self.truncate(space, 0);
        for t in elems.into_iter() {
            self.push(space, t);
        }
    }

    pub fn get_self<'a>(&'a self) -> Option<&'a T> {
        let v = self.get_slice(SelfSpace);
        assert!(v.len() <= 1);
        if v.len() == 0 { None } else { Some(&v[0]) }
    }

    pub fn len(&self, space: ParamSpace) -> uint {
        self.get_slice(space).len()
    }

    pub fn is_empty_in(&self, space: ParamSpace) -> bool {
        self.len(space) == 0
    }

    pub fn get_slice<'a>(&'a self, space: ParamSpace) -> &'a [T] {
        let (start, limit) = self.limits(space);
        self.content.slice(start, limit)
    }

    pub fn get_mut_slice<'a>(&'a mut self, space: ParamSpace) -> &'a mut [T] {
        let (start, limit) = self.limits(space);
        self.content.slice_mut(start, limit)
    }

    pub fn opt_get<'a>(&'a self,
                       space: ParamSpace,
                       index: uint)
                       -> Option<&'a T> {
        let v = self.get_slice(space);
        if index < v.len() { Some(&v[index]) } else { None }
    }

    pub fn get<'a>(&'a self, space: ParamSpace, index: uint) -> &'a T {
        &self.get_slice(space)[index]
    }

    pub fn iter<'a>(&'a self) -> Items<'a,T> {
        self.content.iter()
    }

    pub fn all_vecs(&self, pred: |&[T]| -> bool) -> bool {
        let spaces = [TypeSpace, SelfSpace, FnSpace];
        spaces.iter().all(|&space| { pred(self.get_slice(space)) })
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
        let result = self.iter().map(pred).collect();
        VecPerParamSpace::new_internal(result,
                                       self.type_limit,
                                       self.self_limit)
    }

    pub fn map_move<U>(self, pred: |T| -> U) -> VecPerParamSpace<U> {
        let (t, s, f) = self.split();
        VecPerParamSpace::new(t.into_iter().map(|p| pred(p)).collect(),
                              s.into_iter().map(|p| pred(p)).collect(),
                              f.into_iter().map(|p| pred(p)).collect())
    }

    pub fn split(self) -> (Vec<T>, Vec<T>, Vec<T>) {
        // FIXME (#15418): this does two traversals when in principle
        // one would suffice.  i.e. change to use `move_iter`.
        let VecPerParamSpace { type_limit, self_limit, content } = self;
        let mut i = 0;
        let (prefix, fn_vec) = content.partition(|_| {
            let on_left = i < self_limit;
            i += 1;
            on_left
        });

        let mut i = 0;
        let (type_vec, self_vec) = prefix.partition(|_| {
            let on_left = i < type_limit;
            i += 1;
            on_left
        });

        (type_vec, self_vec, fn_vec)
    }

    pub fn with_vec(mut self, space: ParamSpace, vec: Vec<T>)
                    -> VecPerParamSpace<T>
    {
        assert!(self.is_empty_in(space));
        self.replace(space, vec);
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

struct SubstFolder<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    substs: &'a Substs,

    // The location for which the substitution is performed, if available.
    span: Option<Span>,

    // The root type that is being substituted, if available.
    root_ty: Option<ty::t>,

    // Depth of type stack
    ty_stack_depth: uint,
}

impl<'a, 'tcx> TypeFolder<'tcx> for SubstFolder<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.tcx }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region substitution of the bound
        // regions that appear in a function signature is done using
        // the specialized routine
        // `middle::typeck::check::regionmanip::replace_late_regions_in_fn_sig()`.
        match r {
            ty::ReEarlyBound(_, space, i, region_name) => {
                match self.substs.regions {
                    ErasedRegions => ty::ReStatic,
                    NonerasedRegions(ref regions) =>
                        match regions.opt_get(space, i) {
                            Some(t) => *t,
                            None => {
                                let span = self.span.unwrap_or(DUMMY_SP);
                                self.tcx().sess.span_bug(
                                    span,
                                    format!("Type parameter out of range \
                                     when substituting in region {} (root type={}) \
                                     (space={}, index={})",
                                    region_name.as_str(),
                                    self.root_ty.repr(self.tcx()),
                                    space, i).as_slice());
                            }
                        }
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
                check(self,
                      p,
                      t,
                      self.substs.types.opt_get(p.space, p.idx),
                      p.space,
                      p.idx)
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
                 p: ty::ParamTy,
                 source_ty: ty::t,
                 opt_ty: Option<&ty::t>,
                 space: ParamSpace,
                 index: uint)
                 -> ty::t {
            match opt_ty {
                Some(t) => *t,
                None => {
                    let span = this.span.unwrap_or(DUMMY_SP);
                    this.tcx().sess.span_bug(
                        span,
                        format!("Type parameter `{}` ({}/{}/{}) out of range \
                                 when substituting (root type={})",
                                p.repr(this.tcx()),
                                source_ty.repr(this.tcx()),
                                space,
                                index,
                                this.root_ty.repr(this.tcx())).as_slice());
                }
            }
        }
    }
}
