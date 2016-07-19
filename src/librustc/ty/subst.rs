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

pub use self::ParamSpace::*;

use middle::cstore;
use hir::def_id::DefId;
use ty::{self, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder};

use serialize::{Encodable, Encoder, Decodable, Decoder};
use std::fmt;
use std::iter::IntoIterator;
use std::slice::Iter;
use std::vec::{Vec, IntoIter};
use syntax_pos::{Span, DUMMY_SP};

///////////////////////////////////////////////////////////////////////////

/// A substitution mapping type/region parameters to new values. We
/// identify each in-scope parameter by an *index* and a *parameter
/// space* (which indices where the parameter is defined; see
/// `ParamSpace`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Substs<'tcx> {
    pub types: VecPerParamSpace<Ty<'tcx>>,
    pub regions: VecPerParamSpace<ty::Region>,
}

impl<'a, 'gcx, 'tcx> Substs<'tcx> {
    pub fn new(t: VecPerParamSpace<Ty<'tcx>>,
               r: VecPerParamSpace<ty::Region>)
               -> Substs<'tcx>
    {
        Substs { types: t, regions: r }
    }

    pub fn new_type(t: Vec<Ty<'tcx>>,
                    r: Vec<ty::Region>)
                    -> Substs<'tcx>
    {
        Substs::new(VecPerParamSpace::new(t, Vec::new(), Vec::new()),
                    VecPerParamSpace::new(r, Vec::new(), Vec::new()))
    }

    pub fn new_trait(t: Vec<Ty<'tcx>>,
                     r: Vec<ty::Region>,
                     s: Ty<'tcx>)
                    -> Substs<'tcx>
    {
        Substs::new(VecPerParamSpace::new(t, vec!(s), Vec::new()),
                    VecPerParamSpace::new(r, Vec::new(), Vec::new()))
    }

    pub fn empty() -> Substs<'tcx> {
        Substs {
            types: VecPerParamSpace::empty(),
            regions: VecPerParamSpace::empty(),
        }
    }

    pub fn is_noop(&self) -> bool {
        self.regions.is_empty() && self.types.is_empty()
    }

    pub fn type_for_def(&self, ty_param_def: &ty::TypeParameterDef) -> Ty<'tcx> {
        *self.types.get(ty_param_def.space, ty_param_def.index as usize)
    }

    pub fn self_ty(&self) -> Option<Ty<'tcx>> {
        self.types.get_self().cloned()
    }

    pub fn with_self_ty(&self, self_ty: Ty<'tcx>) -> Substs<'tcx> {
        assert!(self.self_ty().is_none());
        let mut s = (*self).clone();
        s.types.push(SelfSpace, self_ty);
        s
    }

    pub fn erase_regions(self) -> Substs<'tcx> {
        let Substs { types, regions } = self;
        let regions = regions.map(|_| ty::ReErased);
        Substs { types: types, regions: regions }
    }

    pub fn with_method(self,
                       m_types: Vec<Ty<'tcx>>,
                       m_regions: Vec<ty::Region>)
                       -> Substs<'tcx>
    {
        let Substs { types, regions } = self;
        let types = types.with_slice(FnSpace, &m_types);
        let regions = regions.with_slice(FnSpace, &m_regions);
        Substs { types: types, regions: regions }
    }

    pub fn with_method_from(&self,
                            meth_substs: &Substs<'tcx>)
                            -> Substs<'tcx>
    {
        let Substs { types, regions } = self.clone();
        let types = types.with_slice(FnSpace, meth_substs.types.get_slice(FnSpace));
        let regions = regions.with_slice(FnSpace, meth_substs.regions.get_slice(FnSpace));
        Substs { types: types, regions: regions }
    }

    pub fn with_method_from_subst(&self, other: &Substs<'tcx>) -> Substs<'tcx> {
        let Substs { types, regions } = self.clone();
        let types = types.with_slice(FnSpace, other.types.get_slice(FnSpace));
        let regions = regions.with_slice(FnSpace, other.regions.get_slice(FnSpace));
        Substs { types: types, regions: regions }
    }

    /// Creates a trait-ref out of this substs, ignoring the FnSpace substs
    pub fn to_trait_ref(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, trait_id: DefId)
                        -> ty::TraitRef<'tcx> {
        let Substs { mut types, mut regions } = self.clone();
        types.truncate(FnSpace, 0);
        regions.truncate(FnSpace, 0);

        ty::TraitRef {
            def_id: trait_id,
            substs: tcx.mk_substs(Substs { types: types, regions: regions })
        }
    }
}

impl<'tcx> Encodable for Substs<'tcx> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        cstore::tls::with_encoding_context(s, |ecx, rbml_w| {
            ecx.encode_substs(rbml_w, self);
            Ok(())
        })
    }
}

impl<'tcx> Decodable for Substs<'tcx> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Substs<'tcx>, D::Error> {
        cstore::tls::with_decoding_context(d, |dcx, rbml_r| {
            Ok(dcx.decode_substs(rbml_r))
        })
    }
}

impl<'tcx> Decodable for &'tcx Substs<'tcx> {
    fn decode<D: Decoder>(d: &mut D) -> Result<&'tcx Substs<'tcx>, D::Error> {
        let substs = cstore::tls::with_decoding_context(d, |dcx, rbml_r| {
            let substs = dcx.decode_substs(rbml_r);
            dcx.tcx().mk_substs(substs)
        });

        Ok(substs)
    }
}

///////////////////////////////////////////////////////////////////////////
// ParamSpace

#[derive(PartialOrd, Ord, PartialEq, Eq, Copy,
           Clone, Hash, RustcEncodable, RustcDecodable, Debug)]
pub enum ParamSpace {
    TypeSpace,  // Type parameters attached to a type definition, trait, or impl
    SelfSpace,  // Self parameter on a trait
    FnSpace,    // Type parameters attached to a method or fn
}

impl ParamSpace {
    pub fn all() -> [ParamSpace; 3] {
        [TypeSpace, SelfSpace, FnSpace]
    }

    pub fn to_uint(self) -> usize {
        match self {
            TypeSpace => 0,
            SelfSpace => 1,
            FnSpace => 2,
        }
    }

    pub fn from_uint(u: usize) -> ParamSpace {
        match u {
            0 => TypeSpace,
            1 => SelfSpace,
            2 => FnSpace,
            _ => bug!("Invalid ParamSpace: {}", u)
        }
    }
}

/// Vector of things sorted by param space. Used to keep
/// the set of things declared on the type, self, or method
/// distinct.
#[derive(PartialEq, Eq, Clone, Hash, RustcEncodable, RustcDecodable)]
pub struct VecPerParamSpace<T> {
    // This was originally represented as a tuple with one Vec<T> for
    // each variant of ParamSpace, and that remains the abstraction
    // that it provides to its clients.
    //
    // Here is how the representation corresponds to the abstraction
    // i.e. the "abstraction function" AF:
    //
    // AF(self) = (self.content[..self.type_limit],
    //             self.content[self.type_limit..self.self_limit],
    //             self.content[self.self_limit..])
    type_limit: usize,
    self_limit: usize,
    content: Vec<T>,
}

/// The `split` function converts one `VecPerParamSpace` into this
/// `SeparateVecsPerParamSpace` structure.
pub struct SeparateVecsPerParamSpace<T> {
    pub types: Vec<T>,
    pub selfs: Vec<T>,
    pub fns: Vec<T>,
}

impl<T: fmt::Debug> fmt::Debug for VecPerParamSpace<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:?};{:?};{:?}]",
               self.get_slice(TypeSpace),
               self.get_slice(SelfSpace),
               self.get_slice(FnSpace))
    }
}

impl<T> VecPerParamSpace<T> {
    fn limits(&self, space: ParamSpace) -> (usize, usize) {
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

    /// `t` is the type space.
    /// `s` is the self space.
    /// `f` is the fn space.
    pub fn new(t: Vec<T>, s: Vec<T>, f: Vec<T>) -> VecPerParamSpace<T> {
        let type_limit = t.len();
        let self_limit = type_limit + s.len();

        let mut content = t;
        content.extend(s);
        content.extend(f);

        VecPerParamSpace {
            type_limit: type_limit,
            self_limit: self_limit,
            content: content,
        }
    }

    fn new_internal(content: Vec<T>, type_limit: usize, self_limit: usize)
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
            FnSpace => { }
        }
        self.content.insert(limit, value);
    }

    /// Appends `values` to the vector associated with `space`.
    ///
    /// Unlike the `extend` method in `Vec`, this should not be assumed
    /// to be a cheap operation (even when amortized over many calls).
    pub fn extend<I:Iterator<Item=T>>(&mut self, space: ParamSpace, values: I) {
        // This could be made more efficient, obviously.
        for item in values {
            self.push(space, item);
        }
    }

    pub fn pop(&mut self, space: ParamSpace) -> Option<T> {
        let (start, limit) = self.limits(space);
        if start == limit {
            None
        } else {
            match space {
                TypeSpace => { self.type_limit -= 1; self.self_limit -= 1; }
                SelfSpace => { self.self_limit -= 1; }
                FnSpace => {}
            }
            if self.content.is_empty() {
                None
            } else {
                Some(self.content.remove(limit - 1))
            }
        }
    }

    pub fn truncate(&mut self, space: ParamSpace, len: usize) {
        // FIXME (#15435): slow; O(n^2); could enhance vec to make it O(n).
        while self.len(space) > len {
            self.pop(space);
        }
    }

    pub fn replace(&mut self, space: ParamSpace, elems: Vec<T>) {
        // FIXME (#15435): slow; O(n^2); could enhance vec to make it O(n).
        self.truncate(space, 0);
        for t in elems {
            self.push(space, t);
        }
    }

    pub fn get_self<'a>(&'a self) -> Option<&'a T> {
        let v = self.get_slice(SelfSpace);
        assert!(v.len() <= 1);
        if v.is_empty() { None } else { Some(&v[0]) }
    }

    pub fn len(&self, space: ParamSpace) -> usize {
        self.get_slice(space).len()
    }

    pub fn is_empty_in(&self, space: ParamSpace) -> bool {
        self.len(space) == 0
    }

    pub fn get_slice<'a>(&'a self, space: ParamSpace) -> &'a [T] {
        let (start, limit) = self.limits(space);
        &self.content[start.. limit]
    }

    pub fn get_mut_slice<'a>(&'a mut self, space: ParamSpace) -> &'a mut [T] {
        let (start, limit) = self.limits(space);
        &mut self.content[start.. limit]
    }

    pub fn opt_get<'a>(&'a self,
                       space: ParamSpace,
                       index: usize)
                       -> Option<&'a T> {
        let v = self.get_slice(space);
        if index < v.len() { Some(&v[index]) } else { None }
    }

    pub fn get<'a>(&'a self, space: ParamSpace, index: usize) -> &'a T {
        &self.get_slice(space)[index]
    }

    pub fn iter<'a>(&'a self) -> Iter<'a,T> {
        self.content.iter()
    }

    pub fn into_iter(self) -> IntoIter<T> {
        self.content.into_iter()
    }

    pub fn iter_enumerated<'a>(&'a self) -> EnumeratedItems<'a,T> {
        EnumeratedItems::new(self)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.content
    }

    pub fn into_vec(self) -> Vec<T> {
        self.content
    }

    pub fn all_vecs<P>(&self, mut pred: P) -> bool where
        P: FnMut(&[T]) -> bool,
    {
        let spaces = [TypeSpace, SelfSpace, FnSpace];
        spaces.iter().all(|&space| { pred(self.get_slice(space)) })
    }

    pub fn all<P>(&self, pred: P) -> bool where P: FnMut(&T) -> bool {
        self.iter().all(pred)
    }

    pub fn any<P>(&self, pred: P) -> bool where P: FnMut(&T) -> bool {
        self.iter().any(pred)
    }

    pub fn is_empty(&self) -> bool {
        self.all_vecs(|v| v.is_empty())
    }

    pub fn map<U, P>(&self, pred: P) -> VecPerParamSpace<U> where P: FnMut(&T) -> U {
        let result = self.iter().map(pred).collect();
        VecPerParamSpace::new_internal(result,
                                       self.type_limit,
                                       self.self_limit)
    }

    pub fn map_enumerated<U, P>(&self, pred: P) -> VecPerParamSpace<U> where
        P: FnMut((ParamSpace, usize, &T)) -> U,
    {
        let result = self.iter_enumerated().map(pred).collect();
        VecPerParamSpace::new_internal(result,
                                       self.type_limit,
                                       self.self_limit)
    }

    pub fn split(self) -> SeparateVecsPerParamSpace<T> {
        let VecPerParamSpace { type_limit, self_limit, content } = self;

        let mut content_iter = content.into_iter();

        SeparateVecsPerParamSpace {
            types: content_iter.by_ref().take(type_limit).collect(),
            selfs: content_iter.by_ref().take(self_limit - type_limit).collect(),
            fns: content_iter.collect()
        }
    }

    pub fn with_slice(mut self, space: ParamSpace, slice: &[T])
                    -> VecPerParamSpace<T>
        where T: Clone
    {
        assert!(self.is_empty_in(space));
        for t in slice {
            self.push(space, t.clone());
        }

        self
    }
}

#[derive(Clone)]
pub struct EnumeratedItems<'a,T:'a> {
    vec: &'a VecPerParamSpace<T>,
    space_index: usize,
    elem_index: usize
}

impl<'a,T> EnumeratedItems<'a,T> {
    fn new(v: &'a VecPerParamSpace<T>) -> EnumeratedItems<'a,T> {
        let mut result = EnumeratedItems { vec: v, space_index: 0, elem_index: 0 };
        result.adjust_space();
        result
    }

    fn adjust_space(&mut self) {
        let spaces = ParamSpace::all();
        while
            self.space_index < spaces.len() &&
            self.elem_index >= self.vec.len(spaces[self.space_index])
        {
            self.space_index += 1;
            self.elem_index = 0;
        }
    }
}

impl<'a,T> Iterator for EnumeratedItems<'a,T> {
    type Item = (ParamSpace, usize, &'a T);

    fn next(&mut self) -> Option<(ParamSpace, usize, &'a T)> {
        let spaces = ParamSpace::all();
        if self.space_index < spaces.len() {
            let space = spaces[self.space_index];
            let index = self.elem_index;
            let item = self.vec.get(space, index);

            self.elem_index += 1;
            self.adjust_space();

            Some((space, index, item))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.vec.as_slice().len();
        (size, Some(size))
    }
}

impl<T> IntoIterator for VecPerParamSpace<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.into_vec().into_iter()
    }
}

impl<'a,T> IntoIterator for &'a VecPerParamSpace<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.as_slice().into_iter()
    }
}


///////////////////////////////////////////////////////////////////////////
// Public trait `Subst`
//
// Just call `foo.subst(tcx, substs)` to perform a substitution across
// `foo`. Or use `foo.subst_spanned(tcx, substs, Some(span))` when
// there is more information available (for better errors).

pub trait Subst<'tcx> : Sized {
    fn subst<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      substs: &Substs<'tcx>) -> Self {
        self.subst_spanned(tcx, substs, None)
    }

    fn subst_spanned<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                               substs: &Substs<'tcx>,
                               span: Option<Span>)
                               -> Self;
}

impl<'tcx, T:TypeFoldable<'tcx>> Subst<'tcx> for T {
    fn subst_spanned<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                               substs: &Substs<'tcx>,
                               span: Option<Span>)
                               -> T
    {
        let mut folder = SubstFolder { tcx: tcx,
                                       substs: substs,
                                       span: span,
                                       root_ty: None,
                                       ty_stack_depth: 0,
                                       region_binders_passed: 0 };
        (*self).fold_with(&mut folder)
    }
}

///////////////////////////////////////////////////////////////////////////
// The actual substitution engine itself is a type folder.

struct SubstFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    substs: &'a Substs<'tcx>,

    // The location for which the substitution is performed, if available.
    span: Option<Span>,

    // The root type that is being substituted, if available.
    root_ty: Option<Ty<'tcx>>,

    // Depth of type stack
    ty_stack_depth: usize,

    // Number of region binders we have passed through while doing the substitution
    region_binders_passed: u32,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for SubstFolder<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T> {
        self.region_binders_passed += 1;
        let t = t.super_fold_with(self);
        self.region_binders_passed -= 1;
        t
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region substitution of the bound
        // regions that appear in a function signature is done using
        // the specialized routine `ty::replace_late_regions()`.
        match r {
            ty::ReEarlyBound(data) => {
                match self.substs.regions.opt_get(data.space, data.index as usize) {
                    Some(&r) => {
                        self.shift_region_through_binders(r)
                    }
                    None => {
                        let span = self.span.unwrap_or(DUMMY_SP);
                        span_bug!(
                            span,
                            "Region parameter out of range \
                             when substituting in region {} (root type={:?}) \
                             (space={:?}, index={})",
                            data.name,
                            self.root_ty,
                            data.space,
                            data.index);
                    }
                }
            }
            _ => r
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_subst() {
            return t;
        }

        // track the root type we were asked to substitute
        let depth = self.ty_stack_depth;
        if depth == 0 {
            self.root_ty = Some(t);
        }
        self.ty_stack_depth += 1;

        let t1 = match t.sty {
            ty::TyParam(p) => {
                self.ty_for_param(p, t)
            }
            _ => {
                t.super_fold_with(self)
            }
        };

        assert_eq!(depth + 1, self.ty_stack_depth);
        self.ty_stack_depth -= 1;
        if depth == 0 {
            self.root_ty = None;
        }

        return t1;
    }
}

impl<'a, 'gcx, 'tcx> SubstFolder<'a, 'gcx, 'tcx> {
    fn ty_for_param(&self, p: ty::ParamTy, source_ty: Ty<'tcx>) -> Ty<'tcx> {
        // Look up the type in the substitutions. It really should be in there.
        let opt_ty = self.substs.types.opt_get(p.space, p.idx as usize);
        let ty = match opt_ty {
            Some(t) => *t,
            None => {
                let span = self.span.unwrap_or(DUMMY_SP);
                span_bug!(
                    span,
                    "Type parameter `{:?}` ({:?}/{:?}/{}) out of range \
                         when substituting (root type={:?}) substs={:?}",
                    p,
                    source_ty,
                    p.space,
                    p.idx,
                    self.root_ty,
                    self.substs);
            }
        };

        self.shift_regions_through_binders(ty)
    }

    /// It is sometimes necessary to adjust the debruijn indices during substitution. This occurs
    /// when we are substituting a type with escaping regions into a context where we have passed
    /// through region binders. That's quite a mouthful. Let's see an example:
    ///
    /// ```
    /// type Func<A> = fn(A);
    /// type MetaFunc = for<'a> fn(Func<&'a int>)
    /// ```
    ///
    /// The type `MetaFunc`, when fully expanded, will be
    ///
    ///     for<'a> fn(fn(&'a int))
    ///             ^~ ^~ ^~~
    ///             |  |  |
    ///             |  |  DebruijnIndex of 2
    ///             Binders
    ///
    /// Here the `'a` lifetime is bound in the outer function, but appears as an argument of the
    /// inner one. Therefore, that appearance will have a DebruijnIndex of 2, because we must skip
    /// over the inner binder (remember that we count Debruijn indices from 1). However, in the
    /// definition of `MetaFunc`, the binder is not visible, so the type `&'a int` will have a
    /// debruijn index of 1. It's only during the substitution that we can see we must increase the
    /// depth by 1 to account for the binder that we passed through.
    ///
    /// As a second example, consider this twist:
    ///
    /// ```
    /// type FuncTuple<A> = (A,fn(A));
    /// type MetaFuncTuple = for<'a> fn(FuncTuple<&'a int>)
    /// ```
    ///
    /// Here the final type will be:
    ///
    ///     for<'a> fn((&'a int, fn(&'a int)))
    ///                 ^~~         ^~~
    ///                 |           |
    ///          DebruijnIndex of 1 |
    ///                      DebruijnIndex of 2
    ///
    /// As indicated in the diagram, here the same type `&'a int` is substituted once, but in the
    /// first case we do not increase the Debruijn index and in the second case we do. The reason
    /// is that only in the second case have we passed through a fn binder.
    fn shift_regions_through_binders(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        debug!("shift_regions(ty={:?}, region_binders_passed={:?}, has_escaping_regions={:?})",
               ty, self.region_binders_passed, ty.has_escaping_regions());

        if self.region_binders_passed == 0 || !ty.has_escaping_regions() {
            return ty;
        }

        let result = ty::fold::shift_regions(self.tcx(), self.region_binders_passed, &ty);
        debug!("shift_regions: shifted result = {:?}", result);

        result
    }

    fn shift_region_through_binders(&self, region: ty::Region) -> ty::Region {
        ty::fold::shift_region(region, self.region_binders_passed)
    }
}
