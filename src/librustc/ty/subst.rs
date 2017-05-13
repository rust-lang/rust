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

use hir::def_id::DefId;
use ty::{self, Slice, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};

use serialize::{self, Encodable, Encoder, Decodable, Decoder};
use syntax_pos::{Span, DUMMY_SP};
use rustc_data_structures::accumulate_vec::AccumulateVec;

use core::nonzero::NonZero;
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;

/// An entity in the Rust typesystem, which can be one of
/// several kinds (only types and lifetimes for now).
/// To reduce memory usage, a `Kind` is a interned pointer,
/// with the lowest 2 bits being reserved for a tag to
/// indicate the type (`Ty` or `Region`) it points to.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Kind<'tcx> {
    ptr: NonZero<usize>,
    marker: PhantomData<(Ty<'tcx>, &'tcx ty::Region)>
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const REGION_TAG: usize = 0b01;

impl<'tcx> From<Ty<'tcx>> for Kind<'tcx> {
    fn from(ty: Ty<'tcx>) -> Kind<'tcx> {
        // Ensure we can use the tag bits.
        assert_eq!(mem::align_of_val(ty) & TAG_MASK, 0);

        let ptr = ty as *const _ as usize;
        Kind {
            ptr: unsafe {
                NonZero::new(ptr | TYPE_TAG)
            },
            marker: PhantomData
        }
    }
}

impl<'tcx> From<&'tcx ty::Region> for Kind<'tcx> {
    fn from(r: &'tcx ty::Region) -> Kind<'tcx> {
        // Ensure we can use the tag bits.
        assert_eq!(mem::align_of_val(r) & TAG_MASK, 0);

        let ptr = r as *const _ as usize;
        Kind {
            ptr: unsafe {
                NonZero::new(ptr | REGION_TAG)
            },
            marker: PhantomData
        }
    }
}

impl<'tcx> Kind<'tcx> {
    #[inline]
    unsafe fn downcast<T>(self, tag: usize) -> Option<&'tcx T> {
        let ptr = *self.ptr;
        if ptr & TAG_MASK == tag {
            Some(&*((ptr & !TAG_MASK) as *const _))
        } else {
            None
        }
    }

    #[inline]
    pub fn as_type(self) -> Option<Ty<'tcx>> {
        unsafe {
            self.downcast(TYPE_TAG)
        }
    }

    #[inline]
    pub fn as_region(self) -> Option<&'tcx ty::Region> {
        unsafe {
            self.downcast(REGION_TAG)
        }
    }
}

impl<'tcx> fmt::Debug for Kind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(ty) = self.as_type() {
            write!(f, "{:?}", ty)
        } else if let Some(r) = self.as_region() {
            write!(f, "{:?}", r)
        } else {
            write!(f, "<unknwon @ {:p}>", *self.ptr as *const ())
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for Kind<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        if let Some(ty) = self.as_type() {
            Kind::from(ty.fold_with(folder))
        } else if let Some(r) = self.as_region() {
            Kind::from(r.fold_with(folder))
        } else {
            bug!()
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        if let Some(ty) = self.as_type() {
            ty.visit_with(visitor)
        } else if let Some(r) = self.as_region() {
            r.visit_with(visitor)
        } else {
            bug!()
        }
    }
}

impl<'tcx> Encodable for Kind<'tcx> {
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
        e.emit_enum("Kind", |e| {
            if let Some(ty) = self.as_type() {
                e.emit_enum_variant("Ty", TYPE_TAG, 1, |e| {
                    e.emit_enum_variant_arg(0, |e| ty.encode(e))
                })
            } else if let Some(r) = self.as_region() {
                e.emit_enum_variant("Region", REGION_TAG, 1, |e| {
                    e.emit_enum_variant_arg(0, |e| r.encode(e))
                })
            } else {
                bug!()
            }
        })
    }
}

impl<'tcx> Decodable for Kind<'tcx> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Kind<'tcx>, D::Error> {
        d.read_enum("Kind", |d| {
            d.read_enum_variant(&["Ty", "Region"], |d, tag| {
                match tag {
                    TYPE_TAG => Ty::decode(d).map(Kind::from),
                    REGION_TAG => <&ty::Region>::decode(d).map(Kind::from),
                    _ => Err(d.error("invalid Kind tag"))
                }
            })
        })
    }
}

/// A substitution mapping type/region parameters to new values.
pub type Substs<'tcx> = Slice<Kind<'tcx>>;

impl<'a, 'gcx, 'tcx> Substs<'tcx> {
    /// Creates a Substs that maps each generic parameter to itself.
    pub fn identity_for_item(tcx: TyCtxt<'a, 'gcx, 'tcx>, def_id: DefId)
                             -> &'tcx Substs<'tcx> {
        Substs::for_item(tcx, def_id, |def, _| {
            tcx.mk_region(ty::ReEarlyBound(def.to_early_bound_region_data()))
        }, |def, _| tcx.mk_param_from_def(def))
    }

    /// Creates a Substs for generic parameter definitions,
    /// by calling closures to obtain each region and type.
    /// The closures get to observe the Substs as they're
    /// being built, which can be used to correctly
    /// substitute defaults of type parameters.
    pub fn for_item<FR, FT>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                            def_id: DefId,
                            mut mk_region: FR,
                            mut mk_type: FT)
                            -> &'tcx Substs<'tcx>
    where FR: FnMut(&ty::RegionParameterDef, &[Kind<'tcx>]) -> &'tcx ty::Region,
          FT: FnMut(&ty::TypeParameterDef<'tcx>, &[Kind<'tcx>]) -> Ty<'tcx> {
        let defs = tcx.item_generics(def_id);
        let mut substs = Vec::with_capacity(defs.count());
        Substs::fill_item(&mut substs, tcx, defs, &mut mk_region, &mut mk_type);
        tcx.intern_substs(&substs)
    }

    pub fn extend_to<FR, FT>(&self,
                             tcx: TyCtxt<'a, 'gcx, 'tcx>,
                             def_id: DefId,
                             mut mk_region: FR,
                             mut mk_type: FT)
                             -> &'tcx Substs<'tcx>
    where FR: FnMut(&ty::RegionParameterDef, &[Kind<'tcx>]) -> &'tcx ty::Region,
          FT: FnMut(&ty::TypeParameterDef<'tcx>, &[Kind<'tcx>]) -> Ty<'tcx>
    {
        let defs = tcx.item_generics(def_id);
        let mut result = Vec::with_capacity(defs.count());
        result.extend(self[..].iter().cloned());
        Substs::fill_single(&mut result, defs, &mut mk_region, &mut mk_type);
        tcx.intern_substs(&result)
    }

    fn fill_item<FR, FT>(substs: &mut Vec<Kind<'tcx>>,
                         tcx: TyCtxt<'a, 'gcx, 'tcx>,
                         defs: &ty::Generics<'tcx>,
                         mk_region: &mut FR,
                         mk_type: &mut FT)
    where FR: FnMut(&ty::RegionParameterDef, &[Kind<'tcx>]) -> &'tcx ty::Region,
          FT: FnMut(&ty::TypeParameterDef<'tcx>, &[Kind<'tcx>]) -> Ty<'tcx> {

        if let Some(def_id) = defs.parent {
            let parent_defs = tcx.item_generics(def_id);
            Substs::fill_item(substs, tcx, parent_defs, mk_region, mk_type);
        }
        Substs::fill_single(substs, defs, mk_region, mk_type)
    }

    fn fill_single<FR, FT>(substs: &mut Vec<Kind<'tcx>>,
                           defs: &ty::Generics<'tcx>,
                           mk_region: &mut FR,
                           mk_type: &mut FT)
    where FR: FnMut(&ty::RegionParameterDef, &[Kind<'tcx>]) -> &'tcx ty::Region,
          FT: FnMut(&ty::TypeParameterDef<'tcx>, &[Kind<'tcx>]) -> Ty<'tcx> {
        // Handle Self first, before all regions.
        let mut types = defs.types.iter();
        if defs.parent.is_none() && defs.has_self {
            let def = types.next().unwrap();
            let ty = mk_type(def, substs);
            assert_eq!(def.index as usize, substs.len());
            substs.push(Kind::from(ty));
        }

        for def in &defs.regions {
            let region = mk_region(def, substs);
            assert_eq!(def.index as usize, substs.len());
            substs.push(Kind::from(region));
        }

        for def in types {
            let ty = mk_type(def, substs);
            assert_eq!(def.index as usize, substs.len());
            substs.push(Kind::from(ty));
        }
    }

    pub fn is_noop(&self) -> bool {
        self.is_empty()
    }

    #[inline]
    pub fn params(&self) -> &[Kind<'tcx>] {
        // FIXME (dikaiosune) this should be removed, and corresponding compilation errors fixed
        self
    }

    #[inline]
    pub fn types(&'a self) -> impl DoubleEndedIterator<Item=Ty<'tcx>> + 'a {
        self.iter().filter_map(|k| k.as_type())
    }

    #[inline]
    pub fn regions(&'a self) -> impl DoubleEndedIterator<Item=&'tcx ty::Region> + 'a {
        self.iter().filter_map(|k| k.as_region())
    }

    #[inline]
    pub fn type_at(&self, i: usize) -> Ty<'tcx> {
        self[i].as_type().unwrap_or_else(|| {
            bug!("expected type for param #{} in {:?}", i, self);
        })
    }

    #[inline]
    pub fn region_at(&self, i: usize) -> &'tcx ty::Region {
        self[i].as_region().unwrap_or_else(|| {
            bug!("expected region for param #{} in {:?}", i, self);
        })
    }

    #[inline]
    pub fn type_for_def(&self, ty_param_def: &ty::TypeParameterDef) -> Ty<'tcx> {
        self.type_at(ty_param_def.index as usize)
    }

    #[inline]
    pub fn region_for_def(&self, def: &ty::RegionParameterDef) -> &'tcx ty::Region {
        self.region_at(def.index as usize)
    }

    /// Transform from substitutions for a child of `source_ancestor`
    /// (e.g. a trait or impl) to substitutions for the same child
    /// in a different item, with `target_substs` as the base for
    /// the target impl/trait, with the source child-specific
    /// parameters (e.g. method parameters) on top of that base.
    pub fn rebase_onto(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                       source_ancestor: DefId,
                       target_substs: &Substs<'tcx>)
                       -> &'tcx Substs<'tcx> {
        let defs = tcx.item_generics(source_ancestor);
        tcx.mk_substs(target_substs.iter().chain(&self[defs.own_count()..]).cloned())
    }

    pub fn truncate_to(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, generics: &ty::Generics<'tcx>)
                       -> &'tcx Substs<'tcx> {
        tcx.mk_substs(self.iter().take(generics.count()).cloned())
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx Substs<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let params: AccumulateVec<[_; 8]> = self.iter().map(|k| k.fold_with(folder)).collect();

        // If folding doesn't change the substs, it's faster to avoid
        // calling `mk_substs` and instead reuse the existing substs.
        if params[..] == self[..] {
            self
        } else {
            folder.tcx().intern_substs(&params)
        }
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_substs(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> serialize::UseSpecializedDecodable for &'tcx Substs<'tcx> {}

///////////////////////////////////////////////////////////////////////////
// Public trait `Subst`
//
// Just call `foo.subst(tcx, substs)` to perform a substitution across
// `foo`. Or use `foo.subst_spanned(tcx, substs, Some(span))` when
// there is more information available (for better errors).

pub trait Subst<'tcx> : Sized {
    fn subst<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      substs: &[Kind<'tcx>]) -> Self {
        self.subst_spanned(tcx, substs, None)
    }

    fn subst_spanned<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                               substs: &[Kind<'tcx>],
                               span: Option<Span>)
                               -> Self;
}

impl<'tcx, T:TypeFoldable<'tcx>> Subst<'tcx> for T {
    fn subst_spanned<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                               substs: &[Kind<'tcx>],
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
    substs: &'a [Kind<'tcx>],

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

    fn fold_region(&mut self, r: &'tcx ty::Region) -> &'tcx ty::Region {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region substitution of the bound
        // regions that appear in a function signature is done using
        // the specialized routine `ty::replace_late_regions()`.
        match *r {
            ty::ReEarlyBound(data) => {
                let r = self.substs.get(data.index as usize)
                            .and_then(|k| k.as_region());
                match r {
                    Some(r) => {
                        self.shift_region_through_binders(r)
                    }
                    None => {
                        let span = self.span.unwrap_or(DUMMY_SP);
                        span_bug!(
                            span,
                            "Region parameter out of range \
                             when substituting in region {} (root type={:?}) \
                             (index={})",
                            data.name,
                            self.root_ty,
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
        let opt_ty = self.substs.get(p.idx as usize)
                         .and_then(|k| k.as_type());
        let ty = match opt_ty {
            Some(t) => t,
            None => {
                let span = self.span.unwrap_or(DUMMY_SP);
                span_bug!(
                    span,
                    "Type parameter `{:?}` ({:?}/{}) out of range \
                         when substituting (root type={:?}) substs={:?}",
                    p,
                    source_ty,
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

    fn shift_region_through_binders(&self, region: &'tcx ty::Region) -> &'tcx ty::Region {
        self.tcx().mk_region(ty::fold::shift_region(*region, self.region_binders_passed))
    }
}

// Helper methods that modify substitutions.

impl<'a, 'gcx, 'tcx> ty::TraitRef<'tcx> {
    pub fn from_method(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                       trait_id: DefId,
                       substs: &Substs<'tcx>)
                       -> ty::TraitRef<'tcx> {
        let defs = tcx.item_generics(trait_id);

        ty::TraitRef {
            def_id: trait_id,
            substs: tcx.intern_substs(&substs[..defs.own_count()])
        }
    }
}

impl<'a, 'gcx, 'tcx> ty::ExistentialTraitRef<'tcx> {
    pub fn erase_self_ty(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                         trait_ref: ty::TraitRef<'tcx>)
                         -> ty::ExistentialTraitRef<'tcx> {
        // Assert there is a Self.
        trait_ref.substs.type_at(0);

        ty::ExistentialTraitRef {
            def_id: trait_ref.def_id,
            substs: tcx.intern_substs(&trait_ref.substs[1..])
        }
    }
}

impl<'a, 'gcx, 'tcx> ty::PolyExistentialTraitRef<'tcx> {
    /// Object types don't have a self-type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self-type. A common choice is `mk_err()`
    /// or some skolemized type.
    pub fn with_self_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                        self_ty: Ty<'tcx>)
                        -> ty::PolyTraitRef<'tcx>  {
        // otherwise the escaping regions would be captured by the binder
        assert!(!self_ty.has_escaping_regions());

        self.map_bound(|trait_ref| {
            ty::TraitRef {
                def_id: trait_ref.def_id,
                substs: tcx.mk_substs(
                    iter::once(Kind::from(self_ty)).chain(trait_ref.substs.iter().cloned()))
            }
        })
    }
}
