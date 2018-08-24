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
use infer::canonical::Canonical;
use ty::{self, CanonicalVar, Lift, List, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};

use serialize::{self, Encodable, Encoder, Decodable, Decoder};
use syntax_pos::{Span, DUMMY_SP};
use rustc_data_structures::accumulate_vec::AccumulateVec;
use rustc_data_structures::array_vec::ArrayVec;
use rustc_data_structures::indexed_vec::Idx;

use core::intrinsics;
use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::num::NonZeroUsize;

/// An entity in the Rust typesystem, which can be one of
/// several kinds (only types and lifetimes for now).
/// To reduce memory usage, a `Kind` is a interned pointer,
/// with the lowest 2 bits being reserved for a tag to
/// indicate the type (`Ty` or `Region`) it points to.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Kind<'tcx> {
    ptr: NonZeroUsize,
    marker: PhantomData<(Ty<'tcx>, ty::Region<'tcx>)>
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const REGION_TAG: usize = 0b01;

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub enum UnpackedKind<'tcx> {
    Lifetime(ty::Region<'tcx>),
    Type(Ty<'tcx>),
}

impl<'tcx> UnpackedKind<'tcx> {
    fn pack(self) -> Kind<'tcx> {
        let (tag, ptr) = match self {
            UnpackedKind::Lifetime(lt) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(lt) & TAG_MASK, 0);
                (REGION_TAG, lt as *const _ as usize)
            }
            UnpackedKind::Type(ty) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(ty) & TAG_MASK, 0);
                (TYPE_TAG, ty as *const _ as usize)
            }
        };

        Kind {
            ptr: unsafe {
                NonZeroUsize::new_unchecked(ptr | tag)
            },
            marker: PhantomData
        }
    }
}

impl<'tcx> Ord for Kind<'tcx> {
    fn cmp(&self, other: &Kind) -> Ordering {
        match (self.unpack(), other.unpack()) {
            (UnpackedKind::Type(_), UnpackedKind::Lifetime(_)) => Ordering::Greater,

            (UnpackedKind::Type(ty1), UnpackedKind::Type(ty2)) => {
                ty1.sty.cmp(&ty2.sty)
            }

            (UnpackedKind::Lifetime(reg1), UnpackedKind::Lifetime(reg2)) => reg1.cmp(reg2),

            (UnpackedKind::Lifetime(_), UnpackedKind::Type(_))  => Ordering::Less,
        }
    }
}

impl<'tcx> PartialOrd for Kind<'tcx> {
    fn partial_cmp(&self, other: &Kind) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<'tcx> From<ty::Region<'tcx>> for Kind<'tcx> {
    fn from(r: ty::Region<'tcx>) -> Kind<'tcx> {
        UnpackedKind::Lifetime(r).pack()
    }
}

impl<'tcx> From<Ty<'tcx>> for Kind<'tcx> {
    fn from(ty: Ty<'tcx>) -> Kind<'tcx> {
        UnpackedKind::Type(ty).pack()
    }
}

impl<'tcx> Kind<'tcx> {
    #[inline]
    pub fn unpack(self) -> UnpackedKind<'tcx> {
        let ptr = self.ptr.get();
        unsafe {
            match ptr & TAG_MASK {
                REGION_TAG => UnpackedKind::Lifetime(&*((ptr & !TAG_MASK) as *const _)),
                TYPE_TAG => UnpackedKind::Type(&*((ptr & !TAG_MASK) as *const _)),
                _ => intrinsics::unreachable()
            }
        }
    }
}

impl<'tcx> fmt::Debug for Kind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.unpack() {
            UnpackedKind::Lifetime(lt) => write!(f, "{:?}", lt),
            UnpackedKind::Type(ty) => write!(f, "{:?}", ty),
        }
    }
}

impl<'tcx> fmt::Display for Kind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.unpack() {
            UnpackedKind::Lifetime(lt) => write!(f, "{}", lt),
            UnpackedKind::Type(ty) => write!(f, "{}", ty),
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for Kind<'a> {
    type Lifted = Kind<'tcx>;

    fn lift_to_tcx<'cx, 'gcx>(&self, tcx: TyCtxt<'cx, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match self.unpack() {
            UnpackedKind::Lifetime(a) => a.lift_to_tcx(tcx).map(|a| a.into()),
            UnpackedKind::Type(a) => a.lift_to_tcx(tcx).map(|a| a.into()),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for Kind<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match self.unpack() {
            UnpackedKind::Lifetime(lt) => lt.fold_with(folder).into(),
            UnpackedKind::Type(ty) => ty.fold_with(folder).into(),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match self.unpack() {
            UnpackedKind::Lifetime(lt) => lt.visit_with(visitor),
            UnpackedKind::Type(ty) => ty.visit_with(visitor),
        }
    }
}

impl<'tcx> Encodable for Kind<'tcx> {
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
        self.unpack().encode(e)
    }
}

impl<'tcx> Decodable for Kind<'tcx> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Kind<'tcx>, D::Error> {
        Ok(UnpackedKind::decode(d)?.pack())
    }
}

/// A substitution mapping generic parameters to new values.
pub type Substs<'tcx> = List<Kind<'tcx>>;

impl<'a, 'gcx, 'tcx> Substs<'tcx> {
    /// Creates a Substs that maps each generic parameter to itself.
    pub fn identity_for_item(tcx: TyCtxt<'a, 'gcx, 'tcx>, def_id: DefId)
                             -> &'tcx Substs<'tcx> {
        Substs::for_item(tcx, def_id, |param, _| {
            tcx.mk_param_from_def(param)
        })
    }

    /// Creates a Substs for generic parameter definitions,
    /// by calling closures to obtain each kind.
    /// The closures get to observe the Substs as they're
    /// being built, which can be used to correctly
    /// substitute defaults of generic parameters.
    pub fn for_item<F>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                       def_id: DefId,
                       mut mk_kind: F)
                       -> &'tcx Substs<'tcx>
    where F: FnMut(&ty::GenericParamDef, &[Kind<'tcx>]) -> Kind<'tcx>
    {
        let defs = tcx.generics_of(def_id);
        let count = defs.count();
        let mut substs = if count <= 8 {
            AccumulateVec::Array(ArrayVec::new())
        } else {
            AccumulateVec::Heap(Vec::with_capacity(count))
        };
        Substs::fill_item(&mut substs, tcx, defs, &mut mk_kind);
        tcx.intern_substs(&substs)
    }

    pub fn extend_to<F>(&self,
                        tcx: TyCtxt<'a, 'gcx, 'tcx>,
                        def_id: DefId,
                        mut mk_kind: F)
                        -> &'tcx Substs<'tcx>
    where F: FnMut(&ty::GenericParamDef, &[Kind<'tcx>]) -> Kind<'tcx>
    {
        Substs::for_item(tcx, def_id, |param, substs| {
            match self.get(param.index as usize) {
                Some(&kind) => kind,
                None => mk_kind(param, substs),
            }
        })
    }

    fn fill_item<F>(substs: &mut AccumulateVec<[Kind<'tcx>; 8]>,
                    tcx: TyCtxt<'a, 'gcx, 'tcx>,
                    defs: &ty::Generics,
                    mk_kind: &mut F)
    where F: FnMut(&ty::GenericParamDef, &[Kind<'tcx>]) -> Kind<'tcx>
    {
        if let Some(def_id) = defs.parent {
            let parent_defs = tcx.generics_of(def_id);
            Substs::fill_item(substs, tcx, parent_defs, mk_kind);
        }
        Substs::fill_single(substs, defs, mk_kind)
    }

    fn fill_single<F>(substs: &mut AccumulateVec<[Kind<'tcx>; 8]>,
                      defs: &ty::Generics,
                      mk_kind: &mut F)
    where F: FnMut(&ty::GenericParamDef, &[Kind<'tcx>]) -> Kind<'tcx>
    {
        for param in &defs.params {
            let kind = mk_kind(param, substs);
            assert_eq!(param.index as usize, substs.len());
            match *substs {
                AccumulateVec::Array(ref mut arr) => arr.push(kind),
                AccumulateVec::Heap(ref mut vec) => vec.push(kind),
            }
        }
    }

    pub fn is_noop(&self) -> bool {
        self.is_empty()
    }

    #[inline]
    pub fn types(&'a self) -> impl DoubleEndedIterator<Item=Ty<'tcx>> + 'a {
        self.iter().filter_map(|k| {
            if let UnpackedKind::Type(ty) = k.unpack() {
                Some(ty)
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn regions(&'a self) -> impl DoubleEndedIterator<Item=ty::Region<'tcx>> + 'a {
        self.iter().filter_map(|k| {
            if let UnpackedKind::Lifetime(lt) = k.unpack() {
                Some(lt)
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn type_at(&self, i: usize) -> Ty<'tcx> {
        if let UnpackedKind::Type(ty) = self[i].unpack() {
            ty
        } else {
            bug!("expected type for param #{} in {:?}", i, self);
        }
    }

    #[inline]
    pub fn region_at(&self, i: usize) -> ty::Region<'tcx> {
        if let UnpackedKind::Lifetime(lt) = self[i].unpack() {
            lt
        } else {
            bug!("expected region for param #{} in {:?}", i, self);
        }
    }

    #[inline]
    pub fn type_for_def(&self, def: &ty::GenericParamDef) -> Kind<'tcx> {
        self.type_at(def.index as usize).into()
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
        let defs = tcx.generics_of(source_ancestor);
        tcx.mk_substs(target_substs.iter().chain(&self[defs.params.len()..]).cloned())
    }

    pub fn truncate_to(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, generics: &ty::Generics)
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

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

pub type CanonicalSubsts<'gcx> = Canonical<'gcx, &'gcx Substs<'gcx>>;

impl<'gcx> CanonicalSubsts<'gcx> {
    /// True if this represents a substitution like
    ///
    /// ```text
    /// [?0, ?1, ?2]
    /// ```
    ///
    /// i.e., each thing is mapped to a canonical variable with the same index.
    pub fn is_identity(&self) -> bool {
        self.value.iter().zip(CanonicalVar::new(0)..).all(|(kind, cvar)| {
            match kind.unpack() {
                UnpackedKind::Type(ty) => match ty.sty {
                    ty::Infer(ty::CanonicalTy(cvar1)) => cvar == cvar1,
                    _ => false,
                },

                UnpackedKind::Lifetime(r) => match r {
                    ty::ReCanonical(cvar1) => cvar == *cvar1,
                    _ => false,
                },
            }
        })
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
        let mut folder = SubstFolder { tcx,
                                       substs,
                                       span,
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

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region substitution of the bound
        // regions that appear in a function signature is done using
        // the specialized routine `ty::replace_late_regions()`.
        match *r {
            ty::ReEarlyBound(data) => {
                let r = self.substs.get(data.index as usize).map(|k| k.unpack());
                match r {
                    Some(UnpackedKind::Lifetime(lt)) => {
                        self.shift_region_through_binders(lt)
                    }
                    _ => {
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
            ty::Param(p) => {
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
        let opt_ty = self.substs.get(p.idx as usize).map(|k| k.unpack());
        let ty = match opt_ty {
            Some(UnpackedKind::Type(ty)) => ty,
            _ => {
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

    fn shift_region_through_binders(&self, region: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if self.region_binders_passed == 0 || !region.has_escaping_regions() {
            return region;
        }
        self.tcx().mk_region(ty::fold::shift_region(*region, self.region_binders_passed))
    }
}
