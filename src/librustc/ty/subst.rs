// Type substitutions.

use crate::hir::def_id::DefId;
use crate::infer::canonical::Canonical;
use crate::ty::{self, Lift, List, Ty, TyCtxt, InferConst, ParamConst};
use crate::ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use crate::mir::interpret::ConstValue;

use serialize::{self, Encodable, Encoder, Decodable, Decoder};
use syntax_pos::{Span, DUMMY_SP};
use smallvec::SmallVec;
use rustc_macros::HashStable;

use core::intrinsics;
use std::fmt;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::mem;
use std::num::NonZeroUsize;

/// An entity in the Rust type system, which can be one of
/// several kinds (types, lifetimes, and consts).
/// To reduce memory usage, a `Kind` is a interned pointer,
/// with the lowest 2 bits being reserved for a tag to
/// indicate the type (`Ty`, `Region`, or `Const`) it points to.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Kind<'tcx> {
    ptr: NonZeroUsize,
    marker: PhantomData<(Ty<'tcx>, ty::Region<'tcx>, &'tcx ty::Const<'tcx>)>
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const REGION_TAG: usize = 0b01;
const CONST_TAG: usize = 0b10;

#[derive(Debug, RustcEncodable, RustcDecodable, PartialEq, Eq, PartialOrd, Ord, HashStable)]
pub enum UnpackedKind<'tcx> {
    Lifetime(ty::Region<'tcx>),
    Type(Ty<'tcx>),
    Const(&'tcx ty::Const<'tcx>),
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
            UnpackedKind::Const(ct) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(ct) & TAG_MASK, 0);
                (CONST_TAG, ct as *const _ as usize)
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

impl fmt::Debug for Kind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unpack() {
            UnpackedKind::Lifetime(lt) => lt.fmt(f),
            UnpackedKind::Type(ty) => ty.fmt(f),
            UnpackedKind::Const(ct) => ct.fmt(f),
        }
    }
}

impl<'tcx> Ord for Kind<'tcx> {
    fn cmp(&self, other: &Kind<'_>) -> Ordering {
        self.unpack().cmp(&other.unpack())
    }
}

impl<'tcx> PartialOrd for Kind<'tcx> {
    fn partial_cmp(&self, other: &Kind<'_>) -> Option<Ordering> {
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

impl<'tcx> From<&'tcx ty::Const<'tcx>> for Kind<'tcx> {
    fn from(c: &'tcx ty::Const<'tcx>) -> Kind<'tcx> {
        UnpackedKind::Const(c).pack()
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
                CONST_TAG => UnpackedKind::Const(&*((ptr & !TAG_MASK) as *const _)),
                _ => intrinsics::unreachable()
            }
        }
    }

    /// Unpack the `Kind` as a type when it is known certainly to be a type.
    /// This is true in cases where `Substs` is used in places where the kinds are known
    /// to be limited (e.g. in tuples, where the only parameters are type parameters).
    pub fn expect_ty(self) -> Ty<'tcx> {
        match self.unpack() {
            UnpackedKind::Type(ty) => ty,
            _ => bug!("expected a type, but found another kind"),
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for Kind<'a> {
    type Lifted = Kind<'tcx>;

    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self.unpack() {
            UnpackedKind::Lifetime(lt) => tcx.lift(&lt).map(|lt| lt.into()),
            UnpackedKind::Type(ty) => tcx.lift(&ty).map(|ty| ty.into()),
            UnpackedKind::Const(ct) => tcx.lift(&ct).map(|ct| ct.into()),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for Kind<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        match self.unpack() {
            UnpackedKind::Lifetime(lt) => lt.fold_with(folder).into(),
            UnpackedKind::Type(ty) => ty.fold_with(folder).into(),
            UnpackedKind::Const(ct) => ct.fold_with(folder).into(),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match self.unpack() {
            UnpackedKind::Lifetime(lt) => lt.visit_with(visitor),
            UnpackedKind::Type(ty) => ty.visit_with(visitor),
            UnpackedKind::Const(ct) => ct.visit_with(visitor),
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
pub type InternalSubsts<'tcx> = List<Kind<'tcx>>;

pub type SubstsRef<'tcx> = &'tcx InternalSubsts<'tcx>;

impl<'a, 'tcx> InternalSubsts<'tcx> {
    /// Creates a `InternalSubsts` that maps each generic parameter to itself.
    pub fn identity_for_item(tcx: TyCtxt<'tcx>, def_id: DefId) -> SubstsRef<'tcx> {
        Self::for_item(tcx, def_id, |param, _| {
            tcx.mk_param_from_def(param)
        })
    }

    /// Creates a `InternalSubsts` that maps each generic parameter to a higher-ranked
    /// var bound at index `0`. For types, we use a `BoundVar` index equal to
    /// the type parameter index. For regions, we use the `BoundRegion::BrNamed`
    /// variant (which has a `DefId`).
    pub fn bound_vars_for_item(tcx: TyCtxt<'tcx>, def_id: DefId) -> SubstsRef<'tcx> {
        Self::for_item(tcx, def_id, |param, _| {
            match param.kind {
                ty::GenericParamDefKind::Type { .. } => {
                    tcx.mk_ty(
                        ty::Bound(ty::INNERMOST, ty::BoundTy {
                            var: ty::BoundVar::from(param.index),
                            kind: ty::BoundTyKind::Param(param.name),
                        })
                    ).into()
                }

                ty::GenericParamDefKind::Lifetime => {
                    tcx.mk_region(ty::RegionKind::ReLateBound(
                        ty::INNERMOST,
                        ty::BoundRegion::BrNamed(param.def_id, param.name)
                    )).into()
                }

                ty::GenericParamDefKind::Const => {
                    tcx.mk_const(ty::Const {
                        val: ConstValue::Infer(
                            InferConst::Canonical(ty::INNERMOST, ty::BoundVar::from(param.index))
                        ),
                        ty: tcx.type_of(def_id),
                    }).into()
                }
            }
        })
    }

    /// Creates a `InternalSubsts` for generic parameter definitions,
    /// by calling closures to obtain each kind.
    /// The closures get to observe the `InternalSubsts` as they're
    /// being built, which can be used to correctly
    /// substitute defaults of generic parameters.
    pub fn for_item<F>(tcx: TyCtxt<'tcx>, def_id: DefId, mut mk_kind: F) -> SubstsRef<'tcx>
    where
        F: FnMut(&ty::GenericParamDef, &[Kind<'tcx>]) -> Kind<'tcx>,
    {
        let defs = tcx.generics_of(def_id);
        let count = defs.count();
        let mut substs = SmallVec::with_capacity(count);
        Self::fill_item(&mut substs, tcx, defs, &mut mk_kind);
        tcx.intern_substs(&substs)
    }

    pub fn extend_to<F>(&self, tcx: TyCtxt<'tcx>, def_id: DefId, mut mk_kind: F) -> SubstsRef<'tcx>
    where
        F: FnMut(&ty::GenericParamDef, &[Kind<'tcx>]) -> Kind<'tcx>,
    {
        Self::for_item(tcx, def_id, |param, substs| {
            self.get(param.index as usize)
                .cloned()
                .unwrap_or_else(|| mk_kind(param, substs))
        })
    }

    fn fill_item<F>(
        substs: &mut SmallVec<[Kind<'tcx>; 8]>,
        tcx: TyCtxt<'tcx>,
        defs: &ty::Generics,
        mk_kind: &mut F,
    ) where
        F: FnMut(&ty::GenericParamDef, &[Kind<'tcx>]) -> Kind<'tcx>,
    {
        if let Some(def_id) = defs.parent {
            let parent_defs = tcx.generics_of(def_id);
            Self::fill_item(substs, tcx, parent_defs, mk_kind);
        }
        Self::fill_single(substs, defs, mk_kind)
    }

    fn fill_single<F>(substs: &mut SmallVec<[Kind<'tcx>; 8]>,
                      defs: &ty::Generics,
                      mk_kind: &mut F)
    where F: FnMut(&ty::GenericParamDef, &[Kind<'tcx>]) -> Kind<'tcx>
    {
        substs.reserve(defs.params.len());
        for param in &defs.params {
            let kind = mk_kind(param, substs);
            assert_eq!(param.index as usize, substs.len());
            substs.push(kind);
        }
    }

    pub fn is_noop(&self) -> bool {
        self.is_empty()
    }

    #[inline]
    pub fn types(&'a self) -> impl DoubleEndedIterator<Item = Ty<'tcx>> + 'a {
        self.iter().filter_map(|k| {
            if let UnpackedKind::Type(ty) = k.unpack() {
                Some(ty)
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn regions(&'a self) -> impl DoubleEndedIterator<Item = ty::Region<'tcx>> + 'a {
        self.iter().filter_map(|k| {
            if let UnpackedKind::Lifetime(lt) = k.unpack() {
                Some(lt)
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn consts(&'a self) -> impl DoubleEndedIterator<Item = &'tcx ty::Const<'tcx>> + 'a {
        self.iter().filter_map(|k| {
            if let UnpackedKind::Const(ct) = k.unpack() {
                Some(ct)
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn non_erasable_generics(
        &'a self
    ) -> impl DoubleEndedIterator<Item = UnpackedKind<'tcx>> + 'a {
        self.iter().filter_map(|k| {
            match k.unpack() {
                UnpackedKind::Lifetime(_) => None,
                generic => Some(generic),
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
    pub fn const_at(&self, i: usize) -> &'tcx ty::Const<'tcx> {
        if let UnpackedKind::Const(ct) = self[i].unpack() {
            ct
        } else {
            bug!("expected const for param #{} in {:?}", i, self);
        }
    }

    #[inline]
    pub fn type_for_def(&self, def: &ty::GenericParamDef) -> Kind<'tcx> {
        self.type_at(def.index as usize).into()
    }

    /// Transform from substitutions for a child of `source_ancestor`
    /// (e.g., a trait or impl) to substitutions for the same child
    /// in a different item, with `target_substs` as the base for
    /// the target impl/trait, with the source child-specific
    /// parameters (e.g., method parameters) on top of that base.
    pub fn rebase_onto(
        &self,
        tcx: TyCtxt<'tcx>,
        source_ancestor: DefId,
        target_substs: SubstsRef<'tcx>,
    ) -> SubstsRef<'tcx> {
        let defs = tcx.generics_of(source_ancestor);
        tcx.mk_substs(target_substs.iter().chain(&self[defs.params.len()..]).cloned())
    }

    pub fn truncate_to(&self, tcx: TyCtxt<'tcx>, generics: &ty::Generics) -> SubstsRef<'tcx> {
        tcx.mk_substs(self.iter().take(generics.count()).cloned())
    }
}

impl<'tcx> TypeFoldable<'tcx> for SubstsRef<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let params: SmallVec<[_; 8]> = self.iter().map(|k| k.fold_with(folder)).collect();

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

impl<'tcx> serialize::UseSpecializedDecodable for SubstsRef<'tcx> {}

///////////////////////////////////////////////////////////////////////////
// Public trait `Subst`
//
// Just call `foo.subst(tcx, substs)` to perform a substitution across
// `foo`. Or use `foo.subst_spanned(tcx, substs, Some(span))` when
// there is more information available (for better errors).

pub trait Subst<'tcx>: Sized {
    fn subst(&self, tcx: TyCtxt<'tcx>, substs: &[Kind<'tcx>]) -> Self {
        self.subst_spanned(tcx, substs, None)
    }

    fn subst_spanned(&self, tcx: TyCtxt<'tcx>, substs: &[Kind<'tcx>], span: Option<Span>) -> Self;
}

impl<'tcx, T: TypeFoldable<'tcx>> Subst<'tcx> for T {
    fn subst_spanned(&self, tcx: TyCtxt<'tcx>, substs: &[Kind<'tcx>], span: Option<Span>) -> T {
        let mut folder = SubstFolder { tcx,
                                       substs,
                                       span,
                                       root_ty: None,
                                       ty_stack_depth: 0,
                                       binders_passed: 0 };
        (*self).fold_with(&mut folder)
    }
}

///////////////////////////////////////////////////////////////////////////
// The actual substitution engine itself is a type folder.

struct SubstFolder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    substs: &'a [Kind<'tcx>],

    /// The location for which the substitution is performed, if available.
    span: Option<Span>,

    /// The root type that is being substituted, if available.
    root_ty: Option<Ty<'tcx>>,

    /// Depth of type stack
    ty_stack_depth: usize,

    /// Number of region binders we have passed through while doing the substitution
    binders_passed: u32,
}

impl<'a, 'tcx> TypeFolder<'tcx> for SubstFolder<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> { self.tcx }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T> {
        self.binders_passed += 1;
        let t = t.super_fold_with(self);
        self.binders_passed -= 1;
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
                let rk = self.substs.get(data.index as usize).map(|k| k.unpack());
                match rk {
                    Some(UnpackedKind::Lifetime(lt)) => {
                        self.shift_region_through_binders(lt)
                    }
                    _ => {
                        let span = self.span.unwrap_or(DUMMY_SP);
                        let msg = format!(
                            "Region parameter out of range \
                             when substituting in region {} (root type={:?}) \
                             (index={})",
                            data.name,
                            self.root_ty,
                            data.index);
                        self.tcx.sess.delay_span_bug(span, &msg);
                        r
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

    fn fold_const(&mut self, c: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        if !c.needs_subst() {
            return c;
        }

        if let ConstValue::Param(p) = c.val {
            self.const_for_param(p, c)
        } else {
            c.super_fold_with(self)
        }
    }
}

impl<'a, 'tcx> SubstFolder<'a, 'tcx> {
    fn ty_for_param(&self, p: ty::ParamTy, source_ty: Ty<'tcx>) -> Ty<'tcx> {
        // Look up the type in the substitutions. It really should be in there.
        let opt_ty = self.substs.get(p.index as usize).map(|k| k.unpack());
        let ty = match opt_ty {
            Some(UnpackedKind::Type(ty)) => ty,
            Some(kind) => {
                let span = self.span.unwrap_or(DUMMY_SP);
                span_bug!(
                    span,
                    "expected type for `{:?}` ({:?}/{}) but found {:?} \
                     when substituting (root type={:?}) substs={:?}",
                    p,
                    source_ty,
                    p.index,
                    kind,
                    self.root_ty,
                    self.substs,
                );
            }
            None => {
                let span = self.span.unwrap_or(DUMMY_SP);
                span_bug!(
                    span,
                    "type parameter `{:?}` ({:?}/{}) out of range \
                     when substituting (root type={:?}) substs={:?}",
                    p,
                    source_ty,
                    p.index,
                    self.root_ty,
                    self.substs,
                );
            }
        };

        self.shift_vars_through_binders(ty)
    }

    fn const_for_param(
        &self,
        p: ParamConst,
        source_ct: &'tcx ty::Const<'tcx>
    ) -> &'tcx ty::Const<'tcx> {
        // Look up the const in the substitutions. It really should be in there.
        let opt_ct = self.substs.get(p.index as usize).map(|k| k.unpack());
        let ct = match opt_ct {
            Some(UnpackedKind::Const(ct)) => ct,
            Some(kind) => {
                let span = self.span.unwrap_or(DUMMY_SP);
                span_bug!(
                    span,
                    "expected const for `{:?}` ({:?}/{}) but found {:?} \
                     when substituting substs={:?}",
                    p,
                    source_ct,
                    p.index,
                    kind,
                    self.substs,
                );
            }
            None => {
                let span = self.span.unwrap_or(DUMMY_SP);
                span_bug!(
                    span,
                    "const parameter `{:?}` ({:?}/{}) out of range \
                     when substituting substs={:?}",
                    p,
                    source_ct,
                    p.index,
                    self.substs,
                );
            }
        };

        self.shift_vars_through_binders(ct)
    }

    /// It is sometimes necessary to adjust the De Bruijn indices during substitution. This occurs
    /// when we are substituting a type with escaping bound vars into a context where we have
    /// passed through binders. That's quite a mouthful. Let's see an example:
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
    /// over the inner binder (remember that we count De Bruijn indices from 1). However, in the
    /// definition of `MetaFunc`, the binder is not visible, so the type `&'a int` will have a
    /// De Bruijn index of 1. It's only during the substitution that we can see we must increase the
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
    /// first case we do not increase the De Bruijn index and in the second case we do. The reason
    /// is that only in the second case have we passed through a fn binder.
    fn shift_vars_through_binders<T: TypeFoldable<'tcx>>(&self, val: T) -> T {
        debug!("shift_vars(val={:?}, binders_passed={:?}, has_escaping_bound_vars={:?})",
               val, self.binders_passed, val.has_escaping_bound_vars());

        if self.binders_passed == 0 || !val.has_escaping_bound_vars() {
            return val;
        }

        let result = ty::fold::shift_vars(self.tcx(), &val, self.binders_passed);
        debug!("shift_vars: shifted result = {:?}", result);

        result
    }

    fn shift_region_through_binders(&self, region: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if self.binders_passed == 0 || !region.has_escaping_bound_vars() {
            return region;
        }
        ty::fold::shift_region(self.tcx, region, self.binders_passed)
    }
}

pub type CanonicalUserSubsts<'tcx> = Canonical<'tcx, UserSubsts<'tcx>>;

/// Stores the user-given substs to reach some fully qualified path
/// (e.g., `<T>::Item` or `<T as Trait>::Item`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, HashStable)]
pub struct UserSubsts<'tcx> {
    /// The substitutions for the item as given by the user.
    pub substs: SubstsRef<'tcx>,

    /// The self type, in the case of a `<T>::Item` path (when applied
    /// to an inherent impl). See `UserSelfTy` below.
    pub user_self_ty: Option<UserSelfTy<'tcx>>,
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for UserSubsts<'tcx> {
        substs,
        user_self_ty,
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for UserSubsts<'a> {
        type Lifted = UserSubsts<'tcx>;
        substs,
        user_self_ty,
    }
}

/// Specifies the user-given self type. In the case of a path that
/// refers to a member in an inherent impl, this self type is
/// sometimes needed to constrain the type parameters on the impl. For
/// example, in this code:
///
/// ```
/// struct Foo<T> { }
/// impl<A> Foo<A> { fn method() { } }
/// ```
///
/// when you then have a path like `<Foo<&'static u32>>::method`,
/// this struct would carry the `DefId` of the impl along with the
/// self type `Foo<u32>`. Then we can instantiate the parameters of
/// the impl (with the substs from `UserSubsts`) and apply those to
/// the self type, giving `Foo<?A>`. Finally, we unify that with
/// the self type here, which contains `?A` to be `&'static u32`
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, HashStable)]
pub struct UserSelfTy<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for UserSelfTy<'tcx> {
        impl_def_id,
        self_ty,
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for UserSelfTy<'a> {
        type Lifted = UserSelfTy<'tcx>;
        impl_def_id,
        self_ty,
    }
}
