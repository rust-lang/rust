// Type substitutions.

use crate::ty::codec::{TyDecoder, TyEncoder};
use crate::ty::fold::{ir::TypeFolder, FallibleTypeFolder, TypeFoldable, TypeSuperFoldable};
use crate::ty::sty::{ClosureSubsts, GeneratorSubsts, InlineConstSubsts};
use crate::ty::visit::{TypeVisitable, TypeVisitor};
use crate::ty::{self, ir, Lift, List, ParamConst, Ty, TyCtxt};

use rustc_data_structures::intern::Interned;
use rustc_errors::{DiagnosticArgValue, IntoDiagnosticArg};
use rustc_hir::def_id::DefId;
use rustc_macros::HashStable;
use rustc_serialize::{self, Decodable, Encodable};
use rustc_type_ir::WithCachedTypeInfo;
use smallvec::SmallVec;

use core::intrinsics;
use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::{ControlFlow, Deref};
use std::slice;

/// An entity in the Rust type system, which can be one of
/// several kinds (types, lifetimes, and consts).
/// To reduce memory usage, a `GenericArg` is an interned pointer,
/// with the lowest 2 bits being reserved for a tag to
/// indicate the type (`Ty`, `Region`, or `Const`) it points to.
///
/// Note: the `PartialEq`, `Eq` and `Hash` derives are only valid because `Ty`,
/// `Region` and `Const` are all interned.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct GenericArg<'tcx> {
    ptr: NonZeroUsize,
    marker: PhantomData<(Ty<'tcx>, ty::Region<'tcx>, ty::Const<'tcx>)>,
}

impl<'tcx> IntoDiagnosticArg for GenericArg<'tcx> {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.to_string().into_diagnostic_arg()
    }
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const REGION_TAG: usize = 0b01;
const CONST_TAG: usize = 0b10;

#[derive(Debug, TyEncodable, TyDecodable, PartialEq, Eq, PartialOrd, Ord)]
pub enum GenericArgKind<'tcx> {
    Lifetime(ty::Region<'tcx>),
    Type(Ty<'tcx>),
    Const(ty::Const<'tcx>),
}

/// This function goes from `&'a [Ty<'tcx>]` to `&'a [GenericArg<'tcx>]`
///
/// This is sound as, for types, `GenericArg` is just
/// `NonZeroUsize::new_unchecked(ty as *const _ as usize)` as
/// long as we use `0` for the `TYPE_TAG`.
pub fn ty_slice_as_generic_args<'a, 'tcx>(ts: &'a [Ty<'tcx>]) -> &'a [GenericArg<'tcx>] {
    assert_eq!(TYPE_TAG, 0);
    // SAFETY: the whole slice is valid and immutable.
    // `Ty` and `GenericArg` is explained above.
    unsafe { slice::from_raw_parts(ts.as_ptr().cast(), ts.len()) }
}

impl<'tcx> List<Ty<'tcx>> {
    /// Allows to freely switch between `List<Ty<'tcx>>` and `List<GenericArg<'tcx>>`.
    ///
    /// As lists are interned, `List<Ty<'tcx>>` and `List<GenericArg<'tcx>>` have
    /// be interned together, see `intern_type_list` for more details.
    #[inline]
    pub fn as_substs(&'tcx self) -> SubstsRef<'tcx> {
        assert_eq!(TYPE_TAG, 0);
        // SAFETY: `List<T>` is `#[repr(C)]`. `Ty` and `GenericArg` is explained above.
        unsafe { &*(self as *const List<Ty<'tcx>> as *const List<GenericArg<'tcx>>) }
    }
}

impl<'tcx> GenericArgKind<'tcx> {
    #[inline]
    fn pack(self) -> GenericArg<'tcx> {
        let (tag, ptr) = match self {
            GenericArgKind::Lifetime(lt) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(&*lt.0.0) & TAG_MASK, 0);
                (REGION_TAG, lt.0.0 as *const ty::RegionKind<'tcx> as usize)
            }
            GenericArgKind::Type(ty) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(&*ty.0.0) & TAG_MASK, 0);
                (TYPE_TAG, ty.0.0 as *const WithCachedTypeInfo<ty::TyKind<'tcx>> as usize)
            }
            GenericArgKind::Const(ct) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(&*ct.0.0) & TAG_MASK, 0);
                (CONST_TAG, ct.0.0 as *const ty::ConstData<'tcx> as usize)
            }
        };

        GenericArg { ptr: unsafe { NonZeroUsize::new_unchecked(ptr | tag) }, marker: PhantomData }
    }
}

impl<'tcx> fmt::Debug for GenericArg<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unpack() {
            GenericArgKind::Lifetime(lt) => lt.fmt(f),
            GenericArgKind::Type(ty) => ty.fmt(f),
            GenericArgKind::Const(ct) => ct.fmt(f),
        }
    }
}

impl<'tcx> Ord for GenericArg<'tcx> {
    fn cmp(&self, other: &GenericArg<'tcx>) -> Ordering {
        self.unpack().cmp(&other.unpack())
    }
}

impl<'tcx> PartialOrd for GenericArg<'tcx> {
    fn partial_cmp(&self, other: &GenericArg<'tcx>) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<'tcx> From<ty::Region<'tcx>> for GenericArg<'tcx> {
    #[inline]
    fn from(r: ty::Region<'tcx>) -> GenericArg<'tcx> {
        GenericArgKind::Lifetime(r).pack()
    }
}

impl<'tcx> From<Ty<'tcx>> for GenericArg<'tcx> {
    #[inline]
    fn from(ty: Ty<'tcx>) -> GenericArg<'tcx> {
        GenericArgKind::Type(ty).pack()
    }
}

impl<'tcx> From<ty::Const<'tcx>> for GenericArg<'tcx> {
    #[inline]
    fn from(c: ty::Const<'tcx>) -> GenericArg<'tcx> {
        GenericArgKind::Const(c).pack()
    }
}

impl<'tcx> From<ty::Term<'tcx>> for GenericArg<'tcx> {
    fn from(value: ty::Term<'tcx>) -> Self {
        match value.unpack() {
            ty::TermKind::Ty(t) => t.into(),
            ty::TermKind::Const(c) => c.into(),
        }
    }
}

impl<'tcx> GenericArg<'tcx> {
    #[inline]
    pub fn unpack(self) -> GenericArgKind<'tcx> {
        let ptr = self.ptr.get();
        // SAFETY: use of `Interned::new_unchecked` here is ok because these
        // pointers were originally created from `Interned` types in `pack()`,
        // and this is just going in the other direction.
        unsafe {
            match ptr & TAG_MASK {
                REGION_TAG => GenericArgKind::Lifetime(ty::Region(Interned::new_unchecked(
                    &*((ptr & !TAG_MASK) as *const ty::RegionKind<'tcx>),
                ))),
                TYPE_TAG => GenericArgKind::Type(Ty(Interned::new_unchecked(
                    &*((ptr & !TAG_MASK) as *const WithCachedTypeInfo<ty::TyKind<'tcx>>),
                ))),
                CONST_TAG => GenericArgKind::Const(ty::Const(Interned::new_unchecked(
                    &*((ptr & !TAG_MASK) as *const ty::ConstData<'tcx>),
                ))),
                _ => intrinsics::unreachable(),
            }
        }
    }

    /// Unpack the `GenericArg` as a region when it is known certainly to be a region.
    pub fn expect_region(self) -> ty::Region<'tcx> {
        match self.unpack() {
            GenericArgKind::Lifetime(lt) => lt,
            _ => bug!("expected a region, but found another kind"),
        }
    }

    /// Unpack the `GenericArg` as a type when it is known certainly to be a type.
    /// This is true in cases where `Substs` is used in places where the kinds are known
    /// to be limited (e.g. in tuples, where the only parameters are type parameters).
    pub fn expect_ty(self) -> Ty<'tcx> {
        match self.unpack() {
            GenericArgKind::Type(ty) => ty,
            _ => bug!("expected a type, but found another kind"),
        }
    }

    /// Unpack the `GenericArg` as a const when it is known certainly to be a const.
    pub fn expect_const(self) -> ty::Const<'tcx> {
        match self.unpack() {
            GenericArgKind::Const(c) => c,
            _ => bug!("expected a const, but found another kind"),
        }
    }

    pub fn is_non_region_infer(self) -> bool {
        match self.unpack() {
            GenericArgKind::Lifetime(_) => false,
            GenericArgKind::Type(ty) => ty.is_ty_or_numeric_infer(),
            GenericArgKind::Const(ct) => ct.is_ct_infer(),
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for GenericArg<'a> {
    type Lifted = GenericArg<'tcx>;

    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self.unpack() {
            GenericArgKind::Lifetime(lt) => tcx.lift(lt).map(|lt| lt.into()),
            GenericArgKind::Type(ty) => tcx.lift(ty).map(|ty| ty.into()),
            GenericArgKind::Const(ct) => tcx.lift(ct).map(|ct| ct.into()),
        }
    }
}

impl<'tcx> ir::TypeFoldable<TyCtxt<'tcx>> for GenericArg<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        match self.unpack() {
            GenericArgKind::Lifetime(lt) => lt.try_fold_with(folder).map(Into::into),
            GenericArgKind::Type(ty) => ty.try_fold_with(folder).map(Into::into),
            GenericArgKind::Const(ct) => ct.try_fold_with(folder).map(Into::into),
        }
    }
}

impl<'tcx> ir::TypeVisitable<TyCtxt<'tcx>> for GenericArg<'tcx> {
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self.unpack() {
            GenericArgKind::Lifetime(lt) => lt.visit_with(visitor),
            GenericArgKind::Type(ty) => ty.visit_with(visitor),
            GenericArgKind::Const(ct) => ct.visit_with(visitor),
        }
    }
}

impl<'tcx, E: TyEncoder<I = TyCtxt<'tcx>>> Encodable<E> for GenericArg<'tcx> {
    fn encode(&self, e: &mut E) {
        self.unpack().encode(e)
    }
}

impl<'tcx, D: TyDecoder<I = TyCtxt<'tcx>>> Decodable<D> for GenericArg<'tcx> {
    fn decode(d: &mut D) -> GenericArg<'tcx> {
        GenericArgKind::decode(d).pack()
    }
}

/// List of generic arguments that are gonna be used to substitute generic parameters.
pub type InternalSubsts<'tcx> = List<GenericArg<'tcx>>;

pub type SubstsRef<'tcx> = &'tcx InternalSubsts<'tcx>;

impl<'tcx> InternalSubsts<'tcx> {
    /// Checks whether all elements of this list are types, if so, transmute.
    pub fn try_as_type_list(&'tcx self) -> Option<&'tcx List<Ty<'tcx>>> {
        if self.iter().all(|arg| matches!(arg.unpack(), GenericArgKind::Type(_))) {
            assert_eq!(TYPE_TAG, 0);
            // SAFETY: All elements are types, see `List<Ty<'tcx>>::as_substs`.
            Some(unsafe { &*(self as *const List<GenericArg<'tcx>> as *const List<Ty<'tcx>>) })
        } else {
            None
        }
    }

    /// Interpret these substitutions as the substitutions of a closure type.
    /// Closure substitutions have a particular structure controlled by the
    /// compiler that encodes information like the signature and closure kind;
    /// see `ty::ClosureSubsts` struct for more comments.
    pub fn as_closure(&'tcx self) -> ClosureSubsts<'tcx> {
        ClosureSubsts { substs: self }
    }

    /// Interpret these substitutions as the substitutions of a generator type.
    /// Generator substitutions have a particular structure controlled by the
    /// compiler that encodes information like the signature and generator kind;
    /// see `ty::GeneratorSubsts` struct for more comments.
    pub fn as_generator(&'tcx self) -> GeneratorSubsts<'tcx> {
        GeneratorSubsts { substs: self }
    }

    /// Interpret these substitutions as the substitutions of an inline const.
    /// Inline const substitutions have a particular structure controlled by the
    /// compiler that encodes information like the inferred type;
    /// see `ty::InlineConstSubsts` struct for more comments.
    pub fn as_inline_const(&'tcx self) -> InlineConstSubsts<'tcx> {
        InlineConstSubsts { substs: self }
    }

    /// Creates an `InternalSubsts` that maps each generic parameter to itself.
    pub fn identity_for_item(tcx: TyCtxt<'tcx>, def_id: DefId) -> SubstsRef<'tcx> {
        Self::for_item(tcx, def_id, |param, _| tcx.mk_param_from_def(param))
    }

    /// Creates an `InternalSubsts` for generic parameter definitions,
    /// by calling closures to obtain each kind.
    /// The closures get to observe the `InternalSubsts` as they're
    /// being built, which can be used to correctly
    /// substitute defaults of generic parameters.
    pub fn for_item<F>(tcx: TyCtxt<'tcx>, def_id: DefId, mut mk_kind: F) -> SubstsRef<'tcx>
    where
        F: FnMut(&ty::GenericParamDef, &[GenericArg<'tcx>]) -> GenericArg<'tcx>,
    {
        let defs = tcx.generics_of(def_id);
        let count = defs.count();
        let mut substs = SmallVec::with_capacity(count);
        Self::fill_item(&mut substs, tcx, defs, &mut mk_kind);
        tcx.intern_substs(&substs)
    }

    pub fn extend_to<F>(&self, tcx: TyCtxt<'tcx>, def_id: DefId, mut mk_kind: F) -> SubstsRef<'tcx>
    where
        F: FnMut(&ty::GenericParamDef, &[GenericArg<'tcx>]) -> GenericArg<'tcx>,
    {
        Self::for_item(tcx, def_id, |param, substs| {
            self.get(param.index as usize).cloned().unwrap_or_else(|| mk_kind(param, substs))
        })
    }

    pub fn fill_item<F>(
        substs: &mut SmallVec<[GenericArg<'tcx>; 8]>,
        tcx: TyCtxt<'tcx>,
        defs: &ty::Generics,
        mk_kind: &mut F,
    ) where
        F: FnMut(&ty::GenericParamDef, &[GenericArg<'tcx>]) -> GenericArg<'tcx>,
    {
        if let Some(def_id) = defs.parent {
            let parent_defs = tcx.generics_of(def_id);
            Self::fill_item(substs, tcx, parent_defs, mk_kind);
        }
        Self::fill_single(substs, defs, mk_kind)
    }

    pub fn fill_single<F>(
        substs: &mut SmallVec<[GenericArg<'tcx>; 8]>,
        defs: &ty::Generics,
        mk_kind: &mut F,
    ) where
        F: FnMut(&ty::GenericParamDef, &[GenericArg<'tcx>]) -> GenericArg<'tcx>,
    {
        substs.reserve(defs.params.len());
        for param in &defs.params {
            let kind = mk_kind(param, substs);
            assert_eq!(param.index as usize, substs.len(), "{substs:#?}, {defs:#?}");
            substs.push(kind);
        }
    }

    // Extend an `original_substs` list to the full number of substs expected by `def_id`,
    // filling in the missing parameters with error ty/ct or 'static regions.
    pub fn extend_with_error(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        original_substs: &[GenericArg<'tcx>],
    ) -> SubstsRef<'tcx> {
        ty::InternalSubsts::for_item(tcx, def_id, |def, substs| {
            if let Some(subst) = original_substs.get(def.index as usize) {
                *subst
            } else {
                def.to_error(tcx, substs)
            }
        })
    }

    #[inline]
    pub fn types(&'tcx self) -> impl DoubleEndedIterator<Item = Ty<'tcx>> + 'tcx {
        self.iter()
            .filter_map(|k| if let GenericArgKind::Type(ty) = k.unpack() { Some(ty) } else { None })
    }

    #[inline]
    pub fn regions(&'tcx self) -> impl DoubleEndedIterator<Item = ty::Region<'tcx>> + 'tcx {
        self.iter().filter_map(|k| {
            if let GenericArgKind::Lifetime(lt) = k.unpack() { Some(lt) } else { None }
        })
    }

    #[inline]
    pub fn consts(&'tcx self) -> impl DoubleEndedIterator<Item = ty::Const<'tcx>> + 'tcx {
        self.iter().filter_map(|k| {
            if let GenericArgKind::Const(ct) = k.unpack() { Some(ct) } else { None }
        })
    }

    #[inline]
    pub fn non_erasable_generics(
        &'tcx self,
    ) -> impl DoubleEndedIterator<Item = GenericArgKind<'tcx>> + 'tcx {
        self.iter().filter_map(|k| match k.unpack() {
            GenericArgKind::Lifetime(_) => None,
            generic => Some(generic),
        })
    }

    #[inline]
    #[track_caller]
    pub fn type_at(&self, i: usize) -> Ty<'tcx> {
        if let GenericArgKind::Type(ty) = self[i].unpack() {
            ty
        } else {
            bug!("expected type for param #{} in {:?}", i, self);
        }
    }

    #[inline]
    #[track_caller]
    pub fn region_at(&self, i: usize) -> ty::Region<'tcx> {
        if let GenericArgKind::Lifetime(lt) = self[i].unpack() {
            lt
        } else {
            bug!("expected region for param #{} in {:?}", i, self);
        }
    }

    #[inline]
    #[track_caller]
    pub fn const_at(&self, i: usize) -> ty::Const<'tcx> {
        if let GenericArgKind::Const(ct) = self[i].unpack() {
            ct
        } else {
            bug!("expected const for param #{} in {:?}", i, self);
        }
    }

    #[inline]
    #[track_caller]
    pub fn type_for_def(&self, def: &ty::GenericParamDef) -> GenericArg<'tcx> {
        self.type_at(def.index as usize).into()
    }

    /// Transform from substitutions for a child of `source_ancestor`
    /// (e.g., a trait or impl) to substitutions for the same child
    /// in a different item, with `target_substs` as the base for
    /// the target impl/trait, with the source child-specific
    /// parameters (e.g., method parameters) on top of that base.
    ///
    /// For example given:
    ///
    /// ```no_run
    /// trait X<S> { fn f<T>(); }
    /// impl<U> X<U> for U { fn f<V>() {} }
    /// ```
    ///
    /// * If `self` is `[Self, S, T]`: the identity substs of `f` in the trait.
    /// * If `source_ancestor` is the def_id of the trait.
    /// * If `target_substs` is `[U]`, the substs for the impl.
    /// * Then we will return `[U, T]`, the subst for `f` in the impl that
    ///   are needed for it to match the trait.
    pub fn rebase_onto(
        &self,
        tcx: TyCtxt<'tcx>,
        source_ancestor: DefId,
        target_substs: SubstsRef<'tcx>,
    ) -> SubstsRef<'tcx> {
        let defs = tcx.generics_of(source_ancestor);
        tcx.mk_substs(target_substs.iter().chain(self.iter().skip(defs.params.len())))
    }

    pub fn truncate_to(&self, tcx: TyCtxt<'tcx>, generics: &ty::Generics) -> SubstsRef<'tcx> {
        tcx.mk_substs(self.iter().take(generics.count()))
    }
}

impl<'tcx> ir::TypeFoldable<TyCtxt<'tcx>> for SubstsRef<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        // This code is hot enough that it's worth specializing for the most
        // common length lists, to avoid the overhead of `SmallVec` creation.
        // The match arms are in order of frequency. The 1, 2, and 0 cases are
        // typically hit in 90--99.99% of cases. When folding doesn't change
        // the substs, it's faster to reuse the existing substs rather than
        // calling `intern_substs`.
        match self.len() {
            1 => {
                let param0 = self[0].try_fold_with(folder)?;
                if param0 == self[0] {
                    Ok(self)
                } else {
                    Ok(folder.interner().intern_substs(&[param0]))
                }
            }
            2 => {
                let param0 = self[0].try_fold_with(folder)?;
                let param1 = self[1].try_fold_with(folder)?;
                if param0 == self[0] && param1 == self[1] {
                    Ok(self)
                } else {
                    Ok(folder.interner().intern_substs(&[param0, param1]))
                }
            }
            0 => Ok(self),
            _ => ty::util::fold_list(self, folder, |tcx, v| tcx.intern_substs(v)),
        }
    }
}

impl<'tcx> ir::TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<Ty<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<'tcx>>(self, folder: &mut F) -> Result<Self, F::Error> {
        // This code is fairly hot, though not as hot as `SubstsRef`.
        //
        // When compiling stage 2, I get the following results:
        //
        // len |   total   |   %
        // --- | --------- | -----
        //  2  |  15083590 |  48.1
        //  3  |   7540067 |  24.0
        //  1  |   5300377 |  16.9
        //  4  |   1351897 |   4.3
        //  0  |   1256849 |   4.0
        //
        // I've tried it with some private repositories and got
        // close to the same result, with 4 and 0 swapping places
        // sometimes.
        match self.len() {
            2 => {
                let param0 = self[0].try_fold_with(folder)?;
                let param1 = self[1].try_fold_with(folder)?;
                if param0 == self[0] && param1 == self[1] {
                    Ok(self)
                } else {
                    Ok(folder.interner().intern_type_list(&[param0, param1]))
                }
            }
            _ => ty::util::fold_list(self, folder, |tcx, v| tcx.intern_type_list(v)),
        }
    }
}

impl<'tcx, T: TypeVisitable<'tcx>> ir::TypeVisitable<TyCtxt<'tcx>> for &'tcx ty::List<T> {
    #[inline]
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

/// Similar to [`super::Binder`] except that it tracks early bound generics, i.e. `struct Foo<T>(T)`
/// needs `T` substituted immediately. This type primarily exists to avoid forgetting to call
/// `subst`.
///
/// If you don't have anything to `subst`, you may be looking for
/// [`subst_identity`](EarlyBinder::subst_identity) or [`skip_binder`](EarlyBinder::skip_binder).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(Encodable, Decodable, HashStable)]
pub struct EarlyBinder<T>(pub T);

/// For early binders, you should first call `subst` before using any visitors.
impl<'tcx, T> !ir::TypeFoldable<TyCtxt<'tcx>> for ty::EarlyBinder<T> {}
impl<'tcx, T> !ir::TypeVisitable<TyCtxt<'tcx>> for ty::EarlyBinder<T> {}

impl<T> EarlyBinder<T> {
    pub fn as_ref(&self) -> EarlyBinder<&T> {
        EarlyBinder(&self.0)
    }

    pub fn map_bound_ref<F, U>(&self, f: F) -> EarlyBinder<U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U>(self, f: F) -> EarlyBinder<U>
    where
        F: FnOnce(T) -> U,
    {
        let value = f(self.0);
        EarlyBinder(value)
    }

    pub fn try_map_bound<F, U, E>(self, f: F) -> Result<EarlyBinder<U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let value = f(self.0)?;
        Ok(EarlyBinder(value))
    }

    pub fn rebind<U>(&self, value: U) -> EarlyBinder<U> {
        EarlyBinder(value)
    }

    /// Skips the binder and returns the "bound" value.
    /// This can be used to extract data that does not depend on generic parameters
    /// (e.g., getting the `DefId` of the inner value or getting the number of
    /// arguments of an `FnSig`). Otherwise, consider using
    /// [`subst_identity`](EarlyBinder::subst_identity).
    ///
    /// See also [`Binder::skip_binder`](super::Binder::skip_binder), which is
    /// the analogous operation on [`super::Binder`].
    pub fn skip_binder(self) -> T {
        self.0
    }
}

impl<T> EarlyBinder<Option<T>> {
    pub fn transpose(self) -> Option<EarlyBinder<T>> {
        self.0.map(|v| EarlyBinder(v))
    }
}

impl<T, U> EarlyBinder<(T, U)> {
    pub fn transpose_tuple2(self) -> (EarlyBinder<T>, EarlyBinder<U>) {
        (EarlyBinder(self.0.0), EarlyBinder(self.0.1))
    }
}

impl<'tcx, 's, I: IntoIterator> EarlyBinder<I>
where
    I::Item: TypeFoldable<'tcx>,
{
    pub fn subst_iter(
        self,
        tcx: TyCtxt<'tcx>,
        substs: &'s [GenericArg<'tcx>],
    ) -> SubstIter<'s, 'tcx, I> {
        SubstIter { it: self.0.into_iter(), tcx, substs }
    }
}

pub struct SubstIter<'s, 'tcx, I: IntoIterator> {
    it: I::IntoIter,
    tcx: TyCtxt<'tcx>,
    substs: &'s [GenericArg<'tcx>],
}

impl<'tcx, I: IntoIterator> Iterator for SubstIter<'_, 'tcx, I>
where
    I::Item: TypeFoldable<'tcx>,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        Some(EarlyBinder(self.it.next()?).subst(self.tcx, self.substs))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'tcx, I: IntoIterator> DoubleEndedIterator for SubstIter<'_, 'tcx, I>
where
    I::IntoIter: DoubleEndedIterator,
    I::Item: TypeFoldable<'tcx>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(EarlyBinder(self.it.next_back()?).subst(self.tcx, self.substs))
    }
}

impl<'tcx, I: IntoIterator> ExactSizeIterator for SubstIter<'_, 'tcx, I>
where
    I::IntoIter: ExactSizeIterator,
    I::Item: TypeFoldable<'tcx>,
{
}

impl<'tcx, 's, I: IntoIterator> EarlyBinder<I>
where
    I::Item: Deref,
    <I::Item as Deref>::Target: Copy + TypeFoldable<'tcx>,
{
    pub fn subst_iter_copied(
        self,
        tcx: TyCtxt<'tcx>,
        substs: &'s [GenericArg<'tcx>],
    ) -> SubstIterCopied<'s, 'tcx, I> {
        SubstIterCopied { it: self.0.into_iter(), tcx, substs }
    }
}

pub struct SubstIterCopied<'a, 'tcx, I: IntoIterator> {
    it: I::IntoIter,
    tcx: TyCtxt<'tcx>,
    substs: &'a [GenericArg<'tcx>],
}

impl<'tcx, I: IntoIterator> Iterator for SubstIterCopied<'_, 'tcx, I>
where
    I::Item: Deref,
    <I::Item as Deref>::Target: Copy + TypeFoldable<'tcx>,
{
    type Item = <I::Item as Deref>::Target;

    fn next(&mut self) -> Option<Self::Item> {
        Some(EarlyBinder(*self.it.next()?).subst(self.tcx, self.substs))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'tcx, I: IntoIterator> DoubleEndedIterator for SubstIterCopied<'_, 'tcx, I>
where
    I::IntoIter: DoubleEndedIterator,
    I::Item: Deref,
    <I::Item as Deref>::Target: Copy + TypeFoldable<'tcx>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(EarlyBinder(*self.it.next_back()?).subst(self.tcx, self.substs))
    }
}

impl<'tcx, I: IntoIterator> ExactSizeIterator for SubstIterCopied<'_, 'tcx, I>
where
    I::IntoIter: ExactSizeIterator,
    I::Item: Deref,
    <I::Item as Deref>::Target: Copy + TypeFoldable<'tcx>,
{
}

pub struct EarlyBinderIter<T> {
    t: T,
}

impl<T: IntoIterator> EarlyBinder<T> {
    pub fn transpose_iter(self) -> EarlyBinderIter<T::IntoIter> {
        EarlyBinderIter { t: self.0.into_iter() }
    }
}

impl<T: Iterator> Iterator for EarlyBinderIter<T> {
    type Item = EarlyBinder<T::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.t.next().map(|i| EarlyBinder(i))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.t.size_hint()
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> ty::EarlyBinder<T> {
    pub fn subst(self, tcx: TyCtxt<'tcx>, substs: &[GenericArg<'tcx>]) -> T {
        let mut folder = SubstFolder { tcx, substs, binders_passed: 0 };
        self.0.fold_with(&mut folder)
    }

    /// Makes the identity substitution `T0 => T0, ..., TN => TN`.
    /// Conceptually, this converts universally bound variables into placeholders
    /// when inside of a given item.
    ///
    /// For example, consider `for<T> fn foo<T>(){ .. }`:
    /// - Outside of `foo`, `T` is bound (represented by the presence of `EarlyBinder`).
    /// - Inside of the body of `foo`, we treat `T` as a placeholder by calling
    /// `subst_identity` to discharge the `EarlyBinder`.
    pub fn subst_identity(self) -> T {
        self.0
    }

    /// Returns the inner value, but only if it contains no bound vars.
    pub fn no_bound_vars(self) -> Option<T> {
        if !self.0.needs_subst() { Some(self.0) } else { None }
    }
}

///////////////////////////////////////////////////////////////////////////
// The actual substitution engine itself is a type folder.

struct SubstFolder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    substs: &'a [GenericArg<'tcx>],

    /// Number of region binders we have passed through while doing the substitution
    binders_passed: u32,
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for SubstFolder<'a, 'tcx> {
    #[inline]
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.binders_passed += 1;
        let t = t.super_fold_with(self);
        self.binders_passed -= 1;
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        #[cold]
        #[inline(never)]
        fn region_param_out_of_range(data: ty::EarlyBoundRegion, substs: &[GenericArg<'_>]) -> ! {
            bug!(
                "Region parameter out of range when substituting in region {} (index={}, substs = {:?})",
                data.name,
                data.index,
                substs,
            )
        }

        #[cold]
        #[inline(never)]
        fn region_param_invalid(data: ty::EarlyBoundRegion, other: GenericArgKind<'_>) -> ! {
            bug!(
                "Unexpected parameter {:?} when substituting in region {} (index={})",
                other,
                data.name,
                data.index
            )
        }

        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region substitution of the bound
        // regions that appear in a function signature is done using
        // the specialized routine `ty::replace_late_regions()`.
        match *r {
            ty::ReEarlyBound(data) => {
                let rk = self.substs.get(data.index as usize).map(|k| k.unpack());
                match rk {
                    Some(GenericArgKind::Lifetime(lt)) => self.shift_region_through_binders(lt),
                    Some(other) => region_param_invalid(data, other),
                    None => region_param_out_of_range(data, self.substs),
                }
            }
            _ => r,
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_subst() {
            return t;
        }

        match *t.kind() {
            ty::Param(p) => self.ty_for_param(p, t),
            _ => t.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Param(p) = c.kind() {
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
            Some(GenericArgKind::Type(ty)) => ty,
            Some(kind) => self.type_param_expected(p, source_ty, kind),
            None => self.type_param_out_of_range(p, source_ty),
        };

        self.shift_vars_through_binders(ty)
    }

    #[cold]
    #[inline(never)]
    fn type_param_expected(&self, p: ty::ParamTy, ty: Ty<'tcx>, kind: GenericArgKind<'tcx>) -> ! {
        bug!(
            "expected type for `{:?}` ({:?}/{}) but found {:?} when substituting, substs={:?}",
            p,
            ty,
            p.index,
            kind,
            self.substs,
        )
    }

    #[cold]
    #[inline(never)]
    fn type_param_out_of_range(&self, p: ty::ParamTy, ty: Ty<'tcx>) -> ! {
        bug!(
            "type parameter `{:?}` ({:?}/{}) out of range when substituting, substs={:?}",
            p,
            ty,
            p.index,
            self.substs,
        )
    }

    fn const_for_param(&self, p: ParamConst, source_ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        // Look up the const in the substitutions. It really should be in there.
        let opt_ct = self.substs.get(p.index as usize).map(|k| k.unpack());
        let ct = match opt_ct {
            Some(GenericArgKind::Const(ct)) => ct,
            Some(kind) => self.const_param_expected(p, source_ct, kind),
            None => self.const_param_out_of_range(p, source_ct),
        };

        self.shift_vars_through_binders(ct)
    }

    #[cold]
    #[inline(never)]
    fn const_param_expected(
        &self,
        p: ty::ParamConst,
        ct: ty::Const<'tcx>,
        kind: GenericArgKind<'tcx>,
    ) -> ! {
        bug!(
            "expected const for `{:?}` ({:?}/{}) but found {:?} when substituting substs={:?}",
            p,
            ct,
            p.index,
            kind,
            self.substs,
        )
    }

    #[cold]
    #[inline(never)]
    fn const_param_out_of_range(&self, p: ty::ParamConst, ct: ty::Const<'tcx>) -> ! {
        bug!(
            "const parameter `{:?}` ({:?}/{}) out of range when substituting substs={:?}",
            p,
            ct,
            p.index,
            self.substs,
        )
    }

    /// It is sometimes necessary to adjust the De Bruijn indices during substitution. This occurs
    /// when we are substituting a type with escaping bound vars into a context where we have
    /// passed through binders. That's quite a mouthful. Let's see an example:
    ///
    /// ```
    /// type Func<A> = fn(A);
    /// type MetaFunc = for<'a> fn(Func<&'a i32>);
    /// ```
    ///
    /// The type `MetaFunc`, when fully expanded, will be
    /// ```ignore (illustrative)
    /// for<'a> fn(fn(&'a i32))
    /// //      ^~ ^~ ^~~
    /// //      |  |  |
    /// //      |  |  DebruijnIndex of 2
    /// //      Binders
    /// ```
    /// Here the `'a` lifetime is bound in the outer function, but appears as an argument of the
    /// inner one. Therefore, that appearance will have a DebruijnIndex of 2, because we must skip
    /// over the inner binder (remember that we count De Bruijn indices from 1). However, in the
    /// definition of `MetaFunc`, the binder is not visible, so the type `&'a i32` will have a
    /// De Bruijn index of 1. It's only during the substitution that we can see we must increase the
    /// depth by 1 to account for the binder that we passed through.
    ///
    /// As a second example, consider this twist:
    ///
    /// ```
    /// type FuncTuple<A> = (A,fn(A));
    /// type MetaFuncTuple = for<'a> fn(FuncTuple<&'a i32>);
    /// ```
    ///
    /// Here the final type will be:
    /// ```ignore (illustrative)
    /// for<'a> fn((&'a i32, fn(&'a i32)))
    /// //          ^~~         ^~~
    /// //          |           |
    /// //   DebruijnIndex of 1 |
    /// //               DebruijnIndex of 2
    /// ```
    /// As indicated in the diagram, here the same type `&'a i32` is substituted once, but in the
    /// first case we do not increase the De Bruijn index and in the second case we do. The reason
    /// is that only in the second case have we passed through a fn binder.
    fn shift_vars_through_binders<T: TypeFoldable<'tcx>>(&self, val: T) -> T {
        debug!(
            "shift_vars(val={:?}, binders_passed={:?}, has_escaping_bound_vars={:?})",
            val,
            self.binders_passed,
            val.has_escaping_bound_vars()
        );

        if self.binders_passed == 0 || !val.has_escaping_bound_vars() {
            return val;
        }

        let result = ty::fold::shift_vars(TypeFolder::interner(self), val, self.binders_passed);
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

/// Stores the user-given substs to reach some fully qualified path
/// (e.g., `<T>::Item` or `<T as Trait>::Item`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct UserSubsts<'tcx> {
    /// The substitutions for the item as given by the user.
    pub substs: SubstsRef<'tcx>,

    /// The self type, in the case of a `<T>::Item` path (when applied
    /// to an inherent impl). See `UserSelfTy` below.
    pub user_self_ty: Option<UserSelfTy<'tcx>>,
}

/// Specifies the user-given self type. In the case of a path that
/// refers to a member in an inherent impl, this self type is
/// sometimes needed to constrain the type parameters on the impl. For
/// example, in this code:
///
/// ```ignore (illustrative)
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct UserSelfTy<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
}
