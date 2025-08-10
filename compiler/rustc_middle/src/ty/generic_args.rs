// Generic arguments.

use core::intrinsics;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr::NonNull;

use rustc_data_structures::intern::Interned;
use rustc_errors::{DiagArgValue, IntoDiagArg};
use rustc_hir::def_id::DefId;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, extension};
use rustc_serialize::{Decodable, Encodable};
use rustc_type_ir::WithCachedTypeInfo;
use rustc_type_ir::walk::TypeWalker;
use smallvec::SmallVec;

use crate::ty::codec::{TyDecoder, TyEncoder};
use crate::ty::{
    self, ClosureArgs, CoroutineArgs, CoroutineClosureArgs, FallibleTypeFolder, InlineConstArgs,
    Lift, List, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor, VisitorResult,
    walk_visitable_list,
};

pub type GenericArgKind<'tcx> = rustc_type_ir::GenericArgKind<TyCtxt<'tcx>>;
pub type TermKind<'tcx> = rustc_type_ir::TermKind<TyCtxt<'tcx>>;

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
    ptr: NonNull<()>,
    marker: PhantomData<(Ty<'tcx>, ty::Region<'tcx>, ty::Const<'tcx>)>,
}

impl<'tcx> rustc_type_ir::inherent::GenericArg<TyCtxt<'tcx>> for GenericArg<'tcx> {}

impl<'tcx> rustc_type_ir::inherent::GenericArgs<TyCtxt<'tcx>> for ty::GenericArgsRef<'tcx> {
    fn rebase_onto(
        self,
        tcx: TyCtxt<'tcx>,
        source_ancestor: DefId,
        target_args: GenericArgsRef<'tcx>,
    ) -> GenericArgsRef<'tcx> {
        self.rebase_onto(tcx, source_ancestor, target_args)
    }

    fn type_at(self, i: usize) -> Ty<'tcx> {
        self.type_at(i)
    }

    fn region_at(self, i: usize) -> ty::Region<'tcx> {
        self.region_at(i)
    }

    fn const_at(self, i: usize) -> ty::Const<'tcx> {
        self.const_at(i)
    }

    fn identity_for_item(tcx: TyCtxt<'tcx>, def_id: DefId) -> ty::GenericArgsRef<'tcx> {
        GenericArgs::identity_for_item(tcx, def_id)
    }

    fn extend_with_error(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        original_args: &[ty::GenericArg<'tcx>],
    ) -> ty::GenericArgsRef<'tcx> {
        ty::GenericArgs::extend_with_error(tcx, def_id, original_args)
    }

    fn split_closure_args(self) -> ty::ClosureArgsParts<TyCtxt<'tcx>> {
        match self[..] {
            [ref parent_args @ .., closure_kind_ty, closure_sig_as_fn_ptr_ty, tupled_upvars_ty] => {
                ty::ClosureArgsParts {
                    parent_args,
                    closure_kind_ty: closure_kind_ty.expect_ty(),
                    closure_sig_as_fn_ptr_ty: closure_sig_as_fn_ptr_ty.expect_ty(),
                    tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                }
            }
            _ => bug!("closure args missing synthetics"),
        }
    }

    fn split_coroutine_closure_args(self) -> ty::CoroutineClosureArgsParts<TyCtxt<'tcx>> {
        match self[..] {
            [
                ref parent_args @ ..,
                closure_kind_ty,
                signature_parts_ty,
                tupled_upvars_ty,
                coroutine_captures_by_ref_ty,
            ] => ty::CoroutineClosureArgsParts {
                parent_args,
                closure_kind_ty: closure_kind_ty.expect_ty(),
                signature_parts_ty: signature_parts_ty.expect_ty(),
                tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                coroutine_captures_by_ref_ty: coroutine_captures_by_ref_ty.expect_ty(),
            },
            _ => bug!("closure args missing synthetics"),
        }
    }

    fn split_coroutine_args(self) -> ty::CoroutineArgsParts<TyCtxt<'tcx>> {
        match self[..] {
            [ref parent_args @ .., kind_ty, resume_ty, yield_ty, return_ty, tupled_upvars_ty] => {
                ty::CoroutineArgsParts {
                    parent_args,
                    kind_ty: kind_ty.expect_ty(),
                    resume_ty: resume_ty.expect_ty(),
                    yield_ty: yield_ty.expect_ty(),
                    return_ty: return_ty.expect_ty(),
                    tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                }
            }
            _ => bug!("coroutine args missing synthetics"),
        }
    }
}

impl<'tcx> rustc_type_ir::inherent::IntoKind for GenericArg<'tcx> {
    type Kind = GenericArgKind<'tcx>;

    fn kind(self) -> Self::Kind {
        self.kind()
    }
}

unsafe impl<'tcx> rustc_data_structures::sync::DynSend for GenericArg<'tcx> where
    &'tcx (Ty<'tcx>, ty::Region<'tcx>, ty::Const<'tcx>): rustc_data_structures::sync::DynSend
{
}
unsafe impl<'tcx> rustc_data_structures::sync::DynSync for GenericArg<'tcx> where
    &'tcx (Ty<'tcx>, ty::Region<'tcx>, ty::Const<'tcx>): rustc_data_structures::sync::DynSync
{
}
unsafe impl<'tcx> Send for GenericArg<'tcx> where
    &'tcx (Ty<'tcx>, ty::Region<'tcx>, ty::Const<'tcx>): Send
{
}
unsafe impl<'tcx> Sync for GenericArg<'tcx> where
    &'tcx (Ty<'tcx>, ty::Region<'tcx>, ty::Const<'tcx>): Sync
{
}

impl<'tcx> IntoDiagArg for GenericArg<'tcx> {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_string().into_diag_arg(&mut None)
    }
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const REGION_TAG: usize = 0b01;
const CONST_TAG: usize = 0b10;

#[extension(trait GenericArgPackExt<'tcx>)]
impl<'tcx> GenericArgKind<'tcx> {
    #[inline]
    fn pack(self) -> GenericArg<'tcx> {
        let (tag, ptr) = match self {
            GenericArgKind::Lifetime(lt) => {
                // Ensure we can use the tag bits.
                assert_eq!(align_of_val(&*lt.0.0) & TAG_MASK, 0);
                (REGION_TAG, NonNull::from(lt.0.0).cast())
            }
            GenericArgKind::Type(ty) => {
                // Ensure we can use the tag bits.
                assert_eq!(align_of_val(&*ty.0.0) & TAG_MASK, 0);
                (TYPE_TAG, NonNull::from(ty.0.0).cast())
            }
            GenericArgKind::Const(ct) => {
                // Ensure we can use the tag bits.
                assert_eq!(align_of_val(&*ct.0.0) & TAG_MASK, 0);
                (CONST_TAG, NonNull::from(ct.0.0).cast())
            }
        };

        GenericArg { ptr: ptr.map_addr(|addr| addr | tag), marker: PhantomData }
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
        match value.kind() {
            ty::TermKind::Ty(t) => t.into(),
            ty::TermKind::Const(c) => c.into(),
        }
    }
}

impl<'tcx> GenericArg<'tcx> {
    #[inline]
    pub fn kind(self) -> GenericArgKind<'tcx> {
        let ptr =
            unsafe { self.ptr.map_addr(|addr| NonZero::new_unchecked(addr.get() & !TAG_MASK)) };
        // SAFETY: use of `Interned::new_unchecked` here is ok because these
        // pointers were originally created from `Interned` types in `pack()`,
        // and this is just going in the other direction.
        unsafe {
            match self.ptr.addr().get() & TAG_MASK {
                REGION_TAG => GenericArgKind::Lifetime(ty::Region(Interned::new_unchecked(
                    ptr.cast::<ty::RegionKind<'tcx>>().as_ref(),
                ))),
                TYPE_TAG => GenericArgKind::Type(Ty(Interned::new_unchecked(
                    ptr.cast::<WithCachedTypeInfo<ty::TyKind<'tcx>>>().as_ref(),
                ))),
                CONST_TAG => GenericArgKind::Const(ty::Const(Interned::new_unchecked(
                    ptr.cast::<WithCachedTypeInfo<ty::ConstKind<'tcx>>>().as_ref(),
                ))),
                _ => intrinsics::unreachable(),
            }
        }
    }

    #[inline]
    pub fn as_region(self) -> Option<ty::Region<'tcx>> {
        match self.kind() {
            GenericArgKind::Lifetime(re) => Some(re),
            _ => None,
        }
    }

    #[inline]
    pub fn as_type(self) -> Option<Ty<'tcx>> {
        match self.kind() {
            GenericArgKind::Type(ty) => Some(ty),
            _ => None,
        }
    }

    #[inline]
    pub fn as_const(self) -> Option<ty::Const<'tcx>> {
        match self.kind() {
            GenericArgKind::Const(ct) => Some(ct),
            _ => None,
        }
    }

    #[inline]
    pub fn as_term(self) -> Option<ty::Term<'tcx>> {
        match self.kind() {
            GenericArgKind::Lifetime(_) => None,
            GenericArgKind::Type(ty) => Some(ty.into()),
            GenericArgKind::Const(ct) => Some(ct.into()),
        }
    }

    /// Unpack the `GenericArg` as a region when it is known certainly to be a region.
    pub fn expect_region(self) -> ty::Region<'tcx> {
        self.as_region().unwrap_or_else(|| bug!("expected a region, but found another kind"))
    }

    /// Unpack the `GenericArg` as a type when it is known certainly to be a type.
    /// This is true in cases where `GenericArgs` is used in places where the kinds are known
    /// to be limited (e.g. in tuples, where the only parameters are type parameters).
    pub fn expect_ty(self) -> Ty<'tcx> {
        self.as_type().unwrap_or_else(|| bug!("expected a type, but found another kind"))
    }

    /// Unpack the `GenericArg` as a const when it is known certainly to be a const.
    pub fn expect_const(self) -> ty::Const<'tcx> {
        self.as_const().unwrap_or_else(|| bug!("expected a const, but found another kind"))
    }

    pub fn is_non_region_infer(self) -> bool {
        match self.kind() {
            GenericArgKind::Lifetime(_) => false,
            // FIXME: This shouldn't return numerical/float.
            GenericArgKind::Type(ty) => ty.is_ty_or_numeric_infer(),
            GenericArgKind::Const(ct) => ct.is_ct_infer(),
        }
    }

    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```text
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    pub fn walk(self) -> TypeWalker<TyCtxt<'tcx>> {
        TypeWalker::new(self)
    }
}

impl<'a, 'tcx> Lift<TyCtxt<'tcx>> for GenericArg<'a> {
    type Lifted = GenericArg<'tcx>;

    fn lift_to_interner(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
        match self.kind() {
            GenericArgKind::Lifetime(lt) => tcx.lift(lt).map(|lt| lt.into()),
            GenericArgKind::Type(ty) => tcx.lift(ty).map(|ty| ty.into()),
            GenericArgKind::Const(ct) => tcx.lift(ct).map(|ct| ct.into()),
        }
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for GenericArg<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        match self.kind() {
            GenericArgKind::Lifetime(lt) => lt.try_fold_with(folder).map(Into::into),
            GenericArgKind::Type(ty) => ty.try_fold_with(folder).map(Into::into),
            GenericArgKind::Const(ct) => ct.try_fold_with(folder).map(Into::into),
        }
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        match self.kind() {
            GenericArgKind::Lifetime(lt) => lt.fold_with(folder).into(),
            GenericArgKind::Type(ty) => ty.fold_with(folder).into(),
            GenericArgKind::Const(ct) => ct.fold_with(folder).into(),
        }
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for GenericArg<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        match self.kind() {
            GenericArgKind::Lifetime(lt) => lt.visit_with(visitor),
            GenericArgKind::Type(ty) => ty.visit_with(visitor),
            GenericArgKind::Const(ct) => ct.visit_with(visitor),
        }
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for GenericArg<'tcx> {
    fn encode(&self, e: &mut E) {
        self.kind().encode(e)
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for GenericArg<'tcx> {
    fn decode(d: &mut D) -> GenericArg<'tcx> {
        GenericArgKind::decode(d).pack()
    }
}

/// List of generic arguments that are gonna be used to replace generic parameters.
pub type GenericArgs<'tcx> = List<GenericArg<'tcx>>;

pub type GenericArgsRef<'tcx> = &'tcx GenericArgs<'tcx>;

impl<'tcx> GenericArgs<'tcx> {
    /// Converts generic args to a type list.
    ///
    /// # Panics
    ///
    /// If any of the generic arguments are not types.
    pub fn into_type_list(&self, tcx: TyCtxt<'tcx>) -> &'tcx List<Ty<'tcx>> {
        tcx.mk_type_list_from_iter(self.iter().map(|arg| match arg.kind() {
            GenericArgKind::Type(ty) => ty,
            _ => bug!("`into_type_list` called on generic arg with non-types"),
        }))
    }

    /// Interpret these generic args as the args of a closure type.
    /// Closure args have a particular structure controlled by the
    /// compiler that encodes information like the signature and closure kind;
    /// see `ty::ClosureArgs` struct for more comments.
    pub fn as_closure(&'tcx self) -> ClosureArgs<TyCtxt<'tcx>> {
        ClosureArgs { args: self }
    }

    /// Interpret these generic args as the args of a coroutine-closure type.
    /// Coroutine-closure args have a particular structure controlled by the
    /// compiler that encodes information like the signature and closure kind;
    /// see `ty::CoroutineClosureArgs` struct for more comments.
    pub fn as_coroutine_closure(&'tcx self) -> CoroutineClosureArgs<TyCtxt<'tcx>> {
        CoroutineClosureArgs { args: self }
    }

    /// Interpret these generic args as the args of a coroutine type.
    /// Coroutine args have a particular structure controlled by the
    /// compiler that encodes information like the signature and coroutine kind;
    /// see `ty::CoroutineArgs` struct for more comments.
    pub fn as_coroutine(&'tcx self) -> CoroutineArgs<TyCtxt<'tcx>> {
        CoroutineArgs { args: self }
    }

    /// Interpret these generic args as the args of an inline const.
    /// Inline const args have a particular structure controlled by the
    /// compiler that encodes information like the inferred type;
    /// see `ty::InlineConstArgs` struct for more comments.
    pub fn as_inline_const(&'tcx self) -> InlineConstArgs<'tcx> {
        InlineConstArgs { args: self }
    }

    /// Creates a [`GenericArgs`] that maps each generic parameter to itself.
    pub fn identity_for_item(tcx: TyCtxt<'tcx>, def_id: impl Into<DefId>) -> GenericArgsRef<'tcx> {
        Self::for_item(tcx, def_id.into(), |param, _| tcx.mk_param_from_def(param))
    }

    /// Creates a [`GenericArgs`] for generic parameter definitions,
    /// by calling closures to obtain each kind.
    /// The closures get to observe the [`GenericArgs`] as they're
    /// being built, which can be used to correctly
    /// replace defaults of generic parameters.
    pub fn for_item<F>(tcx: TyCtxt<'tcx>, def_id: DefId, mut mk_kind: F) -> GenericArgsRef<'tcx>
    where
        F: FnMut(&ty::GenericParamDef, &[GenericArg<'tcx>]) -> GenericArg<'tcx>,
    {
        let defs = tcx.generics_of(def_id);
        let count = defs.count();
        let mut args = SmallVec::with_capacity(count);
        Self::fill_item(&mut args, tcx, defs, &mut mk_kind);
        tcx.mk_args(&args)
    }

    pub fn extend_to<F>(
        &self,
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        mut mk_kind: F,
    ) -> GenericArgsRef<'tcx>
    where
        F: FnMut(&ty::GenericParamDef, &[GenericArg<'tcx>]) -> GenericArg<'tcx>,
    {
        Self::for_item(tcx, def_id, |param, args| {
            self.get(param.index as usize).cloned().unwrap_or_else(|| mk_kind(param, args))
        })
    }

    pub fn fill_item<F>(
        args: &mut SmallVec<[GenericArg<'tcx>; 8]>,
        tcx: TyCtxt<'tcx>,
        defs: &ty::Generics,
        mk_kind: &mut F,
    ) where
        F: FnMut(&ty::GenericParamDef, &[GenericArg<'tcx>]) -> GenericArg<'tcx>,
    {
        if let Some(def_id) = defs.parent {
            let parent_defs = tcx.generics_of(def_id);
            Self::fill_item(args, tcx, parent_defs, mk_kind);
        }
        Self::fill_single(args, defs, mk_kind)
    }

    pub fn fill_single<F>(
        args: &mut SmallVec<[GenericArg<'tcx>; 8]>,
        defs: &ty::Generics,
        mk_kind: &mut F,
    ) where
        F: FnMut(&ty::GenericParamDef, &[GenericArg<'tcx>]) -> GenericArg<'tcx>,
    {
        args.reserve(defs.own_params.len());
        for param in &defs.own_params {
            let kind = mk_kind(param, args);
            assert_eq!(param.index as usize, args.len(), "{args:#?}, {defs:#?}");
            args.push(kind);
        }
    }

    // Extend an `original_args` list to the full number of args expected by `def_id`,
    // filling in the missing parameters with error ty/ct or 'static regions.
    pub fn extend_with_error(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        original_args: &[GenericArg<'tcx>],
    ) -> GenericArgsRef<'tcx> {
        ty::GenericArgs::for_item(tcx, def_id, |def, _| {
            if let Some(arg) = original_args.get(def.index as usize) {
                *arg
            } else {
                def.to_error(tcx)
            }
        })
    }

    #[inline]
    pub fn types(&self) -> impl DoubleEndedIterator<Item = Ty<'tcx>> {
        self.iter().filter_map(|k| k.as_type())
    }

    #[inline]
    pub fn regions(&self) -> impl DoubleEndedIterator<Item = ty::Region<'tcx>> {
        self.iter().filter_map(|k| k.as_region())
    }

    #[inline]
    pub fn consts(&self) -> impl DoubleEndedIterator<Item = ty::Const<'tcx>> {
        self.iter().filter_map(|k| k.as_const())
    }

    /// Returns generic arguments that are not lifetimes.
    #[inline]
    pub fn non_erasable_generics(&self) -> impl DoubleEndedIterator<Item = GenericArgKind<'tcx>> {
        self.iter().filter_map(|arg| match arg.kind() {
            ty::GenericArgKind::Lifetime(_) => None,
            generic => Some(generic),
        })
    }

    #[inline]
    #[track_caller]
    pub fn type_at(&self, i: usize) -> Ty<'tcx> {
        self[i].as_type().unwrap_or_else(
            #[track_caller]
            || bug!("expected type for param #{} in {:?}", i, self),
        )
    }

    #[inline]
    #[track_caller]
    pub fn region_at(&self, i: usize) -> ty::Region<'tcx> {
        self[i].as_region().unwrap_or_else(
            #[track_caller]
            || bug!("expected region for param #{} in {:?}", i, self),
        )
    }

    #[inline]
    #[track_caller]
    pub fn const_at(&self, i: usize) -> ty::Const<'tcx> {
        self[i].as_const().unwrap_or_else(
            #[track_caller]
            || bug!("expected const for param #{} in {:?}", i, self),
        )
    }

    #[inline]
    #[track_caller]
    pub fn type_for_def(&self, def: &ty::GenericParamDef) -> GenericArg<'tcx> {
        self.type_at(def.index as usize).into()
    }

    /// Transform from generic args for a child of `source_ancestor`
    /// (e.g., a trait or impl) to args for the same child
    /// in a different item, with `target_args` as the base for
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
    /// * If `self` is `[Self, S, T]`: the identity args of `f` in the trait.
    /// * If `source_ancestor` is the def_id of the trait.
    /// * If `target_args` is `[U]`, the args for the impl.
    /// * Then we will return `[U, T]`, the arg for `f` in the impl that
    ///   are needed for it to match the trait.
    pub fn rebase_onto(
        &self,
        tcx: TyCtxt<'tcx>,
        source_ancestor: DefId,
        target_args: GenericArgsRef<'tcx>,
    ) -> GenericArgsRef<'tcx> {
        let defs = tcx.generics_of(source_ancestor);
        tcx.mk_args_from_iter(target_args.iter().chain(self.iter().skip(defs.count())))
    }

    /// Truncates this list of generic args to have at most the number of args in `generics`.
    ///
    /// You might be looking for [`TraitRef::from_assoc`](super::TraitRef::from_assoc).
    pub fn truncate_to(&self, tcx: TyCtxt<'tcx>, generics: &ty::Generics) -> GenericArgsRef<'tcx> {
        tcx.mk_args(&self[..generics.count()])
    }

    pub fn print_as_list(&self) -> String {
        let v = self.iter().map(|arg| arg.to_string()).collect::<Vec<_>>();
        format!("[{}]", v.join(", "))
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for GenericArgsRef<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        // This code is hot enough that it's worth specializing for the most
        // common length lists, to avoid the overhead of `SmallVec` creation.
        // The match arms are in order of frequency. The 1, 2, and 0 cases are
        // typically hit in 90--99.99% of cases. When folding doesn't change
        // the args, it's faster to reuse the existing args rather than
        // calling `mk_args`.
        match self.len() {
            1 => {
                let param0 = self[0].try_fold_with(folder)?;
                if param0 == self[0] { Ok(self) } else { Ok(folder.cx().mk_args(&[param0])) }
            }
            2 => {
                let param0 = self[0].try_fold_with(folder)?;
                let param1 = self[1].try_fold_with(folder)?;
                if param0 == self[0] && param1 == self[1] {
                    Ok(self)
                } else {
                    Ok(folder.cx().mk_args(&[param0, param1]))
                }
            }
            0 => Ok(self),
            _ => ty::util::try_fold_list(self, folder, |tcx, v| tcx.mk_args(v)),
        }
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        // See justification for this behavior in `try_fold_with`.
        match self.len() {
            1 => {
                let param0 = self[0].fold_with(folder);
                if param0 == self[0] { self } else { folder.cx().mk_args(&[param0]) }
            }
            2 => {
                let param0 = self[0].fold_with(folder);
                let param1 = self[1].fold_with(folder);
                if param0 == self[0] && param1 == self[1] {
                    self
                } else {
                    folder.cx().mk_args(&[param0, param1])
                }
            }
            0 => self,
            _ => ty::util::fold_list(self, folder, |tcx, v| tcx.mk_args(v)),
        }
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for &'tcx ty::List<Ty<'tcx>> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        // This code is fairly hot, though not as hot as `GenericArgsRef`.
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
                    Ok(folder.cx().mk_type_list(&[param0, param1]))
                }
            }
            _ => ty::util::try_fold_list(self, folder, |tcx, v| tcx.mk_type_list(v)),
        }
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        // See comment justifying behavior in `try_fold_with`.
        match self.len() {
            2 => {
                let param0 = self[0].fold_with(folder);
                let param1 = self[1].fold_with(folder);
                if param0 == self[0] && param1 == self[1] {
                    self
                } else {
                    folder.cx().mk_type_list(&[param0, param1])
                }
            }
            _ => ty::util::fold_list(self, folder, |tcx, v| tcx.mk_type_list(v)),
        }
    }
}

impl<'tcx, T: TypeVisitable<TyCtxt<'tcx>>> TypeVisitable<TyCtxt<'tcx>> for &'tcx ty::List<T> {
    #[inline]
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        walk_visitable_list!(visitor, self.iter());
        V::Result::output()
    }
}

/// Stores the user-given args to reach some fully qualified path
/// (e.g., `<T>::Item` or `<T as Trait>::Item`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct UserArgs<'tcx> {
    /// The args for the item as given by the user.
    pub args: GenericArgsRef<'tcx>,

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
/// the impl (with the args from `UserArgs`) and apply those to
/// the self type, giving `Foo<?A>`. Finally, we unify that with
/// the self type here, which contains `?A` to be `&'static u32`
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct UserSelfTy<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
}
