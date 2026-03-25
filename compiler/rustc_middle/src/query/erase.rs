//! To improve compile times and code size for the compiler itself, query
//! values are "erased" in some contexts (e.g. inside in-memory cache types),
//! to reduce the number of generic instantiations created during codegen.
//!
//! See <https://github.com/rust-lang/rust/pull/151715> for some bootstrap-time
//! and performance benchmarks.

use std::ffi::OsStr;
use std::intrinsics::transmute_unchecked;
use std::mem::MaybeUninit;

use rustc_ast::tokenstream::TokenStream;
use rustc_span::{ErrorGuaranteed, Spanned};

use crate::mir::mono::{MonoItem, NormalizationErrorInMono};
use crate::traits::solve;
use crate::ty::{self, Ty, TyCtxt};
use crate::{mir, traits};

/// Internal implementation detail of [`Erased`].
#[derive(Copy, Clone)]
pub struct ErasedData<Storage: Copy> {
    /// We use `MaybeUninit` here to make sure it's legal to store a transmuted
    /// value that isn't actually of type `Storage`.
    data: MaybeUninit<Storage>,
}

/// Trait for types that can be erased into [`Erased<Self>`].
///
/// Erasing and unerasing values is performed by [`erase_val`] and [`restore_val`].
///
/// FIXME: This whole trait could potentially be replaced by `T: Copy` and the
/// storage type `[u8; size_of::<T>()]` when support for that is more mature.
pub trait Erasable: Copy {
    /// Storage type to used for erased values of this type.
    /// Should be `[u8; N]`, where N is equal to `size_of::<Self>`.
    ///
    /// [`ErasedData`] wraps this storage type in `MaybeUninit` to ensure that
    /// transmutes to/from erased storage are well-defined.
    type Storage: Copy;
}

/// A value of `T` that has been "erased" into some opaque storage type.
///
/// This is helpful for reducing the number of concrete instantiations needed
/// during codegen when building the compiler.
///
/// Using an opaque type alias allows the type checker to enforce that
/// `Erased<T>` and `Erased<U>` are still distinct types, while allowing
/// monomorphization to see that they might actually use the same storage type.
pub type Erased<T: Erasable> = ErasedData<impl Copy>;

/// Erases a value of type `T` into `Erased<T>`.
///
/// `Erased<T>` and `Erased<U>` are type-checked as distinct types, but codegen
/// can see whether they actually have the same storage type.
///
/// FIXME: This might have soundness issues with erasable types that don't
/// implement the same auto-traits as `[u8; _]`; see
/// <https://github.com/rust-lang/rust/pull/151715#discussion_r2740113250>
#[inline(always)]
#[define_opaque(Erased)]
pub fn erase_val<T: Erasable>(value: T) -> Erased<T> {
    // Ensure the sizes match
    const {
        if size_of::<T>() != size_of::<T::Storage>() {
            panic!("size of T must match erased type <T as Erasable>::Storage")
        }
    };

    ErasedData::<<T as Erasable>::Storage> {
        // `transmute_unchecked` is needed here because it does not have `transmute`'s size check
        // (and thus allows to transmute between `T` and `MaybeUninit<T::Storage>`) (we do the size
        // check ourselves in the `const` block above).
        //
        // `transmute_copy` is also commonly used for this (and it would work here since
        // `Erasable: Copy`), but `transmute_unchecked` better explains the intent.
        //
        // SAFETY: It is safe to transmute to MaybeUninit for types with the same sizes.
        data: unsafe { transmute_unchecked::<T, MaybeUninit<T::Storage>>(value) },
    }
}

/// Restores an erased value to its real type.
///
/// This relies on the fact that `Erased<T>` and `Erased<U>` are type-checked
/// as distinct types, even if they use the same storage type.
#[inline(always)]
#[define_opaque(Erased)]
pub fn restore_val<T: Erasable>(erased_value: Erased<T>) -> T {
    let ErasedData { data }: ErasedData<<T as Erasable>::Storage> = erased_value;
    // See comment in `erase_val` for why we use `transmute_unchecked`.
    //
    // SAFETY: Due to the use of impl Trait in `Erased` the only way to safely create an instance
    // of `Erased` is to call `erase_val`, so we know that `erased_value.data` is a valid instance
    // of `T` of the right size.
    unsafe { transmute_unchecked::<MaybeUninit<T::Storage>, T>(data) }
}

// FIXME(#151565): Using `T: ?Sized` here should let us remove the separate
// impls for fat reference types.
impl<T> Erasable for &'_ T {
    type Storage = [u8; size_of::<&'static ()>()];
}

impl<T> Erasable for &'_ [T] {
    type Storage = [u8; size_of::<&'static [()]>()];
}

impl<T> Erasable for &'_ ty::List<T> {
    type Storage = [u8; size_of::<&'static ty::List<()>>()];
}

impl<T> Erasable for &'_ ty::ListWithCachedTypeInfo<T> {
    type Storage = [u8; size_of::<&'static ty::ListWithCachedTypeInfo<()>>()];
}

impl<T> Erasable for Result<&'_ T, traits::query::NoSolution> {
    type Storage = [u8; size_of::<Result<&'static (), traits::query::NoSolution>>()];
}

impl<T> Erasable for Result<&'_ T, ErrorGuaranteed> {
    type Storage = [u8; size_of::<Result<&'static (), ErrorGuaranteed>>()];
}

impl<T> Erasable for Result<&'_ T, traits::CodegenObligationError> {
    type Storage = [u8; size_of::<Result<&'static (), traits::CodegenObligationError>>()];
}

impl<T> Erasable for Result<&'_ T, &'_ ty::layout::FnAbiError<'_>> {
    type Storage = [u8; size_of::<Result<&'static (), &'static ty::layout::FnAbiError<'static>>>()];
}

impl<T> Erasable for Result<(&'_ T, crate::thir::ExprId), ErrorGuaranteed> {
    type Storage = [u8; size_of::<Result<(&'static (), crate::thir::ExprId), ErrorGuaranteed>>()];
}

impl Erasable for Result<Option<ty::Instance<'_>>, ErrorGuaranteed> {
    type Storage = [u8; size_of::<Result<Option<ty::Instance<'static>>, ErrorGuaranteed>>()];
}

impl Erasable for Result<Option<ty::EarlyBinder<'_, ty::Const<'_>>>, ErrorGuaranteed> {
    type Storage = [u8; size_of::<
        Result<Option<ty::EarlyBinder<'static, ty::Const<'static>>>, ErrorGuaranteed>,
    >()];
}

impl Erasable for Result<ty::GenericArg<'_>, traits::query::NoSolution> {
    type Storage = [u8; size_of::<Result<ty::GenericArg<'static>, traits::query::NoSolution>>()];
}

impl Erasable for Result<bool, &ty::layout::LayoutError<'_>> {
    type Storage = [u8; size_of::<Result<bool, &'static ty::layout::LayoutError<'static>>>()];
}

impl Erasable for Result<rustc_abi::TyAndLayout<'_, Ty<'_>>, &ty::layout::LayoutError<'_>> {
    type Storage = [u8; size_of::<
        Result<
            rustc_abi::TyAndLayout<'static, Ty<'static>>,
            &'static ty::layout::LayoutError<'static>,
        >,
    >()];
}

impl Erasable for Result<mir::ConstAlloc<'_>, mir::interpret::ErrorHandled> {
    type Storage =
        [u8; size_of::<Result<mir::ConstAlloc<'static>, mir::interpret::ErrorHandled>>()];
}

impl Erasable for Option<(mir::ConstValue, Ty<'_>)> {
    type Storage = [u8; size_of::<Option<(mir::ConstValue, Ty<'_>)>>()];
}

impl Erasable for Result<&'_ ty::List<Ty<'_>>, ty::util::AlwaysRequiresDrop> {
    type Storage =
        [u8; size_of::<Result<&'static ty::List<Ty<'static>>, ty::util::AlwaysRequiresDrop>>()];
}

impl Erasable
    for Result<(&'_ [Spanned<MonoItem<'_>>], &'_ [Spanned<MonoItem<'_>>]), NormalizationErrorInMono>
{
    type Storage = [u8; size_of::<
        Result<
            (&'static [Spanned<MonoItem<'static>>], &'static [Spanned<MonoItem<'static>>]),
            NormalizationErrorInMono,
        >,
    >()];
}

impl Erasable for Result<&'_ TokenStream, ()> {
    type Storage = [u8; size_of::<Result<&'static TokenStream, ()>>()];
}

impl<T> Erasable for Option<&'_ T> {
    type Storage = [u8; size_of::<Option<&'static ()>>()];
}

impl<T> Erasable for Option<&'_ [T]> {
    type Storage = [u8; size_of::<Option<&'static [()]>>()];
}

impl Erasable for Option<&'_ OsStr> {
    type Storage = [u8; size_of::<Option<&'static OsStr>>()];
}

impl Erasable for Option<ty::EarlyBinder<'_, Ty<'_>>> {
    type Storage = [u8; size_of::<Option<ty::EarlyBinder<'static, Ty<'static>>>>()];
}

impl Erasable for Option<ty::Value<'_>> {
    type Storage = [u8; size_of::<Option<ty::Value<'static>>>()];
}

impl<T: Erasable> Erasable for ty::EarlyBinder<'_, T> {
    type Storage = T::Storage;
}

impl Erasable for ty::Binder<'_, ty::FnSig<'_>> {
    type Storage = [u8; size_of::<ty::Binder<'static, ty::FnSig<'static>>>()];
}

impl Erasable for ty::Binder<'_, ty::CoroutineWitnessTypes<TyCtxt<'_>>> {
    type Storage =
        [u8; size_of::<ty::Binder<'static, ty::CoroutineWitnessTypes<TyCtxt<'static>>>>()];
}

impl<T0, T1> Erasable for (&'_ T0, &'_ T1) {
    type Storage = [u8; size_of::<(&'static (), &'static ())>()];
}

impl<T0> Erasable for (solve::QueryResult<'_>, &'_ T0) {
    type Storage = [u8; size_of::<(solve::QueryResult<'static>, &'static ())>()];
}

impl<T0> Erasable for (&'_ T0, Result<(), ErrorGuaranteed>) {
    type Storage = [u8; size_of::<(&'static (), Result<(), ErrorGuaranteed>)>()];
}

macro_rules! impl_erasable_for_simple_types {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Erasable for $ty {
                type Storage = [u8; size_of::<$ty>()];
            }
        )*
    }
}

// For concrete types with no lifetimes, the erased storage for `Foo` is
// `[u8; size_of::<Foo>()]`.
impl_erasable_for_simple_types! {
    // tidy-alphabetical-start
    (),
    Option<(rustc_span::def_id::DefId, rustc_session::config::EntryFnType)>,
    Option<rustc_abi::Align>,
    Option<rustc_ast::expand::allocator::AllocatorKind>,
    Option<rustc_data_structures::svh::Svh>,
    Option<rustc_hir::ConstStability>,
    Option<rustc_hir::CoroutineKind>,
    Option<rustc_hir::DefaultBodyStability>,
    Option<rustc_hir::Stability>,
    Option<rustc_middle::middle::stability::DeprecationEntry>,
    Option<rustc_middle::ty::AsyncDestructor>,
    Option<rustc_middle::ty::Destructor>,
    Option<rustc_middle::ty::IntrinsicDef>,
    Option<rustc_middle::ty::ScalarInt>,
    Option<rustc_span::Span>,
    Option<rustc_span::def_id::CrateNum>,
    Option<rustc_span::def_id::DefId>,
    Option<rustc_span::def_id::LocalDefId>,
    Option<rustc_target::spec::PanicStrategy>,
    Option<usize>,
    Result<(), ErrorGuaranteed>,
    Result<mir::ConstValue, mir::interpret::ErrorHandled>,
    Result<rustc_middle::traits::EvaluationResult, rustc_middle::traits::OverflowError>,
    Result<rustc_middle::ty::adjustment::CoerceUnsizedInfo, ErrorGuaranteed>,
    bool,
    rustc_data_structures::svh::Svh,
    rustc_hir::Constness,
    rustc_hir::Defaultness,
    rustc_hir::HirId,
    rustc_hir::OpaqueTyOrigin<rustc_hir::def_id::DefId>,
    rustc_hir::def::DefKind,
    rustc_hir::def_id::DefId,
    rustc_middle::middle::codegen_fn_attrs::SanitizerFnAttrs,
    rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault,
    rustc_middle::mir::ConstQualifs,
    rustc_middle::mir::ConstValue,
    rustc_middle::mir::interpret::AllocId,
    rustc_middle::ty::AnonConstKind,
    rustc_middle::ty::AssocItem,
    rustc_middle::ty::Asyncness,
    rustc_middle::ty::Visibility<rustc_span::def_id::DefId>,
    rustc_session::Limits,
    rustc_session::config::OptLevel,
    rustc_session::config::SymbolManglingVersion,
    rustc_session::cstore::CrateDepKind,
    rustc_span::ExpnId,
    rustc_span::Span,
    rustc_span::Symbol,
    rustc_target::spec::PanicStrategy,
    usize,
    // tidy-alphabetical-end
}

macro_rules! impl_erasable_for_single_lifetime_types {
    ($($($fake_path:ident)::+),+ $(,)?) => {
        $(
            impl Erasable for $($fake_path)::+<'_> {
                type Storage = [u8; size_of::<$($fake_path)::+<'static>>()];
            }
        )*
    }
}

// For types containing a single lifetime and no other generics, e.g.
// `Foo<'tcx>`, the erased storage is `[u8; size_of::<Foo<'static>>()]`.
impl_erasable_for_single_lifetime_types! {
    // tidy-alphabetical-start
    rustc_hir::MaybeOwner,
    rustc_middle::mir::interpret::EvalStaticInitializerRawResult,
    rustc_middle::mir::interpret::EvalToValTreeResult,
    rustc_middle::mir::mono::MonoItemPartitions,
    rustc_middle::traits::query::MethodAutoderefStepsResult,
    rustc_middle::ty::AdtDef,
    rustc_middle::ty::ClosureTypeInfo,
    rustc_middle::ty::Const,
    rustc_middle::ty::ConstConditions,
    rustc_middle::ty::GenericPredicates,
    rustc_middle::ty::ImplTraitHeader,
    rustc_middle::ty::ParamEnv,
    rustc_middle::ty::SymbolName,
    rustc_middle::ty::Ty,
    rustc_middle::ty::TypingEnv,
    rustc_middle::ty::inhabitedness::InhabitedPredicate,
    // tidy-alphabetical-end
}
