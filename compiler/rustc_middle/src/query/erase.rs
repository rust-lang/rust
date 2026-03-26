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
use rustc_data_structures::steal::Steal;
use rustc_span::{ErrorGuaranteed, Spanned};

use crate::mir::mono::{MonoItem, NormalizationErrorInMono};
use crate::ty::{self, Ty, TyCtxt};
use crate::{mir, thir, traits};

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
    type Storage = [u8; size_of::<&'_ ()>()];
}

impl<T> Erasable for &'_ [T] {
    type Storage = [u8; size_of::<&'_ [()]>()];
}

impl<T> Erasable for &'_ ty::List<T> {
    type Storage = [u8; size_of::<&'_ ty::List<()>>()];
}

impl<T> Erasable for &'_ ty::ListWithCachedTypeInfo<T> {
    type Storage = [u8; size_of::<&'_ ty::ListWithCachedTypeInfo<()>>()];
}

impl<T> Erasable for Result<&'_ T, traits::query::NoSolution> {
    type Storage = [u8; size_of::<Result<&'_ (), traits::query::NoSolution>>()];
}

impl<T> Erasable for Result<&'_ T, ErrorGuaranteed> {
    type Storage = [u8; size_of::<Result<&'_ (), ErrorGuaranteed>>()];
}

impl<T> Erasable for Option<&'_ T> {
    type Storage = [u8; size_of::<Option<&'_ ()>>()];
}

impl<T: Erasable> Erasable for ty::EarlyBinder<'_, T> {
    type Storage = T::Storage;
}

impl<T0, T1> Erasable for (&'_ T0, &'_ T1) {
    type Storage = [u8; size_of::<(&'_ (), &'_ ())>()];
}

macro_rules! impl_erasable_for_types_with_no_type_params {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Erasable for $ty {
                type Storage = [u8; size_of::<$ty>()];
            }
        )*
    }
}

// For types with no type parameters the erased storage for `Foo` is
// `[u8; size_of::<Foo>()]`. ('_ lifetimes are allowed.)
impl_erasable_for_types_with_no_type_params! {
    // tidy-alphabetical-start
    (&'_ ty::CrateInherentImpls, Result<(), ErrorGuaranteed>),
    (),
    (traits::solve::QueryResult<'_>, &'_ traits::solve::inspect::Probe<TyCtxt<'_>>),
    Option<&'_ OsStr>,
    Option<&'_ [rustc_hir::PreciseCapturingArgKind<rustc_span::Symbol, rustc_span::Symbol>]>,
    Option<(mir::ConstValue, Ty<'_>)>,
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
    Option<ty::EarlyBinder<'_, Ty<'_>>>,
    Option<ty::Value<'_>>,
    Option<usize>,
    Result<&'_ TokenStream, ()>,
    Result<&'_ rustc_target::callconv::FnAbi<'_, Ty<'_>>, &'_ ty::layout::FnAbiError<'_>>,
    Result<&'_ traits::ImplSource<'_, ()>, traits::CodegenObligationError>,
    Result<&'_ ty::List<Ty<'_>>, ty::util::AlwaysRequiresDrop>,
    Result<(&'_ Steal<thir::Thir<'_>>, thir::ExprId), ErrorGuaranteed>,
    Result<(&'_ [Spanned<MonoItem<'_>>], &'_ [Spanned<MonoItem<'_>>]), NormalizationErrorInMono>,
    Result<(), ErrorGuaranteed>,
    Result<Option<ty::EarlyBinder<'_, ty::Const<'_>>>, ErrorGuaranteed>,
    Result<Option<ty::Instance<'_>>, ErrorGuaranteed>,
    Result<bool, &ty::layout::LayoutError<'_>>,
    Result<mir::ConstAlloc<'_>, mir::interpret::ErrorHandled>,
    Result<mir::ConstValue, mir::interpret::ErrorHandled>,
    Result<rustc_abi::TyAndLayout<'_, Ty<'_>>, &ty::layout::LayoutError<'_>>,
    Result<rustc_middle::traits::EvaluationResult, rustc_middle::traits::OverflowError>,
    Result<rustc_middle::ty::adjustment::CoerceUnsizedInfo, ErrorGuaranteed>,
    Result<ty::GenericArg<'_>, traits::query::NoSolution>,
    Ty<'_>,
    bool,
    rustc_data_structures::svh::Svh,
    rustc_hir::Constness,
    rustc_hir::Defaultness,
    rustc_hir::HirId,
    rustc_hir::MaybeOwner<'_>,
    rustc_hir::OpaqueTyOrigin<rustc_hir::def_id::DefId>,
    rustc_hir::def::DefKind,
    rustc_hir::def_id::DefId,
    rustc_middle::middle::codegen_fn_attrs::SanitizerFnAttrs,
    rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault,
    rustc_middle::mir::ConstQualifs,
    rustc_middle::mir::ConstValue,
    rustc_middle::mir::interpret::AllocId,
    rustc_middle::mir::interpret::EvalStaticInitializerRawResult<'_>,
    rustc_middle::mir::interpret::EvalToValTreeResult<'_>,
    rustc_middle::mir::mono::MonoItemPartitions<'_>,
    rustc_middle::traits::query::MethodAutoderefStepsResult<'_>,
    rustc_middle::ty::AdtDef<'_>,
    rustc_middle::ty::AnonConstKind,
    rustc_middle::ty::AssocItem,
    rustc_middle::ty::Asyncness,
    rustc_middle::ty::Binder<'_, ty::CoroutineWitnessTypes<TyCtxt<'_>>>,
    rustc_middle::ty::Binder<'_, ty::FnSig<'_>>,
    rustc_middle::ty::ClosureTypeInfo<'_>,
    rustc_middle::ty::Const<'_>,
    rustc_middle::ty::ConstConditions<'_>,
    rustc_middle::ty::GenericPredicates<'_>,
    rustc_middle::ty::ImplTraitHeader<'_>,
    rustc_middle::ty::ParamEnv<'_>,
    rustc_middle::ty::SymbolName<'_>,
    rustc_middle::ty::TypingEnv<'_>,
    rustc_middle::ty::Visibility<rustc_span::def_id::DefId>,
    rustc_middle::ty::inhabitedness::InhabitedPredicate<'_>,
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
