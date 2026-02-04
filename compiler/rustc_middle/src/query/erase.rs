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
use rustc_span::ErrorGuaranteed;
use rustc_span::source_map::Spanned;

use crate::mir::interpret::EvalToValTreeResult;
use crate::mir::mono::{MonoItem, NormalizationErrorInMono};
use crate::query::plumbing::CyclePlaceholder;
use crate::traits::solve;
use crate::ty::adjustment::CoerceUnsizedInfo;
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

impl Erasable for &'_ OsStr {
    type Storage = [u8; size_of::<&'static OsStr>()];
}

impl<T> Erasable for &'_ ty::List<T> {
    type Storage = [u8; size_of::<&'static ty::List<()>>()];
}

impl<T> Erasable for &'_ ty::ListWithCachedTypeInfo<T> {
    type Storage = [u8; size_of::<&'static ty::ListWithCachedTypeInfo<()>>()];
}

impl<I: rustc_index::Idx, T> Erasable for &'_ rustc_index::IndexSlice<I, T> {
    type Storage = [u8; size_of::<&'static rustc_index::IndexSlice<u32, ()>>()];
}

impl<T> Erasable for Result<&'_ T, traits::query::NoSolution> {
    type Storage = [u8; size_of::<Result<&'static (), traits::query::NoSolution>>()];
}

impl<T> Erasable for Result<&'_ [T], traits::query::NoSolution> {
    type Storage = [u8; size_of::<Result<&'static [()], traits::query::NoSolution>>()];
}

impl<T> Erasable for Result<&'_ T, rustc_errors::ErrorGuaranteed> {
    type Storage = [u8; size_of::<Result<&'static (), rustc_errors::ErrorGuaranteed>>()];
}

impl<T> Erasable for Result<&'_ [T], rustc_errors::ErrorGuaranteed> {
    type Storage = [u8; size_of::<Result<&'static [()], rustc_errors::ErrorGuaranteed>>()];
}

impl<T> Erasable for Result<&'_ T, traits::CodegenObligationError> {
    type Storage = [u8; size_of::<Result<&'static (), traits::CodegenObligationError>>()];
}

impl<T> Erasable for Result<&'_ T, &'_ ty::layout::FnAbiError<'_>> {
    type Storage = [u8; size_of::<Result<&'static (), &'static ty::layout::FnAbiError<'static>>>()];
}

impl<T> Erasable for Result<(&'_ T, crate::thir::ExprId), rustc_errors::ErrorGuaranteed> {
    type Storage = [u8; size_of::<
        Result<(&'static (), crate::thir::ExprId), rustc_errors::ErrorGuaranteed>,
    >()];
}

impl Erasable for Result<Option<ty::Instance<'_>>, rustc_errors::ErrorGuaranteed> {
    type Storage =
        [u8; size_of::<Result<Option<ty::Instance<'static>>, rustc_errors::ErrorGuaranteed>>()];
}

impl Erasable for Result<CoerceUnsizedInfo, rustc_errors::ErrorGuaranteed> {
    type Storage = [u8; size_of::<Result<CoerceUnsizedInfo, rustc_errors::ErrorGuaranteed>>()];
}

impl Erasable
    for Result<Option<ty::EarlyBinder<'_, ty::Const<'_>>>, rustc_errors::ErrorGuaranteed>
{
    type Storage = [u8; size_of::<
        Result<Option<ty::EarlyBinder<'static, ty::Const<'static>>>, rustc_errors::ErrorGuaranteed>,
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

impl Erasable for Result<mir::ConstValue, mir::interpret::ErrorHandled> {
    type Storage = [u8; size_of::<Result<mir::ConstValue, mir::interpret::ErrorHandled>>()];
}

impl Erasable for Option<(mir::ConstValue, Ty<'_>)> {
    type Storage = [u8; size_of::<Option<(mir::ConstValue, Ty<'_>)>>()];
}

impl Erasable for EvalToValTreeResult<'_> {
    type Storage = [u8; size_of::<EvalToValTreeResult<'static>>()];
}

impl Erasable for Result<&'_ ty::List<Ty<'_>>, ty::util::AlwaysRequiresDrop> {
    type Storage =
        [u8; size_of::<Result<&'static ty::List<Ty<'static>>, ty::util::AlwaysRequiresDrop>>()];
}

impl Erasable for Result<ty::EarlyBinder<'_, Ty<'_>>, CyclePlaceholder> {
    type Storage = [u8; size_of::<Result<ty::EarlyBinder<'static, Ty<'_>>, CyclePlaceholder>>()];
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

impl Erasable for Option<mir::DestructuredConstant<'_>> {
    type Storage = [u8; size_of::<Option<mir::DestructuredConstant<'static>>>()];
}

impl Erasable for ty::ImplTraitHeader<'_> {
    type Storage = [u8; size_of::<ty::ImplTraitHeader<'static>>()];
}

impl Erasable for Option<ty::EarlyBinder<'_, Ty<'_>>> {
    type Storage = [u8; size_of::<Option<ty::EarlyBinder<'static, Ty<'static>>>>()];
}

impl Erasable for rustc_hir::MaybeOwner<'_> {
    type Storage = [u8; size_of::<rustc_hir::MaybeOwner<'static>>()];
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

impl Erasable for ty::Binder<'_, &'_ ty::List<Ty<'_>>> {
    type Storage = [u8; size_of::<ty::Binder<'static, &'static ty::List<Ty<'static>>>>()];
}

impl<T0, T1> Erasable for (&'_ T0, &'_ T1) {
    type Storage = [u8; size_of::<(&'static (), &'static ())>()];
}

impl<T0> Erasable for (solve::QueryResult<'_>, &'_ T0) {
    type Storage = [u8; size_of::<(solve::QueryResult<'static>, &'static ())>()];
}

impl<T0, T1> Erasable for (&'_ T0, &'_ [T1]) {
    type Storage = [u8; size_of::<(&'static (), &'static [()])>()];
}

impl<T0, T1> Erasable for (&'_ [T0], &'_ [T1]) {
    type Storage = [u8; size_of::<(&'static [()], &'static [()])>()];
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
    // FIXME(#151565): Add `tidy-alphabetical-{start,end}` and sort this.
    (),
    bool,
    Option<(rustc_span::def_id::DefId, rustc_session::config::EntryFnType)>,
    Option<rustc_ast::expand::allocator::AllocatorKind>,
    Option<rustc_hir::ConstStability>,
    Option<rustc_hir::DefaultBodyStability>,
    Option<rustc_hir::Stability>,
    Option<rustc_data_structures::svh::Svh>,
    Option<rustc_hir::def::DefKind>,
    Option<rustc_hir::CoroutineKind>,
    Option<rustc_hir::HirId>,
    Option<rustc_middle::middle::stability::DeprecationEntry>,
    Option<rustc_middle::ty::AsyncDestructor>,
    Option<rustc_middle::ty::Destructor>,
    Option<rustc_middle::ty::ImplTraitInTraitData>,
    Option<rustc_middle::ty::ScalarInt>,
    Option<rustc_span::def_id::CrateNum>,
    Option<rustc_span::def_id::DefId>,
    Option<rustc_span::def_id::LocalDefId>,
    Option<rustc_span::Span>,
    Option<rustc_abi::FieldIdx>,
    Option<rustc_target::spec::PanicStrategy>,
    Option<usize>,
    Option<rustc_middle::ty::IntrinsicDef>,
    Option<rustc_abi::Align>,
    Result<(), rustc_errors::ErrorGuaranteed>,
    Result<(), rustc_middle::traits::query::NoSolution>,
    Result<rustc_middle::traits::EvaluationResult, rustc_middle::traits::OverflowError>,
    rustc_abi::ReprOptions,
    rustc_ast::expand::allocator::AllocatorKind,
    rustc_hir::DefaultBodyStability,
    rustc_hir::attrs::Deprecation,
    rustc_hir::attrs::EiiDecl,
    rustc_hir::attrs::EiiImpl,
    rustc_data_structures::svh::Svh,
    rustc_errors::ErrorGuaranteed,
    rustc_hir::Constness,
    rustc_hir::ConstStability,
    rustc_hir::def_id::DefId,
    rustc_hir::def_id::DefIndex,
    rustc_hir::def_id::LocalDefId,
    rustc_hir::def_id::LocalModDefId,
    rustc_hir::def::DefKind,
    rustc_hir::Defaultness,
    rustc_hir::definitions::DefKey,
    rustc_hir::CoroutineKind,
    rustc_hir::HirId,
    rustc_hir::IsAsync,
    rustc_hir::ItemLocalId,
    rustc_hir::LangItem,
    rustc_hir::OpaqueTyOrigin<rustc_hir::def_id::DefId>,
    rustc_hir::OwnerId,
    rustc_hir::Stability,
    rustc_hir::Upvar,
    rustc_index::bit_set::FiniteBitSet<u32>,
    rustc_middle::middle::deduced_param_attrs::DeducedParamAttrs,
    rustc_middle::middle::dependency_format::Linkage,
    rustc_middle::middle::exported_symbols::SymbolExportInfo,
    rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault,
    rustc_middle::middle::resolve_bound_vars::ResolvedArg,
    rustc_middle::middle::stability::DeprecationEntry,
    rustc_middle::mir::ConstQualifs,
    rustc_middle::mir::ConstValue,
    rustc_middle::mir::interpret::AllocId,
    rustc_middle::mir::interpret::CtfeProvenance,
    rustc_middle::mir::interpret::ErrorHandled,
    rustc_middle::thir::ExprId,
    rustc_middle::traits::CodegenObligationError,
    rustc_middle::traits::EvaluationResult,
    rustc_middle::traits::OverflowError,
    rustc_middle::traits::query::NoSolution,
    rustc_middle::traits::WellFormedLoc,
    rustc_middle::ty::adjustment::CoerceUnsizedInfo,
    rustc_middle::ty::AssocItem,
    rustc_middle::ty::AssocContainer,
    rustc_middle::ty::Asyncness,
    rustc_middle::ty::AsyncDestructor,
    rustc_middle::ty::AnonConstKind,
    rustc_middle::ty::Destructor,
    rustc_middle::ty::fast_reject::SimplifiedType,
    rustc_middle::ty::ImplPolarity,
    rustc_middle::ty::Representability,
    rustc_middle::ty::UnusedGenericParams,
    rustc_middle::ty::util::AlwaysRequiresDrop,
    rustc_middle::ty::Visibility<rustc_span::def_id::DefId>,
    rustc_middle::middle::codegen_fn_attrs::SanitizerFnAttrs,
    rustc_session::config::CrateType,
    rustc_session::config::EntryFnType,
    rustc_session::config::OptLevel,
    rustc_session::config::SymbolManglingVersion,
    rustc_session::cstore::CrateDepKind,
    rustc_session::cstore::ExternCrate,
    rustc_session::cstore::LinkagePreference,
    rustc_session::Limits,
    rustc_session::lint::LintExpectationId,
    rustc_span::def_id::CrateNum,
    rustc_span::def_id::DefPathHash,
    rustc_span::ExpnHash,
    rustc_span::ExpnId,
    rustc_span::Span,
    rustc_span::Symbol,
    rustc_span::Ident,
    rustc_target::spec::PanicStrategy,
    rustc_type_ir::Variance,
    u32,
    usize,
}

macro_rules! impl_erasable_for_single_lifetime_types {
    ($($($fake_path:ident)::+),+ $(,)?) => {
        $(
            impl<'tcx> Erasable for $($fake_path)::+<'tcx> {
                type Storage = [u8; size_of::<$($fake_path)::+<'static>>()];
            }
        )*
    }
}

// For types containing a single lifetime and no other generics, e.g.
// `Foo<'tcx>`, the erased storage is `[u8; size_of::<Foo<'static>>()]`.
//
// FIXME(#151565): Some of the hand-written impls above that only use one
// lifetime can probably be migrated here.
impl_erasable_for_single_lifetime_types! {
    // FIXME(#151565): Add `tidy-alphabetical-{start,end}` and sort this.
    rustc_middle::middle::exported_symbols::ExportedSymbol,
    rustc_middle::mir::Const,
    rustc_middle::mir::DestructuredConstant,
    rustc_middle::mir::ConstAlloc,
    rustc_middle::mir::interpret::GlobalId,
    rustc_middle::mir::interpret::LitToConstInput,
    rustc_middle::mir::interpret::EvalStaticInitializerRawResult,
    rustc_middle::mir::mono::MonoItemPartitions,
    rustc_middle::traits::query::MethodAutoderefStepsResult,
    rustc_middle::traits::query::type_op::AscribeUserType,
    rustc_middle::traits::query::type_op::Eq,
    rustc_middle::traits::query::type_op::ProvePredicate,
    rustc_middle::traits::query::type_op::Subtype,
    rustc_middle::ty::AdtDef,
    rustc_middle::ty::AliasTy,
    rustc_middle::ty::ClauseKind,
    rustc_middle::ty::ClosureTypeInfo,
    rustc_middle::ty::Const,
    rustc_middle::ty::DestructuredAdtConst,
    rustc_middle::ty::ExistentialTraitRef,
    rustc_middle::ty::FnSig,
    rustc_middle::ty::GenericArg,
    rustc_middle::ty::GenericPredicates,
    rustc_middle::ty::ConstConditions,
    rustc_middle::ty::inhabitedness::InhabitedPredicate,
    rustc_middle::ty::Instance,
    rustc_middle::ty::BoundVariableKind,
    rustc_middle::ty::InstanceKind,
    rustc_middle::ty::layout::FnAbiError,
    rustc_middle::ty::layout::LayoutError,
    rustc_middle::ty::ParamEnv,
    rustc_middle::ty::TypingEnv,
    rustc_middle::ty::Predicate,
    rustc_middle::ty::SymbolName,
    rustc_middle::ty::TraitRef,
    rustc_middle::ty::Ty,
    rustc_middle::ty::UnevaluatedConst,
    rustc_middle::ty::ValTree,
    rustc_middle::ty::VtblEntry,
}
