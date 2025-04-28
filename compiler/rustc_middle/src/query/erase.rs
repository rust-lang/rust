use std::ffi::OsStr;
use std::intrinsics::transmute_unchecked;
use std::mem::MaybeUninit;

use rustc_span::ErrorGuaranteed;

use crate::query::CyclePlaceholder;
use crate::ty::adjustment::CoerceUnsizedInfo;
use crate::ty::{self, Ty};
use crate::{mir, traits};

#[derive(Copy, Clone)]
pub struct Erased<T: Copy> {
    // We use `MaybeUninit` here so we can store any value
    // in `data` since we aren't actually storing a `T`.
    data: MaybeUninit<T>,
}

pub trait EraseType: Copy {
    type Result: Copy;
}

// Allow `type_alias_bounds` since compilation will fail without `EraseType`.
#[allow(type_alias_bounds)]
pub type Erase<T: EraseType> = Erased<impl Copy>;

#[inline(always)]
#[define_opaque(Erase)]
pub fn erase<T: EraseType>(src: T) -> Erase<T> {
    // Ensure the sizes match
    const {
        if size_of::<T>() != size_of::<T::Result>() {
            panic!("size of T must match erased type T::Result")
        }
    };

    Erased::<<T as EraseType>::Result> {
        // `transmute_unchecked` is needed here because it does not have `transmute`'s size check
        // (and thus allows to transmute between `T` and `MaybeUninit<T::Result>`) (we do the size
        // check ourselves in the `const` block above).
        //
        // `transmute_copy` is also commonly used for this (and it would work here since
        // `EraseType: Copy`), but `transmute_unchecked` better explains the intent.
        //
        // SAFETY: It is safe to transmute to MaybeUninit for types with the same sizes.
        data: unsafe { transmute_unchecked::<T, MaybeUninit<T::Result>>(src) },
    }
}

/// Restores an erased value.
#[inline(always)]
#[define_opaque(Erase)]
pub fn restore<T: EraseType>(value: Erase<T>) -> T {
    let value: Erased<<T as EraseType>::Result> = value;
    // See comment in `erase` for why we use `transmute_unchecked`.
    //
    // SAFETY: Due to the use of impl Trait in `Erase` the only way to safely create an instance
    // of `Erase` is to call `erase`, so we know that `value.data` is a valid instance of `T` of
    // the right size.
    unsafe { transmute_unchecked::<MaybeUninit<T::Result>, T>(value.data) }
}

impl<T> EraseType for &'_ T {
    type Result = [u8; size_of::<&'static ()>()];
}

impl<T> EraseType for &'_ [T] {
    type Result = [u8; size_of::<&'static [()]>()];
}

impl EraseType for &'_ OsStr {
    type Result = [u8; size_of::<&'static OsStr>()];
}

impl<T> EraseType for &'_ ty::List<T> {
    type Result = [u8; size_of::<&'static ty::List<()>>()];
}

impl<T> EraseType for &'_ ty::ListWithCachedTypeInfo<T> {
    type Result = [u8; size_of::<&'static ty::ListWithCachedTypeInfo<()>>()];
}

impl<I: rustc_index::Idx, T> EraseType for &'_ rustc_index::IndexSlice<I, T> {
    type Result = [u8; size_of::<&'static rustc_index::IndexSlice<u32, ()>>()];
}

impl<T> EraseType for Result<&'_ T, traits::query::NoSolution> {
    type Result = [u8; size_of::<Result<&'static (), traits::query::NoSolution>>()];
}

impl<T> EraseType for Result<&'_ [T], traits::query::NoSolution> {
    type Result = [u8; size_of::<Result<&'static [()], traits::query::NoSolution>>()];
}

impl<T> EraseType for Result<&'_ T, rustc_errors::ErrorGuaranteed> {
    type Result = [u8; size_of::<Result<&'static (), rustc_errors::ErrorGuaranteed>>()];
}

impl<T> EraseType for Result<&'_ [T], rustc_errors::ErrorGuaranteed> {
    type Result = [u8; size_of::<Result<&'static [()], rustc_errors::ErrorGuaranteed>>()];
}

impl<T> EraseType for Result<&'_ T, traits::CodegenObligationError> {
    type Result = [u8; size_of::<Result<&'static (), traits::CodegenObligationError>>()];
}

impl<T> EraseType for Result<&'_ T, &'_ ty::layout::FnAbiError<'_>> {
    type Result = [u8; size_of::<Result<&'static (), &'static ty::layout::FnAbiError<'static>>>()];
}

impl<T> EraseType for Result<(&'_ T, crate::thir::ExprId), rustc_errors::ErrorGuaranteed> {
    type Result = [u8; size_of::<
        Result<(&'static (), crate::thir::ExprId), rustc_errors::ErrorGuaranteed>,
    >()];
}

impl EraseType for Result<Option<ty::Instance<'_>>, rustc_errors::ErrorGuaranteed> {
    type Result =
        [u8; size_of::<Result<Option<ty::Instance<'static>>, rustc_errors::ErrorGuaranteed>>()];
}

impl EraseType for Result<CoerceUnsizedInfo, rustc_errors::ErrorGuaranteed> {
    type Result = [u8; size_of::<Result<CoerceUnsizedInfo, rustc_errors::ErrorGuaranteed>>()];
}

impl EraseType
    for Result<Option<ty::EarlyBinder<'_, ty::Const<'_>>>, rustc_errors::ErrorGuaranteed>
{
    type Result = [u8; size_of::<
        Result<Option<ty::EarlyBinder<'static, ty::Const<'static>>>, rustc_errors::ErrorGuaranteed>,
    >()];
}

impl EraseType for Result<ty::GenericArg<'_>, traits::query::NoSolution> {
    type Result = [u8; size_of::<Result<ty::GenericArg<'static>, traits::query::NoSolution>>()];
}

impl EraseType for Result<bool, &ty::layout::LayoutError<'_>> {
    type Result = [u8; size_of::<Result<bool, &'static ty::layout::LayoutError<'static>>>()];
}

impl EraseType for Result<rustc_abi::TyAndLayout<'_, Ty<'_>>, &ty::layout::LayoutError<'_>> {
    type Result = [u8; size_of::<
        Result<
            rustc_abi::TyAndLayout<'static, Ty<'static>>,
            &'static ty::layout::LayoutError<'static>,
        >,
    >()];
}

impl EraseType for Result<mir::ConstAlloc<'_>, mir::interpret::ErrorHandled> {
    type Result = [u8; size_of::<Result<mir::ConstAlloc<'static>, mir::interpret::ErrorHandled>>()];
}

impl EraseType for Result<mir::ConstValue<'_>, mir::interpret::ErrorHandled> {
    type Result = [u8; size_of::<Result<mir::ConstValue<'static>, mir::interpret::ErrorHandled>>()];
}

impl EraseType for Result<Result<ty::ValTree<'_>, Ty<'_>>, mir::interpret::ErrorHandled> {
    type Result = [u8; size_of::<
        Result<Result<ty::ValTree<'static>, Ty<'static>>, mir::interpret::ErrorHandled>,
    >()];
}

impl EraseType for Result<&'_ ty::List<Ty<'_>>, ty::util::AlwaysRequiresDrop> {
    type Result =
        [u8; size_of::<Result<&'static ty::List<Ty<'static>>, ty::util::AlwaysRequiresDrop>>()];
}

impl EraseType for Result<ty::EarlyBinder<'_, Ty<'_>>, CyclePlaceholder> {
    type Result = [u8; size_of::<Result<ty::EarlyBinder<'static, Ty<'_>>, CyclePlaceholder>>()];
}

impl<T> EraseType for Option<&'_ T> {
    type Result = [u8; size_of::<Option<&'static ()>>()];
}

impl<T> EraseType for Option<&'_ [T]> {
    type Result = [u8; size_of::<Option<&'static [()]>>()];
}

impl EraseType for Option<&'_ OsStr> {
    type Result = [u8; size_of::<Option<&'static OsStr>>()];
}

impl EraseType for Option<mir::DestructuredConstant<'_>> {
    type Result = [u8; size_of::<Option<mir::DestructuredConstant<'static>>>()];
}

impl EraseType for Option<ty::ImplTraitHeader<'_>> {
    type Result = [u8; size_of::<Option<ty::ImplTraitHeader<'static>>>()];
}

impl EraseType for Option<ty::EarlyBinder<'_, Ty<'_>>> {
    type Result = [u8; size_of::<Option<ty::EarlyBinder<'static, Ty<'static>>>>()];
}

impl EraseType for rustc_hir::MaybeOwner<'_> {
    type Result = [u8; size_of::<rustc_hir::MaybeOwner<'static>>()];
}

impl<T: EraseType> EraseType for ty::EarlyBinder<'_, T> {
    type Result = T::Result;
}

impl EraseType for ty::Binder<'_, ty::FnSig<'_>> {
    type Result = [u8; size_of::<ty::Binder<'static, ty::FnSig<'static>>>()];
}

impl EraseType for ty::Binder<'_, &'_ ty::List<Ty<'_>>> {
    type Result = [u8; size_of::<ty::Binder<'static, &'static ty::List<Ty<'static>>>>()];
}

impl<T0, T1> EraseType for (&'_ T0, &'_ T1) {
    type Result = [u8; size_of::<(&'static (), &'static ())>()];
}

impl<T0, T1> EraseType for (&'_ T0, &'_ [T1]) {
    type Result = [u8; size_of::<(&'static (), &'static [()])>()];
}

impl<T0, T1> EraseType for (&'_ [T0], &'_ [T1]) {
    type Result = [u8; size_of::<(&'static [()], &'static [()])>()];
}

impl<T0> EraseType for (&'_ T0, Result<(), ErrorGuaranteed>) {
    type Result = [u8; size_of::<(&'static (), Result<(), ErrorGuaranteed>)>()];
}

macro_rules! trivial {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl EraseType for $ty {
                type Result = [u8; size_of::<$ty>()];
            }
        )*
    }
}

trivial! {
    (),
    bool,
    Option<(rustc_span::def_id::DefId, rustc_session::config::EntryFnType)>,
    Option<rustc_ast::expand::allocator::AllocatorKind>,
    Option<rustc_attr_data_structures::ConstStability>,
    Option<rustc_attr_data_structures::DefaultBodyStability>,
    Option<rustc_attr_data_structures::Stability>,
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
    Result<(), rustc_errors::ErrorGuaranteed>,
    Result<(), rustc_middle::traits::query::NoSolution>,
    Result<rustc_middle::traits::EvaluationResult, rustc_middle::traits::OverflowError>,
    rustc_abi::ReprOptions,
    rustc_ast::expand::allocator::AllocatorKind,
    rustc_attr_data_structures::ConstStability,
    rustc_attr_data_structures::DefaultBodyStability,
    rustc_attr_data_structures::Deprecation,
    rustc_attr_data_structures::Stability,
    rustc_data_structures::svh::Svh,
    rustc_errors::ErrorGuaranteed,
    rustc_hir::Constness,
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
    rustc_hir::Upvar,
    rustc_index::bit_set::FiniteBitSet<u32>,
    rustc_middle::middle::dependency_format::Linkage,
    rustc_middle::middle::exported_symbols::SymbolExportInfo,
    rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault,
    rustc_middle::middle::resolve_bound_vars::ResolvedArg,
    rustc_middle::middle::stability::DeprecationEntry,
    rustc_middle::mir::ConstQualifs,
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
    rustc_middle::ty::AssocItemContainer,
    rustc_middle::ty::Asyncness,
    rustc_middle::ty::AsyncDestructor,
    rustc_middle::ty::BoundVariableKind,
    rustc_middle::ty::DeducedParamAttrs,
    rustc_middle::ty::Destructor,
    rustc_middle::ty::fast_reject::SimplifiedType,
    rustc_middle::ty::ImplPolarity,
    rustc_middle::ty::Representability,
    rustc_middle::ty::UnusedGenericParams,
    rustc_middle::ty::util::AlwaysRequiresDrop,
    rustc_middle::ty::Visibility<rustc_span::def_id::DefId>,
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

macro_rules! tcx_lifetime {
    ($($($fake_path:ident)::+),+ $(,)?) => {
        $(
            impl<'tcx> EraseType for $($fake_path)::+<'tcx> {
                type Result = [u8; size_of::<$($fake_path)::+<'static>>()];
            }
        )*
    }
}

tcx_lifetime! {
    rustc_middle::middle::exported_symbols::ExportedSymbol,
    rustc_middle::mir::Const,
    rustc_middle::mir::DestructuredConstant,
    rustc_middle::mir::ConstAlloc,
    rustc_middle::mir::ConstValue,
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
    rustc_middle::ty::DestructuredConst,
    rustc_middle::ty::ExistentialTraitRef,
    rustc_middle::ty::FnSig,
    rustc_middle::ty::GenericArg,
    rustc_middle::ty::GenericPredicates,
    rustc_middle::ty::ConstConditions,
    rustc_middle::ty::inhabitedness::InhabitedPredicate,
    rustc_middle::ty::Instance,
    rustc_middle::ty::InstanceKind,
    rustc_middle::ty::layout::FnAbiError,
    rustc_middle::ty::layout::LayoutError,
    rustc_middle::ty::ParamEnv,
    rustc_middle::ty::Predicate,
    rustc_middle::ty::SymbolName,
    rustc_middle::ty::TraitRef,
    rustc_middle::ty::Ty,
    rustc_middle::ty::UnevaluatedConst,
    rustc_middle::ty::ValTree,
    rustc_middle::ty::VtblEntry,
}
