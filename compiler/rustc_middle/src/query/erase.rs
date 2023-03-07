use crate::mir;
use crate::traits;
use crate::ty;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_hir::MaybeOwner;
pub use rustc_middle::traits::query::type_op;
use std::intrinsics::type_name;
use std::{
    fmt,
    mem::{size_of, transmute, transmute_copy, MaybeUninit},
};

#[derive(Copy, Clone)]
struct Erased<T: Copy> {
    data: MaybeUninit<T>,
    formatter: fn(&Self, &mut std::fmt::Formatter<'_>) -> std::fmt::Result,
    #[cfg(debug_assertions)]
    type_id: &'static str,
}

impl<T: Copy> fmt::Debug for Erased<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.formatter)(self, f)
    }
}

pub trait EraseType: Copy {
    type Result: Copy;
}

pub type Erase<T: EraseType + fmt::Debug> = impl Copy + fmt::Debug;

fn formatter<T: EraseType + fmt::Debug>(
    this: &Erase<T>,
    f: &mut std::fmt::Formatter<'_>,
) -> fmt::Result {
    fmt::Debug::fmt(restore_ref(this), f)
}

#[inline(always)]
pub fn erase<T: EraseType + fmt::Debug>(src: T) -> Erase<T> {
    assert_eq!(
        size_of::<T>(),
        size_of::<T::Result>(),
        "size of {} must match erased type {}",
        type_name::<T>(),
        type_name::<T::Result>()
    );
    Erased::<<T as EraseType>::Result> {
        // SAFETY:: Is it safe to transmute to MaybeUninit
        data: unsafe { transmute_copy(&src) },
        #[cfg(debug_assertions)]
        type_id: type_name::<T>(),
        formatter: formatter::<T>,
    }
}

/// Restores an erased value.
///
/// This is only safe if `value` is a valid instance of `T`.
/// For example if `T` was erased with `erase` previously.
#[inline(always)]
pub fn restore_ref<T: EraseType + fmt::Debug>(value: &Erase<T>) -> &T {
    let value: &Erased<<T as EraseType>::Result> = &value;
    #[cfg(debug_assertions)]
    assert_eq!(value.type_id, type_name::<T>());
    assert_eq!(
        size_of::<T>(),
        size_of::<T::Result>(),
        "size of {} must match erased type {}",
        type_name::<T>(),
        type_name::<T::Result>()
    );
    // SAFETY: Thanks to the TAIT, this function can only be called with the result of `erase<T>`.
    unsafe { transmute(&value.data) }
}

/// Restores an erased value.
///
/// This is only safe if `value` is a valid instance of `T`.
/// For example if `T` was erased with `erase` previously.
#[inline(always)]
pub fn restore<T: EraseType + fmt::Debug>(value: Erase<T>) -> T {
    let value: Erased<<T as EraseType>::Result> = value;
    #[cfg(debug_assertions)]
    assert_eq!(value.type_id, type_name::<T>());
    assert_eq!(
        size_of::<T>(),
        size_of::<T::Result>(),
        "size of {} must match erased type {}",
        type_name::<T>(),
        type_name::<T::Result>()
    );
    // SAFETY: Thanks to the TAIT, this function can only be called with the result of `erase<T>`.
    unsafe { transmute_copy(&value.data) }
}

impl<T> EraseType for &'_ T {
    type Result = [u8; size_of::<*const ()>()];
}

impl<T> EraseType for &'_ [T] {
    type Result = [u8; size_of::<*const [()]>()];
}

impl<T> EraseType for &'_ ty::List<T> {
    type Result = [u8; size_of::<*const ()>()];
}

impl<T: Copy, E: Copy> EraseType for Result<T, E> {
    type Result = Self;
}

impl<T: Copy> EraseType for Option<T> {
    type Result = Self;
}

impl<T: Copy> EraseType for MaybeOwner<T> {
    type Result = Self;
}

impl<T: Copy> EraseType for ty::Visibility<T> {
    type Result = Self;
}

impl<T: Copy> EraseType for ty::Binder<'_, T> {
    type Result = Self;
}

impl<T: Copy> EraseType for ty::EarlyBinder<T> {
    type Result = Self;
}

impl<T0: Copy, T1: Copy> EraseType for (T0, T1) {
    type Result = Self;
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
    AllocatorKind,
    bool,
    crate::metadata::ModChild,
    crate::middle::exported_symbols::SymbolExportInfo,
    crate::middle::resolve_bound_vars::ObjectLifetimeDefault,
    crate::middle::resolve_bound_vars::ResolvedArg,
    crate::middle::stability::DeprecationEntry,
    crate::mir::ConstQualifs,
    crate::mir::interpret::ErrorHandled,
    crate::mir::interpret::LitToConstError,
    crate::thir::ExprId,
    crate::traits::CodegenObligationError,
    mir::Field,
    mir::interpret::AllocId,
    rustc_attr::ConstStability,
    rustc_attr::DefaultBodyStability,
    rustc_attr::Deprecation,
    rustc_attr::Stability,
    rustc_data_structures::svh::Svh,
    rustc_errors::ErrorGuaranteed,
    rustc_hir::Constness,
    rustc_hir::def_id::DefId,
    rustc_hir::def_id::DefIndex,
    rustc_hir::def_id::LocalDefId,
    rustc_hir::def::DefKind,
    rustc_hir::Defaultness,
    rustc_hir::definitions::DefKey,
    rustc_hir::GeneratorKind,
    rustc_hir::HirId,
    rustc_hir::IsAsync,
    rustc_hir::ItemLocalId,
    rustc_hir::LangItem,
    rustc_hir::OwnerId,
    rustc_hir::Upvar,
    rustc_index::bit_set::FiniteBitSet<u32>,
    rustc_middle::middle::dependency_format::Linkage,
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
    rustc_span::symbol::Ident,
    rustc_target::spec::PanicStrategy,
    rustc_type_ir::Variance,
    traits::EvaluationResult,
    traits::OverflowError,
    traits::query::NoSolution,
    traits::WellFormedLoc,
    ty::adjustment::CoerceUnsizedInfo,
    ty::AssocItem,
    ty::AssocItemContainer,
    ty::BoundVariableKind,
    ty::DeducedParamAttrs,
    ty::Destructor,
    ty::fast_reject::SimplifiedType,
    ty::ImplPolarity,
    ty::Representability,
    ty::ReprOptions,
    ty::UnusedGenericParams,
    ty::util::AlwaysRequiresDrop,
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
    crate::middle::exported_symbols::ExportedSymbol,
    crate::mir::DestructuredConstant,
    crate::mir::interpret::ConstValue,
    mir::ConstantKind,
    mir::interpret::ConstAlloc,
    mir::interpret::GlobalId,
    mir::interpret::LitToConstInput,
    rustc_middle::hir::Owner,
    traits::ChalkEnvironmentAndGoal,
    traits::query::MethodAutoderefStepsResult,
    ty::AdtDef,
    ty::AliasTy,
    ty::Clause,
    ty::ClosureTypeInfo,
    ty::Const,
    ty::DestructuredConst,
    ty::ExistentialTraitRef,
    ty::FnSig,
    ty::GenericArg,
    ty::GenericPredicates,
    ty::inhabitedness::InhabitedPredicate,
    ty::Instance,
    ty::InstanceDef,
    ty::layout::FnAbiError,
    ty::layout::LayoutError,
    ty::ParamEnv,
    ty::Predicate,
    ty::SymbolName,
    ty::TraitRef,
    ty::Ty,
    ty::UnevaluatedConst,
    ty::ValTree,
    ty::VtblEntry,
    type_op::AscribeUserType,
    type_op::Eq,
    type_op::ProvePredicate,
    type_op::Subtype,
}
