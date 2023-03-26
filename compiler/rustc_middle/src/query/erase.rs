use crate::ty;
use std::mem::{size_of, transmute_copy, MaybeUninit};

#[derive(Copy, Clone)]
pub struct Erased<T: Copy> {
    data: MaybeUninit<T>,
}

pub trait EraseType: Copy {
    type Result: Copy;
}

// Allow `type_alias_bounds` since compilation will fail without `EraseType`.
#[allow(type_alias_bounds)]
pub type Erase<T: Copy + EraseType> = Erased<impl Copy>;

#[inline(always)]
pub fn erase<T: EraseType>(src: T) -> Erase<T> {
    // Ensure the sizes match
    const {
        if std::mem::size_of::<T>() != std::mem::size_of::<T::Result>() {
            panic!("size of T must match erased type T::Result")
        }
    };

    Erased::<<T as EraseType>::Result> {
        // SAFETY: Is it safe to transmute to MaybeUninit for types with the same sizes.
        data: unsafe { transmute_copy(&src) },
    }
}

/// Restores an erased value.
#[inline(always)]
pub fn restore<T: EraseType>(value: Erase<T>) -> T {
    let value: Erased<<T as EraseType>::Result> = value;
    // SAFETY: Due to the use of impl Trait in `Erase` the only way to safetly create an instance
    // of `Erase` is to call `erase`, so we know that `value.data` is a valid instance of `T` of
    // the right size.
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

impl<T: Copy> EraseType for rustc_hir::MaybeOwner<T> {
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
    bool,
    rustc_ast::expand::allocator::AllocatorKind,
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
    rustc_middle::metadata::ModChild,
    rustc_middle::middle::dependency_format::Linkage,
    rustc_middle::middle::exported_symbols::SymbolExportInfo,
    rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault,
    rustc_middle::middle::resolve_bound_vars::ResolvedArg,
    rustc_middle::middle::stability::DeprecationEntry,
    rustc_middle::mir::ConstQualifs,
    rustc_middle::mir::interpret::AllocId,
    rustc_middle::mir::interpret::ErrorHandled,
    rustc_middle::mir::interpret::LitToConstError,
    rustc_middle::thir::ExprId,
    rustc_middle::traits::CodegenObligationError,
    rustc_middle::traits::EvaluationResult,
    rustc_middle::traits::OverflowError,
    rustc_middle::traits::query::NoSolution,
    rustc_middle::traits::WellFormedLoc,
    rustc_middle::ty::adjustment::CoerceUnsizedInfo,
    rustc_middle::ty::AssocItem,
    rustc_middle::ty::AssocItemContainer,
    rustc_middle::ty::BoundVariableKind,
    rustc_middle::ty::DeducedParamAttrs,
    rustc_middle::ty::Destructor,
    rustc_middle::ty::fast_reject::SimplifiedType,
    rustc_middle::ty::ImplPolarity,
    rustc_middle::ty::Representability,
    rustc_middle::ty::ReprOptions,
    rustc_middle::ty::UnusedGenericParams,
    rustc_middle::ty::util::AlwaysRequiresDrop,
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
    rustc_middle::hir::Owner,
    rustc_middle::middle::exported_symbols::ExportedSymbol,
    rustc_middle::mir::ConstantKind,
    rustc_middle::mir::DestructuredConstant,
    rustc_middle::mir::interpret::ConstAlloc,
    rustc_middle::mir::interpret::ConstValue,
    rustc_middle::mir::interpret::GlobalId,
    rustc_middle::mir::interpret::LitToConstInput,
    rustc_middle::traits::ChalkEnvironmentAndGoal,
    rustc_middle::traits::query::MethodAutoderefStepsResult,
    rustc_middle::traits::query::type_op::AscribeUserType,
    rustc_middle::traits::query::type_op::Eq,
    rustc_middle::traits::query::type_op::ProvePredicate,
    rustc_middle::traits::query::type_op::Subtype,
    rustc_middle::ty::AdtDef,
    rustc_middle::ty::AliasTy,
    rustc_middle::ty::Clause,
    rustc_middle::ty::ClosureTypeInfo,
    rustc_middle::ty::Const,
    rustc_middle::ty::DestructuredConst,
    rustc_middle::ty::ExistentialTraitRef,
    rustc_middle::ty::FnSig,
    rustc_middle::ty::GenericArg,
    rustc_middle::ty::GenericPredicates,
    rustc_middle::ty::inhabitedness::InhabitedPredicate,
    rustc_middle::ty::Instance,
    rustc_middle::ty::InstanceDef,
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
