use std::hash::Hash;

use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::DefIndex;
use rustc_index::{Idx, IndexVec};
use rustc_span::Symbol;

use crate::ty;

pub trait ParameterizedOverTcx: 'static {
    type Value<'tcx>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for &'static [T] {
    type Value<'tcx> = &'tcx [T::Value<'tcx>];
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for Option<T> {
    type Value<'tcx> = Option<T::Value<'tcx>>;
}

impl<A: ParameterizedOverTcx, B: ParameterizedOverTcx> ParameterizedOverTcx for (A, B) {
    type Value<'tcx> = (A::Value<'tcx>, B::Value<'tcx>);
}

impl<I: Idx + 'static, T: ParameterizedOverTcx> ParameterizedOverTcx for IndexVec<I, T> {
    type Value<'tcx> = IndexVec<I, T::Value<'tcx>>;
}

impl<I: Hash + Eq + 'static, T: ParameterizedOverTcx> ParameterizedOverTcx for UnordMap<I, T> {
    type Value<'tcx> = UnordMap<I, T::Value<'tcx>>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for ty::Binder<'static, T> {
    type Value<'tcx> = ty::Binder<'tcx, T::Value<'tcx>>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for ty::EarlyBinder<'static, T> {
    type Value<'tcx> = ty::EarlyBinder<'tcx, T::Value<'tcx>>;
}

#[macro_export]
macro_rules! trivially_parameterized_over_tcx {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl $crate::ty::ParameterizedOverTcx for $ty {
                #[allow(unused_lifetimes)]
                type Value<'tcx> = $ty;
            }
        )*
    }
}

trivially_parameterized_over_tcx! {
    usize,
    (),
    u32,
    u64,
    bool,
    std::string::String,
    crate::metadata::ModChild,
    crate::middle::codegen_fn_attrs::CodegenFnAttrs,
    crate::middle::debugger_visualizer::DebuggerVisualizerFile,
    crate::middle::exported_symbols::SymbolExportInfo,
    crate::middle::lib_features::FeatureStability,
    crate::middle::resolve_bound_vars::ObjectLifetimeDefault,
    crate::mir::ConstQualifs,
    ty::AsyncDestructor,
    ty::AssocItemContainer,
    ty::Asyncness,
    ty::DeducedParamAttrs,
    ty::Destructor,
    ty::Generics,
    ty::ImplPolarity,
    ty::ImplTraitInTraitData,
    ty::ReprOptions,
    ty::TraitDef,
    ty::UnusedGenericParams,
    ty::Visibility<DefIndex>,
    ty::adjustment::CoerceUnsizedInfo,
    ty::fast_reject::SimplifiedType,
    ty::IntrinsicDef,
    rustc_ast::Attribute,
    rustc_ast::DelimArgs,
    rustc_ast::expand::StrippedCfgItem<rustc_hir::def_id::DefIndex>,
    rustc_attr_data_structures::ConstStability,
    rustc_attr_data_structures::DefaultBodyStability,
    rustc_attr_data_structures::Deprecation,
    rustc_attr_data_structures::Stability,
    rustc_hir::Constness,
    rustc_hir::Defaultness,
    rustc_hir::Safety,
    rustc_hir::CoroutineKind,
    rustc_hir::IsAsync,
    rustc_hir::LangItem,
    rustc_hir::def::DefKind,
    rustc_hir::def::DocLinkResMap,
    rustc_hir::def_id::DefId,
    rustc_hir::def_id::DefIndex,
    rustc_hir::definitions::DefKey,
    rustc_hir::OpaqueTyOrigin<rustc_hir::def_id::DefId>,
    rustc_hir::PreciseCapturingArgKind<Symbol, Symbol>,
    rustc_index::bit_set::DenseBitSet<u32>,
    rustc_index::bit_set::FiniteBitSet<u32>,
    rustc_session::cstore::ForeignModule,
    rustc_session::cstore::LinkagePreference,
    rustc_session::cstore::NativeLib,
    rustc_session::config::TargetModifier,
    rustc_span::ExpnData,
    rustc_span::ExpnHash,
    rustc_span::ExpnId,
    rustc_span::SourceFile,
    rustc_span::Span,
    rustc_span::Symbol,
    rustc_span::def_id::DefPathHash,
    rustc_span::hygiene::SyntaxContextDataNonRecursive,
    rustc_span::Ident,
    rustc_type_ir::Variance,
    rustc_hir::Attribute,
}

// HACK(compiler-errors): This macro rule can only take a fake path,
// not a real, due to parsing ambiguity reasons.
#[macro_export]
macro_rules! parameterized_over_tcx {
    ($($($fake_path:ident)::+),+ $(,)?) => {
        $(
            impl $crate::ty::ParameterizedOverTcx for $($fake_path)::+<'static> {
                type Value<'tcx> = $($fake_path)::+<'tcx>;
            }
        )*
    }
}

parameterized_over_tcx! {
    crate::middle::exported_symbols::ExportedSymbol,
    crate::mir::Body,
    crate::mir::CoroutineLayout,
    crate::mir::interpret::ConstAllocation,
    ty::Ty,
    ty::FnSig,
    ty::GenericPredicates,
    ty::ConstConditions,
    ty::TraitRef,
    ty::Const,
    ty::Predicate,
    ty::Clause,
    ty::ClauseKind,
    ty::ImplTraitHeader,
}
