use std::hash::Hash;

use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::DefIndex;
use rustc_index::{Idx, IndexVec};
use rustc_middle::ty::{Binder, EarlyBinder};
use rustc_span::Symbol;

use crate::rmeta::{LazyArray, LazyValue};

pub(crate) trait ParameterizedOverTcx: 'static {
    type Value<'tcx>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for Option<T> {
    type Value<'tcx> = Option<T::Value<'tcx>>;
}

impl<A: ParameterizedOverTcx, B: ParameterizedOverTcx> ParameterizedOverTcx for (A, B) {
    type Value<'tcx> = (A::Value<'tcx>, B::Value<'tcx>);
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for Vec<T> {
    type Value<'tcx> = Vec<T::Value<'tcx>>;
}

impl<I: Idx + 'static, T: ParameterizedOverTcx> ParameterizedOverTcx for IndexVec<I, T> {
    type Value<'tcx> = IndexVec<I, T::Value<'tcx>>;
}

impl<I: Hash + Eq + 'static, T: ParameterizedOverTcx> ParameterizedOverTcx for UnordMap<I, T> {
    type Value<'tcx> = UnordMap<I, T::Value<'tcx>>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for Binder<'static, T> {
    type Value<'tcx> = Binder<'tcx, T::Value<'tcx>>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for EarlyBinder<'static, T> {
    type Value<'tcx> = EarlyBinder<'tcx, T::Value<'tcx>>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for LazyValue<T> {
    type Value<'tcx> = LazyValue<T::Value<'tcx>>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for LazyArray<T> {
    type Value<'tcx> = LazyArray<T::Value<'tcx>>;
}

macro_rules! trivially_parameterized_over_tcx {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ParameterizedOverTcx for $ty {
                #[allow(unused_lifetimes)]
                type Value<'tcx> = $ty;
            }
        )*
    }
}

trivially_parameterized_over_tcx! {
    bool,
    u64,
    usize,
    std::string::String,
    // tidy-alphabetical-start
    crate::rmeta::AttrFlags,
    crate::rmeta::CrateDep,
    crate::rmeta::CrateHeader,
    crate::rmeta::CrateRoot,
    crate::rmeta::IncoherentImpls,
    crate::rmeta::RawDefId,
    crate::rmeta::TraitImpls,
    crate::rmeta::VariantData,
    rustc_abi::ReprOptions,
    rustc_ast::DelimArgs,
    rustc_hir::Attribute,
    rustc_hir::ConstStability,
    rustc_hir::Constness,
    rustc_hir::CoroutineKind,
    rustc_hir::DefaultBodyStability,
    rustc_hir::Defaultness,
    rustc_hir::LangItem,
    rustc_hir::OpaqueTyOrigin<rustc_hir::def_id::DefId>,
    rustc_hir::PreciseCapturingArgKind<Symbol, Symbol>,
    rustc_hir::Safety,
    rustc_hir::Stability,
    rustc_hir::attrs::Deprecation,
    rustc_hir::attrs::StrippedCfgItem<rustc_hir::def_id::DefIndex>,
    rustc_hir::def::DefKind,
    rustc_hir::def::DocLinkResMap,
    rustc_hir::def_id::DefId,
    rustc_hir::def_id::DefIndex,
    rustc_hir::definitions::DefKey,
    rustc_index::bit_set::DenseBitSet<u32>,
    rustc_middle::metadata::ModChild,
    rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs,
    rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile,
    rustc_middle::middle::exported_symbols::SymbolExportInfo,
    rustc_middle::middle::lib_features::FeatureStability,
    rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault,
    rustc_middle::mir::ConstQualifs,
    rustc_middle::ty::AnonConstKind,
    rustc_middle::ty::AssocContainer,
    rustc_middle::ty::AsyncDestructor,
    rustc_middle::ty::Asyncness,
    rustc_middle::ty::DeducedParamAttrs,
    rustc_middle::ty::Destructor,
    rustc_middle::ty::Generics,
    rustc_middle::ty::ImplTraitInTraitData,
    rustc_middle::ty::IntrinsicDef,
    rustc_middle::ty::TraitDef,
    rustc_middle::ty::Variance,
    rustc_middle::ty::Visibility<DefIndex>,
    rustc_middle::ty::adjustment::CoerceUnsizedInfo,
    rustc_middle::ty::fast_reject::SimplifiedType,
    rustc_session::config::TargetModifier,
    rustc_session::cstore::ForeignModule,
    rustc_session::cstore::LinkagePreference,
    rustc_session::cstore::NativeLib,
    rustc_span::ExpnData,
    rustc_span::ExpnHash,
    rustc_span::ExpnId,
    rustc_span::Ident,
    rustc_span::SourceFile,
    rustc_span::Span,
    rustc_span::Symbol,
    rustc_span::hygiene::SyntaxContextKey,
    // tidy-alphabetical-end
}

// HACK(compiler-errors): This macro rule can only take a fake path,
// not a real, due to parsing ambiguity reasons.
macro_rules! parameterized_over_tcx {
    ($($( $fake_path:ident )::+ ),+ $(,)?) => {
        $(
            impl ParameterizedOverTcx for $( $fake_path )::+ <'static> {
                type Value<'tcx> = $( $fake_path )::+ <'tcx>;
            }
        )*
    }
}

parameterized_over_tcx! {
    // tidy-alphabetical-start
    crate::rmeta::DefPathHashMapRef,
    rustc_middle::middle::exported_symbols::ExportedSymbol,
    rustc_middle::mir::Body,
    rustc_middle::mir::CoroutineLayout,
    rustc_middle::mir::interpret::ConstAllocation,
    rustc_middle::ty::Clause,
    rustc_middle::ty::Const,
    rustc_middle::ty::ConstConditions,
    rustc_middle::ty::FnSig,
    rustc_middle::ty::GenericPredicates,
    rustc_middle::ty::ImplTraitHeader,
    rustc_middle::ty::TraitRef,
    rustc_middle::ty::Ty,
    // tidy-alphabetical-end
}
