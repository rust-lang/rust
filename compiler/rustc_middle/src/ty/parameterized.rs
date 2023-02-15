use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefIndex;
use rustc_index::vec::{Idx, IndexVec};

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

impl<I: 'static, T: ParameterizedOverTcx> ParameterizedOverTcx for FxHashMap<I, T> {
    type Value<'tcx> = FxHashMap<I, T::Value<'tcx>>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for ty::Binder<'static, T> {
    type Value<'tcx> = ty::Binder<'tcx, T::Value<'tcx>>;
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for ty::EarlyBinder<T> {
    type Value<'tcx> = ty::EarlyBinder<T::Value<'tcx>>;
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
    bool,
    std::string::String,
    crate::metadata::ModChild,
    crate::middle::codegen_fn_attrs::CodegenFnAttrs,
    crate::middle::exported_symbols::SymbolExportInfo,
    crate::middle::resolve_lifetime::ObjectLifetimeDefault,
    crate::mir::ConstQualifs,
    ty::AssocItemContainer,
    ty::DeducedParamAttrs,
    ty::Generics,
    ty::ImplPolarity,
    ty::ReprOptions,
    ty::TraitDef,
    ty::UnusedGenericParams,
    ty::Visibility<DefIndex>,
    ty::adjustment::CoerceUnsizedInfo,
    ty::fast_reject::SimplifiedType,
    rustc_ast::Attribute,
    rustc_ast::DelimArgs,
    rustc_attr::ConstStability,
    rustc_attr::DefaultBodyStability,
    rustc_attr::Deprecation,
    rustc_attr::Stability,
    rustc_hir::Constness,
    rustc_hir::Defaultness,
    rustc_hir::GeneratorKind,
    rustc_hir::IsAsync,
    rustc_hir::LangItem,
    rustc_hir::def::DefKind,
    rustc_hir::def::DocLinkResMap,
    rustc_hir::def_id::DefId,
    rustc_hir::def_id::DefIndex,
    rustc_hir::definitions::DefKey,
    rustc_index::bit_set::BitSet<u32>,
    rustc_index::bit_set::FiniteBitSet<u32>,
    rustc_session::cstore::ForeignModule,
    rustc_session::cstore::LinkagePreference,
    rustc_session::cstore::NativeLib,
    rustc_span::DebuggerVisualizerFile,
    rustc_span::ExpnData,
    rustc_span::ExpnHash,
    rustc_span::ExpnId,
    rustc_span::SourceFile,
    rustc_span::Span,
    rustc_span::Symbol,
    rustc_span::def_id::DefPathHash,
    rustc_span::hygiene::SyntaxContextData,
    rustc_span::symbol::Ident,
    rustc_type_ir::Variance,
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
    crate::mir::GeneratorLayout,
    ty::Ty,
    ty::FnSig,
    ty::GenericPredicates,
    ty::TraitRef,
    ty::Const,
    ty::Predicate,
    ty::Clause,
    ty::GeneratorDiagnosticData,
}
