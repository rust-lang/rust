use rustc_hir::def_id::DefId;
use rustc_index::vec::{Idx, IndexVec};

use crate::middle::exported_symbols::ExportedSymbol;
use crate::mir::Body;
use crate::ty::abstract_const::Node;
use crate::ty::{
    self, Const, FnSig, GeneratorDiagnosticData, GenericPredicates, Predicate, TraitRef, Ty,
};

pub trait ParameterizedOverTcx: 'static {
    #[allow(unused_lifetimes)]
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

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for ty::Binder<'static, T> {
    type Value<'tcx> = ty::Binder<'tcx, T::Value<'tcx>>;
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
    std::string::String,
    crate::metadata::ModChild,
    crate::middle::codegen_fn_attrs::CodegenFnAttrs,
    crate::middle::exported_symbols::SymbolExportInfo,
    crate::middle::resolve_lifetime::ObjectLifetimeDefault,
    crate::mir::ConstQualifs,
    ty::AssocItemContainer,
    ty::Generics,
    ty::ImplPolarity,
    ty::ReprOptions,
    ty::TraitDef,
    ty::Visibility,
    ty::adjustment::CoerceUnsizedInfo,
    ty::fast_reject::SimplifiedTypeGen<DefId>,
    rustc_ast::Attribute,
    rustc_ast::MacArgs,
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
    rustc_hir::def_id::DefIndex,
    rustc_hir::definitions::DefKey,
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

// HACK(compiler-errors): This macro rule can only take an ident,
// not a path, due to parsing ambiguity reasons. That means we gotta
// import all of these types above.
#[macro_export]
macro_rules! parameterized_over_tcx {
    ($($ident:ident),+ $(,)?) => {
        $(
            impl $crate::ty::ParameterizedOverTcx for $ident<'static> {
                type Value<'tcx> = $ident<'tcx>;
            }
        )*
    }
}

parameterized_over_tcx! {
    Ty,
    FnSig,
    GenericPredicates,
    TraitRef,
    Const,
    Predicate,
    GeneratorDiagnosticData,
    Body,
    Node,
    ExportedSymbol,
}
