use std::hash::Hash;

use rustc_data_structures::unord::UnordMap;
use rustc_index::{Idx, IndexVec};
use rustc_middle::ty::{Binder, EarlyBinder, GenericArg, Region, Ty};
use rustc_serialize::Encodable;
use rustc_span::Symbol;

use crate::rmeta::encoder::EncodeContext;
use crate::rmeta::{LazyArray, LazyValue, LocalDefId};

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

impl ParameterizedOverTcx for Region<'static> {
    type Value<'tcx> = Region<'tcx>;
}

impl ParameterizedOverTcx for GenericArg<'static> {
    type Value<'tcx> = GenericArg<'tcx>;
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
    u32,
    u64,
    usize,
    std::string::String,
    // tidy-alphabetical-start
    crate::rmeta::AttrFlags,
    crate::rmeta::CrateDep,
    crate::rmeta::CrateHeader,
    crate::rmeta::CrateRoot,
    crate::rmeta::CrateRootUnhashed,
    crate::rmeta::IncoherentImpls,
    crate::rmeta::LocalDefId,
    crate::rmeta::ProcMacroKind,
    crate::rmeta::TraitImpls,
    crate::rmeta::VariantData,
    crate::rmeta::remapped_def_id::RemappedRawDefId,
    rustc_abi::ReprOptions,
    rustc_ast::DelimArgs,
    rustc_attr_ir::Attribute,
    rustc_attr_ir::ConstStability,
    rustc_attr_ir::DefaultBodyStability,
    rustc_attr_ir::Deprecation,
    rustc_attr_ir::EiiDecl,
    rustc_attr_ir::EiiImpl,
    rustc_attr_ir::Stability,
    rustc_attr_ir::StrippedCfgItem<LocalDefId>,
    rustc_attr_ir::lang_items::LangItem,
    rustc_crate_store::ForeignModule,
    rustc_crate_store::LinkagePreference,
    rustc_crate_store::NativeLib,
    rustc_hir::Constness,
    rustc_hir::CoroutineKind,
    rustc_hir::Defaultness,
    rustc_hir::OpaqueTyOrigin<rustc_hir::def_id::DefId>,
    rustc_hir::PreciseCapturingArgKind<Symbol, Symbol>,
    rustc_hir::Safety,
    rustc_hir::def::DefKind,
    rustc_hir::def_id::DefId,
    rustc_hir::def_id::DefIndex,
    rustc_hir::definitions::DefKey,
    rustc_index::bit_set::DenseBitSet<u32>,
    rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs,
    rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile,
    rustc_middle::middle::deduced_param_attrs::DeducedParamAttrs,
    rustc_middle::middle::exported_symbols::SymbolExportInfo,
    rustc_middle::middle::lib_features::FeatureStability,
    rustc_middle::middle::resolve::AmbigModChild,
    rustc_middle::middle::resolve::DocLinkResMap,
    rustc_middle::middle::resolve::ModChild,
    rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault,
    rustc_middle::mir::ConstQualifs,
    rustc_middle::mir::ConstValue,
    rustc_middle::ty::AnonConstKind,
    rustc_middle::ty::AssocContainer,
    rustc_middle::ty::AsyncDestructor,
    rustc_middle::ty::Asyncness,
    rustc_middle::ty::Destructor,
    rustc_middle::ty::Generics,
    rustc_middle::ty::ImplTraitInTraitData,
    rustc_middle::ty::IntrinsicDef,
    rustc_middle::ty::RestrictionKind,
    rustc_middle::ty::TraitDef,
    rustc_middle::ty::Variance,
    rustc_middle::ty::Visibility<LocalDefId>,
    rustc_middle::ty::adjustment::CoerceUnsizedInfo,
    rustc_middle::ty::fast_reject::SimplifiedType,
    rustc_session::config::TargetModifier,
    rustc_session::config::mitigation_coverage::DeniedPartialMitigation,
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
    rustc_middle::ty::GenericClauses,
    rustc_middle::ty::ImplTraitHeader,
    rustc_middle::ty::TraitRef,
    rustc_middle::ty::Ty,
    // tidy-alphabetical-end
}

/// Converts values from `TyCtxt` stored form into metadata table stored form
pub(super) trait EncodeFrom<'tcx, T> {
    /// When a `T` is trivial to encode, it has a fixed size, no pointers inside, this can be
    /// just a no-op. For more complex types this usually creates a `LazyValue<T>` or `LazyArray<T>`
    /// encoding `value` with `encoder` in the process
    fn encode_from<'a>(encoder: &mut EncodeContext<'a, 'tcx>, value: T) -> Self;
}

macro_rules! trivial_encode_from {
    (
        $($ty:ty,)+
    ) => {
        $(
            impl<'tcx> EncodeFrom<'tcx, $ty> for $ty {
                fn encode_from<'a>(
                    _encoder: &mut EncodeContext<'a, 'tcx>,
                    value: $ty,
                ) -> $ty {
                    value
                }
            }
        )+
    };
}

macro_rules! lazy_encode_from {
    (
        $($ty:ty,)+
    ) => {
        $(
            impl<'tcx, B> EncodeFrom<'tcx, B> for LazyValue<$ty>
            where
                B: std::borrow::Borrow<<$ty as ParameterizedOverTcx>::Value<'tcx>>
            {
                fn encode_from<'a>(
                    encoder: &mut EncodeContext<'a, 'tcx>,
                    value: B,
                ) -> LazyValue<$ty> {
                    encoder.lazy(value)
                }
            }
        )+
    };
}

trivial_encode_from! {
    bool,
    u64,
    // tidy-alphabetical-start
    crate::rmeta::AttrFlags,
    crate::rmeta::remapped_def_id::RemappedRawDefId,
    rustc_hir::Constness,
    rustc_hir::CoroutineKind,
    rustc_hir::Defaultness,
    rustc_hir::Safety,
    rustc_hir::def::DefKind,
    rustc_middle::ty::Asyncness,
    // tidy-alphabetical-end
}

lazy_encode_from! {
    // tidy-alphabetical-start
    (rustc_middle::mir::ConstValue, Ty<'static>),
    crate::rmeta::VariantData,
    rustc_abi::ReprOptions,
    rustc_ast::DelimArgs,
    rustc_attr_ir::ConstStability,
    rustc_attr_ir::DefaultBodyStability,
    rustc_attr_ir::Deprecation,
    rustc_attr_ir::Stability,
    rustc_hir::OpaqueTyOrigin<rustc_span::def_id::DefId>,
    rustc_hir::def_id::DefId,
    rustc_hir::def_id::DefIdMap<rustc_middle::ty::EarlyBinder<'static, Ty<'static>>>,
    rustc_hir::def_id::DefIdMap<std::vec::Vec<rustc_hir::def_id::DefId>>,
    rustc_hir::definitions::DefKey,
    rustc_index::IndexVec<rustc_abi::FieldIdx, rustc_span::Symbol>,
    rustc_index::IndexVec<rustc_middle::mir::Promoted, rustc_middle::mir::Body<'static>>,
    rustc_index::bit_set::DenseBitSet<u32>,
    rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs,
    rustc_middle::middle::resolve::DocLinkResMap,
    rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault,
    rustc_middle::mir::Body<'static>,
    rustc_middle::mir::ConstQualifs,
    rustc_middle::mir::CoroutineLayout<'static>,
    rustc_middle::mir::interpret::ConstAllocation<'static>,
    rustc_middle::ty::AnonConstKind,
    rustc_middle::ty::AssocContainer,
    rustc_middle::ty::AsyncDestructor,
    rustc_middle::ty::ConstConditions<'static>,
    rustc_middle::ty::Destructor,
    rustc_middle::ty::EarlyBinder<'static, Ty<'static>>,
    rustc_middle::ty::EarlyBinder<'static, rustc_middle::ty::Const<'static>>,
    rustc_middle::ty::EarlyBinder<'static, rustc_middle::ty::PolyFnSig<'static>>,
    rustc_middle::ty::GenericClauses<'static>,
    rustc_middle::ty::Generics,
    rustc_middle::ty::ImplTraitHeader<'static>,
    rustc_middle::ty::ImplTraitInTraitData,
    rustc_middle::ty::IntrinsicDef,
    rustc_middle::ty::RestrictionKind,
    rustc_middle::ty::Visibility<rustc_span::def_id::LocalDefId>,
    rustc_middle::ty::adjustment::CoerceUnsizedInfo,
    rustc_middle::ty::trait_def::TraitDef,
    rustc_span::ExpnId,
    rustc_span::Span,
    std::option::Option<rustc_middle::ty::EarlyBinder<'static, rustc_middle::ty::Const<'static>>>,
    std::string::String,
    std::vec::Vec<(usize, rustc_index::bit_set::DenseBitSet<u32>)>,
    // tidy-alphabetical-end
}

impl<'tcx, T, B> EncodeFrom<'tcx, Option<B>> for Option<LazyValue<T>>
where
    T: ParameterizedOverTcx,
    T::Value<'tcx>: for<'b> Encodable<EncodeContext<'b, 'tcx>>,
    B: std::borrow::Borrow<T::Value<'tcx>>,
{
    fn encode_from<'a>(
        encoder: &mut EncodeContext<'a, 'tcx>,
        value: Option<B>,
    ) -> Option<LazyValue<T>> {
        value.map(|v| encoder.lazy(v))
    }
}

impl<'tcx, T, I, B> EncodeFrom<'tcx, I> for LazyArray<T>
where
    T: ParameterizedOverTcx,
    I: IntoIterator<Item = B>,
    B: std::borrow::Borrow<T::Value<'tcx>>,
    T::Value<'tcx>: for<'a> Encodable<EncodeContext<'a, 'tcx>>,
{
    fn encode_from<'a>(encoder: &mut EncodeContext<'a, 'tcx>, values: I) -> Self {
        encoder.lazy_array(values)
    }
}
