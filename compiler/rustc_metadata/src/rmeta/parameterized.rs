use std::hash::Hash;

use rustc_data_structures::stable_hash::StableHash;
use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::DefId;
use rustc_index::{Idx, IndexVec};
use rustc_middle::ty::{Binder, EarlyBinder, GenericArg, Region, Ty};
use rustc_serialize::Encodable;
use rustc_span::Symbol;

use crate::rmeta::encoder::EncodeContext;
use crate::rmeta::encoder::public_api_hasher::{PublicApiHashState, PublicApiHasher};
use crate::rmeta::remapped_def_id::{RemappedRawDefId, RemappedWideRawDefId};
use crate::rmeta::{LazyArray, LazyValue, LocalDefId, MetadataEncoder};

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
    crate::rmeta::TraitImpls<(u32, rustc_hashes::Hash64)>,
    crate::rmeta::TraitImpls<RemappedRawDefId>,
    crate::rmeta::VariantData,
    crate::rmeta::remapped_def_id::RemappedRawDefId,
    crate::rmeta::remapped_def_id::RemappedWideRawDefId,
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
    rustc_data_structures::fingerprint::Fingerprint,
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
pub(super) trait EncodeFrom<'tcx, T, M: MetadataEncoder<'tcx>> {
    /// When a `T` is trivial to encode, it has a fixed size, no pointers inside, this can be
    /// just a no-op. For more complex types this usually creates a `LazyValue<T>` or `LazyArray<T>`
    /// encoding `value` with `encoder` in the process
    fn encode_from<'a>(encoder: &mut EncodeContext<'a, 'tcx, M>, value: T) -> Self;

    fn encode_from_hashed<'a, 'h>(
        encoder: &mut EncodeContext<'a, 'tcx, M>,
        value: T,
        hcx: &mut impl PublicApiHashState<'h>,
        hasher: &mut PublicApiHasher,
    ) -> Self;
}

macro_rules! trivial_encode_from {
    (
        $($ty:ty,)+
    ) => {
        $(
            impl<'tcx, M: MetadataEncoder<'tcx>> EncodeFrom<'tcx, $ty, M> for $ty {
                fn encode_from<'a>(
                    _encoder: &mut EncodeContext<'a, 'tcx, M>,
                    value: $ty,
                ) -> $ty {
                    value
                }

                #[inline(always)]
                fn encode_from_hashed<'a, 'h>(
                    _encoder: &mut EncodeContext<'a, 'tcx, M>,
                    value: $ty,
                    hcx: &mut impl PublicApiHashState<'h>,
                    hasher: &mut PublicApiHasher,
                ) -> $ty {
                    hasher.digest(value, hcx);
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
            impl<'tcx, B, M> EncodeFrom<'tcx, B, M> for LazyValue<$ty>
            where
                B: std::borrow::Borrow<<$ty as ParameterizedOverTcx>::Value<'tcx>>,
                <$ty as ParameterizedOverTcx>::Value<'tcx>: StableHash,
                M: MetadataEncoder<'tcx>,
            {
                fn encode_from<'a>(
                    encoder: &mut EncodeContext<'a, 'tcx, M>,
                    value: B,
                ) -> LazyValue<$ty> {
                    encoder.lazy(value)
                }

                #[inline(always)]
                fn encode_from_hashed<'a, 'h>(
                    encoder: &mut EncodeContext<'a, 'tcx, M>,
                    value: B,
                    hcx: &mut impl PublicApiHashState<'h>,
                    hasher: &mut PublicApiHasher,
                ) -> LazyValue<$ty> {
                    hasher.digest(value.borrow(), hcx);
                    encoder.lazy(value)
                }
            }
        )+
    };
}

macro_rules! lazy_encode_from_unhashed {
    (
        $($ty:ty,)+
    ) => {
        $(
            impl<'tcx, B, M> EncodeFrom<'tcx, B, M> for LazyValue<$ty>
            where
                B: std::borrow::Borrow<<$ty as ParameterizedOverTcx>::Value<'tcx>>,
                M: MetadataEncoder<'tcx>,
            {
                fn encode_from<'a>(
                    encoder: &mut EncodeContext<'a, 'tcx, M>,
                    value: B,
                ) -> LazyValue<$ty> {
                    encoder.lazy(value)
                }

                #[inline(always)]
                fn encode_from_hashed<'a, 'h>(
                    _encoder: &mut EncodeContext<'a, 'tcx, M>,
                    _value: B,
                    _hcx: &mut impl PublicApiHashState<'h>,
                    _hasher: &mut PublicApiHasher,
                ) -> LazyValue<$ty> {
                    unreachable!()
                }
            }
        )+
    };
}

impl<'tcx, M: MetadataEncoder<'tcx>> EncodeFrom<'tcx, DefId, M> for RemappedWideRawDefId {
    fn encode_from<'a>(encoder: &mut EncodeContext<'a, 'tcx, M>, value: DefId) -> Self {
        M::into_remapped_wide_raw_def_id(encoder.tcx(), &encoder.local_defs_remapping, value)
    }

    #[inline(always)]
    fn encode_from_hashed<'a, 'h>(
        encoder: &mut EncodeContext<'a, 'tcx, M>,
        value: DefId,
        hcx: &mut impl PublicApiHashState<'h>,
        hasher: &mut PublicApiHasher,
    ) -> Self {
        hasher.digest(value, hcx);
        M::into_remapped_wide_raw_def_id(encoder.tcx(), &encoder.local_defs_remapping, value)
    }
}

trivial_encode_from! {
    bool,
    u64,
    // tidy-alphabetical-start
    crate::rmeta::AttrFlags,
    rustc_data_structures::fingerprint::Fingerprint,
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
    (rustc_span::hygiene::SyntaxContext, rustc_span::ExpnId, rustc_span::hygiene::Transparency),
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
    rustc_span::ExpnData,
    rustc_span::ExpnHash,
    rustc_span::ExpnId,
    rustc_span::Span,
    std::option::Option<rustc_middle::ty::EarlyBinder<'static, rustc_middle::ty::Const<'static>>>,
    std::string::String,
    std::vec::Vec<(usize, rustc_index::bit_set::DenseBitSet<u32>)>,
    // tidy-alphabetical-end
}

lazy_encode_from_unhashed! {
    rustc_hir::definitions::DefKey,
    rustc_span::SourceFile,
}

impl<'tcx, T, B, M: MetadataEncoder<'tcx>> EncodeFrom<'tcx, Option<B>, M> for Option<LazyValue<T>>
where
    T: ParameterizedOverTcx,
    T::Value<'tcx>: for<'b> Encodable<EncodeContext<'b, 'tcx, M>>,
    B: std::borrow::Borrow<T::Value<'tcx>> + StableHash,
    M: MetadataEncoder<'tcx>,
{
    fn encode_from<'a>(
        encoder: &mut EncodeContext<'a, 'tcx, M>,
        value: Option<B>,
    ) -> Option<LazyValue<T>> {
        value.map(|v| encoder.lazy(v))
    }

    #[inline(always)]
    fn encode_from_hashed<'a, 'h>(
        encoder: &mut EncodeContext<'a, 'tcx, M>,
        value: Option<B>,
        hcx: &mut impl PublicApiHashState<'h>,
        hasher: &mut PublicApiHasher,
    ) -> Self {
        hasher.digest(&value, hcx);
        value.map(|v| encoder.lazy(v))
    }
}

impl<'tcx, T, I, B, M> EncodeFrom<'tcx, I, M> for LazyArray<T>
where
    T: ParameterizedOverTcx,
    I: IntoIterator<Item = B>,
    B: std::borrow::Borrow<T::Value<'tcx>> + StableHash,
    T::Value<'tcx>: for<'a> Encodable<EncodeContext<'a, 'tcx, M>>,
    M: MetadataEncoder<'tcx>,
{
    fn encode_from<'a>(encoder: &mut EncodeContext<'a, 'tcx, M>, values: I) -> Self {
        encoder.lazy_array(values)
    }

    #[inline(always)]
    fn encode_from_hashed<'a, 'h>(
        encoder: &mut EncodeContext<'a, 'tcx, M>,
        values: I,
        hcx: &mut impl PublicApiHashState<'h>,
        hasher: &mut PublicApiHasher,
    ) -> Self {
        encoder.lazy_array(values.into_iter().inspect(|b| hasher.digest(b, hcx)))
    }
}
