use std::marker::PhantomData;
use std::num::NonZero;

use decoder::LazyDecoder;
pub(crate) use decoder::{CrateMetadata, CrateNumMap, MetadataBlob, TargetModifiers};
use def_path_hash_map::DefPathHashMapRef;
use encoder::EncodeContext;
pub use encoder::{EncodedMetadata, encode_metadata, rendered_const};
pub(crate) use parameterized::ParameterizedOverTcx;
use rustc_abi::{FieldIdx, ReprOptions, VariantIdx};
use rustc_ast as ast;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::svh::Svh;
use rustc_hir as hir;
use rustc_hir::attrs::StrippedCfgItem;
use rustc_hir::def::{CtorKind, DefKind, DocLinkResMap, MacroKinds};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, DefIndex, DefPathHash, StableCrateId};
use rustc_hir::definitions::DefKey;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{PreciseCapturingArgKind, attrs};
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_macros::{
    BlobDecodable, Decodable, Encodable, LazyDecodable, MetadataEncodable, StableHash, TyDecodable,
    TyEncodable,
};
use rustc_middle::metadata::{AmbigModChild, ModChild};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile;
use rustc_middle::middle::deduced_param_attrs::DeducedParamAttrs;
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo};
use rustc_middle::middle::lib_features::FeatureStability;
use rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault;
use rustc_middle::mir;
use rustc_middle::mir::ConstValue;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::util::Providers;
use rustc_serialize::opaque::FileEncoder;
use rustc_session::config::mitigation_coverage::DeniedPartialMitigation;
use rustc_session::config::{SymbolManglingVersion, TargetModifier};
use rustc_session::cstore::{CrateDepKind, ForeignModule, LinkagePreference, NativeLib};
use rustc_span::edition::Edition;
use rustc_span::hygiene::{ExpnIndex, MacroKind, SyntaxContextKey};
use rustc_span::{self, ExpnData, ExpnHash, ExpnId, Ident, Span, Symbol};
use rustc_target::spec::{PanicStrategy, TargetTuple};
use table::TableBuilder;

use crate::eii::EiiMapEncodedKeyValue;
use crate::rmeta::encoder::public_api_hasher::{
    Hashed, PublicApiHasher, PublicApiHashingContext, RDRHashAll, RDRHashNone,
};

mod decoder;
mod def_path_hash_map;
mod encoder;
mod parameterized;
mod table;

pub(crate) fn rustc_version(cfg_version: &'static str) -> String {
    format!("rustc {cfg_version}")
}

/// Metadata encoding version.
/// N.B., increment this if you change the format of metadata such that
/// the rustc version can't be found to compare with `rustc_version()`.
const METADATA_VERSION: u8 = 10;

/// Metadata header which includes `METADATA_VERSION`.
///
/// This header is followed by the length of the compressed data, then
/// the position of the `CrateRoot`, which is encoded as a 64-bit little-endian
/// unsigned integer, and further followed by the rustc version string.
pub const METADATA_HEADER: &[u8] = &[b'r', b'u', b's', b't', 0, 0, 0, METADATA_VERSION];

/// A value of type T referred to by its absolute position
/// in the metadata, and which can be decoded lazily.
///
/// Metadata is effective a tree, encoded in post-order,
/// and with the root's position written next to the header.
/// That means every single `LazyValue` points to some previous
/// location in the metadata and is part of a larger node.
///
/// The first `LazyValue` in a node is encoded as the backwards
/// distance from the position where the containing node
/// starts and where the `LazyValue` points to, while the rest
/// use the forward distance from the previous `LazyValue`.
/// Distances start at 1, as 0-byte nodes are invalid.
/// Also invalid are nodes being referred in a different
/// order than they were encoded in.
#[must_use]
struct LazyValue<T> {
    position: NonZero<usize>,
    _marker: PhantomData<fn() -> T>,
}

impl<T> LazyValue<T> {
    fn from_position(position: NonZero<usize>) -> LazyValue<T> {
        LazyValue { position, _marker: PhantomData }
    }
}

/// A list of lazily-decoded values.
///
/// Unlike `LazyValue<Vec<T>>`, the length is encoded next to the
/// position, not at the position, which means that the length
/// doesn't need to be known before encoding all the elements.
///
/// If the length is 0, no position is encoded, but otherwise,
/// the encoding is that of `LazyArray`, with the distinction that
/// the minimal distance the length of the sequence, i.e.
/// it's assumed there's no 0-byte element in the sequence.
struct LazyArray<T> {
    position: NonZero<usize>,
    num_elems: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Default for LazyArray<T> {
    fn default() -> LazyArray<T> {
        LazyArray::from_position_and_num_elems(NonZero::new(1).unwrap(), 0)
    }
}

impl<T> LazyArray<T> {
    fn from_position_and_num_elems(position: NonZero<usize>, num_elems: usize) -> LazyArray<T> {
        LazyArray { position, num_elems, _marker: PhantomData }
    }
}

/// A list of lazily-decoded values, with the added capability of random access.
///
/// Random-access table (i.e. offering constant-time `get`/`set`), similar to
/// `LazyArray<T>`, but without requiring encoding or decoding all the values
/// eagerly and in-order.
struct LazyTable<I, T> {
    position: NonZero<usize>,
    /// The encoded size of the elements of a table is selected at runtime to drop
    /// trailing zeroes. This is the number of bytes used for each table element.
    width: usize,
    /// How many elements are in the table.
    len: usize,
    _marker: PhantomData<fn(I) -> T>,
}

impl<I, T> LazyTable<I, T> {
    fn from_position_and_encoded_size(
        position: NonZero<usize>,
        width: usize,
        len: usize,
    ) -> LazyTable<I, T> {
        LazyTable { position, width, len, _marker: PhantomData }
    }
}

impl<T> Copy for LazyValue<T> {}
impl<T> Clone for LazyValue<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for LazyArray<T> {}
impl<T> Clone for LazyArray<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I, T> Copy for LazyTable<I, T> {}
impl<I, T> Clone for LazyTable<I, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Encoding / decoding state for `Lazy`s (`LazyValue`, `LazyArray`, and `LazyTable`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum LazyState {
    /// Outside of a metadata node.
    NoNode,

    /// Inside a metadata node, and before any `Lazy`s.
    /// The position is that of the node itself.
    NodeStart(NonZero<usize>),

    /// Inside a metadata node, with a previous `Lazy`s.
    /// The position is where that previous `Lazy` would start.
    Previous(NonZero<usize>),
}

type SyntaxContextTable = LazyTable<u32, Option<LazyValue<SyntaxContextKey>>>;
type ExpnDataTable = LazyTable<ExpnIndex, Option<LazyValue<ExpnData>>>;
type ExpnHashTable = LazyTable<ExpnIndex, Option<LazyValue<ExpnHash>>>;

#[derive(MetadataEncodable, LazyDecodable)]
pub(crate) struct ProcMacroData {
    proc_macro_decls_static: DefIndex,
    stability: Option<hir::Stability>,
    macros: LazyArray<DefIndex>,
}

/// Serialized crate metadata.
///
/// This contains just enough information to determine if we should load the `CrateRoot` or not.
/// Prefer [`CrateRoot`] whenever possible to avoid ICEs when using `omit-git-hash` locally.
/// See #76720 for more details.
///
/// If you do modify this struct, also bump the [`METADATA_VERSION`] constant.
#[derive(MetadataEncodable, BlobDecodable)]
pub(crate) struct CrateHeader {
    pub(crate) triple: TargetTuple,
    pub(crate) hashes: CrateHashes,
    pub(crate) name: Symbol,
    /// Whether this is the header for a proc-macro crate.
    ///
    /// This is separate from [`ProcMacroData`] to avoid having to update [`METADATA_VERSION`] every
    /// time ProcMacroData changes.
    pub(crate) is_proc_macro_crate: bool,
    /// Whether this crate metadata section is just a stub.
    /// Stubs do not contain the full metadata (it will be typically stored
    /// in a separate rmeta file).
    ///
    /// This is used inside rlibs and dylibs when using `-Zembed-metadata=no`.
    pub(crate) is_stub: bool,
}

/// Serialized `.rmeta` data for a crate.
///
/// When compiling a proc-macro crate, we encode many of
/// the `LazyArray<T>` fields as `Lazy::empty()`. This serves two purposes:
///
/// 1. We avoid performing unnecessary work. Proc-macro crates can only
/// export proc-macros functions, which are compiled into a shared library.
/// As a result, a large amount of the information we normally store
/// (e.g. optimized MIR) is unneeded by downstream crates.
/// 2. We avoid serializing invalid `CrateNum`s. When we deserialize
/// a proc-macro crate, we don't load any of its dependencies (since we
/// just need to invoke a native function from the shared library).
/// This means that any foreign `CrateNum`s that we serialize cannot be
/// deserialized, since we will not know how to map them into the current
/// compilation session. If we were to serialize a proc-macro crate like
/// a normal crate, much of what we serialized would be unusable in addition
/// to being unused.
#[derive(MetadataEncodable, LazyDecodable)]
pub(crate) struct CrateRoot {
    /// A header used to detect if this is the right crate to load.
    header: CrateHeader,

    extra_filename: String,
    stable_crate_id: StableCrateId,
    required_panic_strategy: Option<PanicStrategy>,
    panic_in_drop_strategy: PanicStrategy,
    edition: Edition,
    has_global_allocator: bool,
    has_alloc_error_handler: bool,
    has_panic_handler: bool,
    has_default_lib_allocator: bool,
    externally_implementable_items: LazyArray<EiiMapEncodedKeyValue>,

    crate_deps: LazyArray<CrateDep>,
    dylib_dependency_formats: LazyArray<Option<LinkagePreference>>,
    lib_features: LazyArray<(Symbol, FeatureStability)>,
    stability_implications: LazyArray<(Symbol, Symbol)>,
    lang_items: LazyArray<(DefIndex, LangItem)>,
    lang_items_missing: LazyArray<LangItem>,
    stripped_cfg_items: LazyArray<StrippedCfgItem<DefIndex>>,
    diagnostic_items: LazyArray<(Symbol, DefIndex)>,
    native_libraries: LazyArray<NativeLib>,
    foreign_modules: LazyArray<ForeignModule>,
    traits: LazyArray<DefIndex>,
    impls: LazyArray<TraitImpls>,
    incoherent_impls: LazyArray<IncoherentImpls>,
    interpret_alloc_index: LazyArray<u64>,
    proc_macro_data: Option<ProcMacroData>,

    tables: LazyTables,
    debugger_visualizers: LazyArray<DebuggerVisualizerFile>,

    exportable_items: LazyArray<DefIndex>,
    stable_order_of_exportable_impls: LazyArray<(DefIndex, usize)>,
    exported_non_generic_symbols: LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>,
    exported_generic_symbols: LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>,

    syntax_contexts: SyntaxContextTable,
    expn_data: ExpnDataTable,
    expn_hashes: ExpnHashTable,

    def_path_hash_map: LazyValue<DefPathHashMapRef<'static>>,

    source_map: LazyTable<u32, Option<LazyValue<rustc_span::SourceFile>>>,
    target_modifiers: LazyArray<TargetModifier>,
    denied_partial_mitigations: LazyArray<DeniedPartialMitigation>,

    compiler_builtins: bool,
    needs_allocator: bool,
    needs_panic_runtime: bool,
    no_builtins: bool,
    panic_runtime: bool,
    profiler_runtime: bool,
    symbol_mangling_version: SymbolManglingVersion,

    specialization_enabled_in: bool,
}

/// All hashes here are equal to the hash from the crate header (the `crate_hash` query) when the public-api-hash unstable feature is disabled.
#[derive(MetadataEncodable, BlobDecodable, Clone, Copy, Eq, PartialEq, Hash)]
pub(crate) struct CrateHashes {
    /// Hash of the crate contents, including private items. For proc macros this includes the
    /// private hashes of all dependencies. When `public-api-hash` is enabled, for other crate
    /// types than proc macro, it only includes the public hash of dependencies. This is only
    /// readable by queries in downstream dependencies if the crate querying is a proc macro.
    pub(crate) private_hash: Svh,
    /// Hash of most data in rmeta. same as `private_hash` if the `public-api-hash` option is
    /// disabled.
    ///
    /// The public hash contains `StableCrateId`, so two crates in the dependency graph should not
    /// have the same public hash just because they have the same "public api". This is asserted
    /// while loading: if two crates have the same public hash but different private hashes, the
    /// resolver reports that there are multiple candidates available for a crate and compilation
    /// aborts.
    pub(crate) public_hash: Svh,
}

/// On-disk representation of `DefId`.
/// This creates a type-safe way to enforce that we remap the CrateNum between the on-disk
/// representation and the compilation session.
#[derive(Copy, Clone)]
pub(crate) struct RawDefId {
    krate: u32,
    index: u32,
}

impl From<DefId> for RawDefId {
    fn from(val: DefId) -> Self {
        RawDefId { krate: val.krate.as_u32(), index: val.index.as_u32() }
    }
}

impl RawDefId {
    /// This exists so that `provide_one!` is happy
    fn decode(self, meta: (&CrateMetadata, TyCtxt<'_>)) -> DefId {
        let krate = CrateNum::from_u32(self.krate);
        let krate = meta.0.map_encoded_cnum_to_current(krate);
        DefId { krate, index: DefIndex::from_u32(self.index) }
    }
}

#[derive(Encodable, BlobDecodable, StableHash)]
pub(crate) struct CrateDep {
    pub name: Symbol,
    pub hash: Svh,
    pub host_hash: Option<Svh>,
    pub kind: CrateDepKind,
    pub extra_filename: String,
    pub is_private: bool,
}

#[derive(MetadataEncodable, LazyDecodable)]
pub(crate) struct TraitImpls {
    trait_id: (u32, DefIndex),
    impls: LazyArray<(DefIndex, Option<SimplifiedType>)>,
}

#[derive(MetadataEncodable, LazyDecodable)]
pub(crate) struct IncoherentImpls {
    self_ty: LazyValue<SimplifiedType>,
    impls: LazyArray<DefIndex>,
}

/// Define `LazyTables` and `TableBuilders` at the same time.
macro_rules! define_tables {
    (
        - defaulted: $($name1:ident: Table<$HASH1:ident, $IDX1:ty, $T1:ty>,)+
        - optional: $($name2:ident: Table<$HASH2:ident, $IDX2:ty, $T2:ty>,)+
    ) => {
        #[derive(MetadataEncodable, LazyDecodable)]
        pub(crate) struct LazyTables {
            $($name1: LazyTable<$IDX1, $T1>,)+
            $($name2: LazyTable<$IDX2, Option<$T2>>,)+
        }

        #[derive(Default)]
        struct TableBuilders {
            $($name1: TableBuilder<$HASH1<$IDX1>, $IDX1, $T1>,)+
            $($name2: TableBuilder<$HASH2<$IDX2>, $IDX2, Option<$T2>>,)+
        }

        impl TableBuilders {
            fn encode(
                &self,
                buf: &mut FileEncoder,
                hcx: &mut PublicApiHashingContext<'_>
            ) -> Hashed<LazyTables>
            {
                let mut hasher = PublicApiHasher::default();
                let tables = LazyTables {
                    $($name1: {
                        let table = self.$name1.encode(buf, hcx);
                        if let Some(hash) = table.hash {
                            tracing::debug!("{}: {hash:x?}", stringify!($name1));
                        }
                        hasher.digest(&table, hcx);
                        table.value
                    },)+
                    $($name2: {
                        let table = self.$name2.encode(buf, hcx);
                        if let Some(hash) = table.hash {
                            tracing::debug!("{}: {hash:x?}", stringify!($name2));
                        }
                        hasher.digest(&table, hcx);
                        table.value
                    },)+
                };
                Hashed { hash: hasher.finish(hcx), value: tables }
            }
        }
    }
}

// When adding a new field, use `RDRHashAll` as the hasher and include a `// FIXME do we need to
// hash this comment?` to note that it wasn't reviewed for public api hashing. The docs of
// `HashableCrateHeader` contains more information about public api hashing.
define_tables! {
- defaulted:
    // FIXME do we need to hash this?
    intrinsic: Table<RDRHashAll, DefIndex, Option<LazyValue<ty::IntrinsicDef>>>,
    // FIXME do we need to hash this?
    is_macro_rules: Table<RDRHashAll, DefIndex, bool>,
    // FIXME do we need to hash this?
    type_alias_is_lazy: Table<RDRHashAll, DefIndex, bool>,
    // FIXME do we need to hash this?
    attr_flags: Table<RDRHashAll, DefIndex, AttrFlags>,
    // The u64 is the crate-local part of the DefPathHash. All hashes in this crate have the same
    // StableCrateId, so we omit encoding those into the table.
    //
    // Note also that this table is fully populated (no gaps) as every DefIndex should have a
    // corresponding DefPathHash.
    //
    // We don't need to include this in the hash, hashing `def_path_hash_map` takes care of it.
    def_path_hashes: Table<RDRHashNone, DefIndex, u64>,
    // FIXME do we need to hash this?
    explicit_item_bounds: Table<RDRHashAll, DefIndex, LazyArray<(ty::Clause<'static>, Span)>>,
    // FIXME do we need to hash this?
    explicit_item_self_bounds: Table<RDRHashAll, DefIndex, LazyArray<(ty::Clause<'static>, Span)>>,
    // FIXME do we need to hash this?
    inferred_outlives_of: Table<RDRHashAll, DefIndex, LazyArray<(ty::Clause<'static>, Span)>>,
    // FIXME do we need to hash this?
    explicit_super_predicates_of: Table<RDRHashAll, DefIndex, LazyArray<(ty::Clause<'static>, Span)>>,
    // FIXME do we need to hash this?
    explicit_implied_predicates_of: Table<RDRHashAll, DefIndex, LazyArray<(ty::Clause<'static>, Span)>>,
    // FIXME do we need to hash this?
    explicit_implied_const_bounds: Table<RDRHashAll, DefIndex, LazyArray<(ty::PolyTraitRef<'static>, Span)>>,
    // FIXME do we need to hash this?
    inherent_impls: Table<RDRHashAll, DefIndex, LazyArray<DefIndex>>,
    // FIXME do we need to hash this?
    opt_rpitit_info: Table<RDRHashAll, DefIndex, Option<LazyValue<ty::ImplTraitInTraitData>>>,
    // Reexported names are not associated with individual `DefId`s,
    // e.g. a glob import can introduce a lot of names, all with the same `DefId`.
    // That's why the encoded list needs to contain `ModChild` structures describing all the names
    // individually instead of `DefId`s.
    // FIXME do we need to hash this?
    module_children_reexports: Table<RDRHashAll, DefIndex, LazyArray<ModChild>>,
    // FIXME do we need to hash this?
    ambig_module_children: Table<RDRHashAll, DefIndex, LazyArray<AmbigModChild>>,
    // FIXME do we need to hash this?
    cross_crate_inlinable: Table<RDRHashAll, DefIndex, bool>,
    // FIXME do we need to hash this?
    asyncness: Table<RDRHashAll, DefIndex, ty::Asyncness>,
    // FIXME do we need to hash this?
    constness: Table<RDRHashAll, DefIndex, hir::Constness>,
    // FIXME do we need to hash this?
    safety: Table<RDRHashAll, DefIndex, hir::Safety>,
    // FIXME do we need to hash this?
    defaultness: Table<RDRHashAll, DefIndex, hir::Defaultness>,

- optional:
    // FIXME do we need to hash this?
    attributes: Table<RDRHashAll, DefIndex, LazyArray<hir::Attribute>>,
    // For non-reexported names in a module every name is associated with a separate `DefId`,
    // so we can take their names, visibilities etc from other encoded tables.
    // FIXME do we need to hash this?
    module_children_non_reexports: Table<RDRHashAll, DefIndex, LazyArray<DefIndex>>,
    // FIXME do we need to hash this?
    associated_item_or_field_def_ids: Table<RDRHashAll, DefIndex, LazyArray<DefIndex>>,
    // FIXME do we need to hash this?
    def_kind: Table<RDRHashAll, DefIndex, DefKind>,
    // FIXME do we need to hash this?
    visibility: Table<RDRHashAll, DefIndex, LazyValue<ty::Visibility<DefIndex>>>,
    // FIXME do we need to hash this?
    def_span: Table<RDRHashAll, DefIndex, LazyValue<Span>>,
    // FIXME do we need to hash this?
    def_ident_span: Table<RDRHashAll, DefIndex, LazyValue<Span>>,
    // FIXME do we need to hash this?
    lookup_stability: Table<RDRHashAll, DefIndex, LazyValue<hir::Stability>>,
    // FIXME do we need to hash this?
    lookup_const_stability: Table<RDRHashAll, DefIndex, LazyValue<hir::ConstStability>>,
    // FIXME do we need to hash this?
    lookup_default_body_stability: Table<RDRHashAll, DefIndex, LazyValue<hir::DefaultBodyStability>>,
    // FIXME do we need to hash this?
    lookup_deprecation_entry: Table<RDRHashAll, DefIndex, LazyValue<attrs::Deprecation>>,
    // FIXME do we need to hash this?
    explicit_predicates_of: Table<RDRHashAll, DefIndex, LazyValue<ty::GenericPredicates<'static>>>,
    // FIXME do we need to hash this?
    generics_of: Table<RDRHashAll, DefIndex, LazyValue<ty::Generics>>,
    // FIXME do we need to hash this?
    type_of: Table<RDRHashAll, DefIndex, LazyValue<ty::EarlyBinder<'static, Ty<'static>>>>,
    // FIXME do we need to hash this?
    variances_of: Table<RDRHashAll, DefIndex, LazyArray<ty::Variance>>,
    // FIXME do we need to hash this?
    fn_sig: Table<RDRHashAll, DefIndex, LazyValue<ty::EarlyBinder<'static, ty::PolyFnSig<'static>>>>,
    // FIXME do we need to hash this?
    codegen_fn_attrs: Table<RDRHashAll, DefIndex, LazyValue<CodegenFnAttrs>>,
    // FIXME do we need to hash this?
    impl_trait_header: Table<RDRHashAll, DefIndex, LazyValue<ty::ImplTraitHeader<'static>>>,
    // FIXME do we need to hash this?
    const_param_default: Table<RDRHashAll, DefIndex, LazyValue<ty::EarlyBinder<'static, rustc_middle::ty::Const<'static>>>>,
    // FIXME do we need to hash this?
    object_lifetime_default: Table<RDRHashAll, DefIndex, LazyValue<ObjectLifetimeDefault>>,
    // FIXME do we need to hash this?
    optimized_mir: Table<RDRHashAll, DefIndex, LazyValue<mir::Body<'static>>>,
    // FIXME do we need to hash this?
    mir_for_ctfe: Table<RDRHashAll, DefIndex, LazyValue<mir::Body<'static>>>,
    // FIXME do we need to hash this?
    trivial_const: Table<RDRHashAll, DefIndex, LazyValue<(ConstValue, Ty<'static>)>>,
    // FIXME do we need to hash this?
    closure_saved_names_of_captured_variables: Table<RDRHashAll, DefIndex, LazyValue<IndexVec<FieldIdx, Symbol>>>,
    // FIXME do we need to hash this?
    mir_coroutine_witnesses: Table<RDRHashAll, DefIndex, LazyValue<mir::CoroutineLayout<'static>>>,
    // FIXME do we need to hash this?
    promoted_mir: Table<RDRHashAll, DefIndex, LazyValue<IndexVec<mir::Promoted, mir::Body<'static>>>>,
    // FIXME do we need to hash this?
    thir_abstract_const: Table<RDRHashAll, DefIndex, LazyValue<ty::EarlyBinder<'static, ty::Const<'static>>>>,
    // FIXME do we need to hash this?
    impl_parent: Table<RDRHashAll, DefIndex, RawDefId>,
    // FIXME do we need to hash this?
    const_conditions: Table<RDRHashAll, DefIndex, LazyValue<ty::ConstConditions<'static>>>,
    // FIXME(eddyb) perhaps compute this on the fly if cheap enough?
    // FIXME do we need to hash this?
    coerce_unsized_info: Table<RDRHashAll, DefIndex, LazyValue<ty::adjustment::CoerceUnsizedInfo>>,
    // FIXME do we need to hash this?
    mir_const_qualif: Table<RDRHashAll, DefIndex, LazyValue<mir::ConstQualifs>>,
    // FIXME do we need to hash this?
    rendered_const: Table<RDRHashAll, DefIndex, LazyValue<String>>,
    // FIXME do we need to hash this?
    rendered_precise_capturing_args: Table<RDRHashAll, DefIndex, LazyArray<PreciseCapturingArgKind<Symbol, Symbol>>>,
    // FIXME do we need to hash this?
    fn_arg_idents: Table<RDRHashAll, DefIndex, LazyArray<Option<Ident>>>,
    // FIXME do we need to hash this?
    coroutine_kind: Table<RDRHashAll, DefIndex, hir::CoroutineKind>,
    // FIXME do we need to hash this?
    coroutine_for_closure: Table<RDRHashAll, DefIndex, RawDefId>,
    // FIXME do we need to hash this?
    adt_destructor: Table<RDRHashAll, DefIndex, LazyValue<ty::Destructor>>,
    // FIXME do we need to hash this?
    adt_async_destructor: Table<RDRHashAll, DefIndex, LazyValue<ty::AsyncDestructor>>,
    // FIXME do we need to hash this?
    coroutine_by_move_body_def_id: Table<RDRHashAll, DefIndex, RawDefId>,
    // FIXME do we need to hash this?
    eval_static_initializer: Table<RDRHashAll, DefIndex, LazyValue<mir::interpret::ConstAllocation<'static>>>,
    // FIXME do we need to hash this?
    trait_def: Table<RDRHashAll, DefIndex, LazyValue<ty::TraitDef>>,
    // FIXME do we need to hash this?
    expn_that_defined: Table<RDRHashAll, DefIndex, LazyValue<ExpnId>>,
    // FIXME do we need to hash this?
    default_fields: Table<RDRHashAll, DefIndex, LazyValue<DefId>>,
    // FIXME do we need to hash this?
    params_in_repr: Table<RDRHashAll, DefIndex, LazyValue<DenseBitSet<u32>>>,
    // FIXME do we need to hash this?
    repr_options: Table<RDRHashAll, DefIndex, LazyValue<ReprOptions>>,
    // `def_keys` and `def_path_hashes` represent a lazy version of a
    // `DefPathTable`. This allows us to avoid deserializing an entire
    // `DefPathTable` up front, since we may only ever use a few
    // definitions from any given crate.
    //
    // We don't need to include this in the hash, hashing `def_path_hash_map` takes care of it.
    def_keys: Table<RDRHashNone, DefIndex, LazyValue<DefKey>>,
    // FIXME do we need to hash this?
    proc_macro_quoted_spans: Table<RDRHashAll, usize, LazyValue<Span>>,
    // FIXME do we need to hash this?
    variant_data: Table<RDRHashAll, DefIndex, LazyValue<VariantData>>,
    // FIXME do we need to hash this?
    assoc_container: Table<RDRHashAll, DefIndex, LazyValue<ty::AssocContainer>>,
    // FIXME do we need to hash this?
    macro_definition: Table<RDRHashAll, DefIndex, LazyValue<ast::DelimArgs>>,
    // FIXME do we need to hash this?
    deduced_param_attrs: Table<RDRHashAll, DefIndex, LazyArray<DeducedParamAttrs>>,
    // FIXME do we need to hash this?
    collect_return_position_impl_trait_in_trait_tys: Table<RDRHashAll, DefIndex, LazyValue<DefIdMap<ty::EarlyBinder<'static, Ty<'static>>>>>,
    // FIXME do we need to hash this?
    doc_link_resolutions: Table<RDRHashAll, DefIndex, LazyValue<DocLinkResMap>>,
    // FIXME do we need to hash this?
    doc_link_traits_in_scope: Table<RDRHashAll, DefIndex, LazyArray<DefId>>,
    // FIXME do we need to hash this?
    assumed_wf_types_for_rpitit: Table<RDRHashAll, DefIndex, LazyArray<(Ty<'static>, Span)>>,
    // FIXME do we need to hash this?
    opaque_ty_origin: Table<RDRHashAll, DefIndex, LazyValue<hir::OpaqueTyOrigin<DefId>>>,
    // FIXME do we need to hash this?
    anon_const_kind: Table<RDRHashAll, DefIndex, LazyValue<ty::AnonConstKind>>,
    // FIXME do we need to hash this?
    const_of_item: Table<RDRHashAll, DefIndex, LazyValue<ty::EarlyBinder<'static, ty::Const<'static>>>>,
    // FIXME do we need to hash this?
    associated_types_for_impl_traits_in_trait_or_impl: Table<RDRHashAll, DefIndex, LazyValue<DefIdMap<Vec<DefId>>>>,
}

#[derive(TyEncodable, TyDecodable)]
struct VariantData {
    idx: VariantIdx,
    discr: ty::VariantDiscr,
    /// If this is unit or tuple-variant/struct, then this is the index of the ctor id.
    ctor: Option<(CtorKind, DefIndex)>,
    is_non_exhaustive: bool,
}

bitflags::bitflags! {
    #[derive(Default, Clone, Copy)]
    pub struct AttrFlags: u8 {
        const IS_DOC_HIDDEN = 1 << 0;
    }
}

/// A span tag byte encodes a bunch of data, so that we can cut out a few extra bytes from span
/// encodings (which are very common, for example, libcore has ~650,000 unique spans and over 1.1
/// million references to prior-written spans).
///
/// The byte format is split into several parts:
///
/// [ a a a a a c d d ]
///
/// `a` bits represent the span length. We have 5 bits, so we can store lengths up to 30 inline, with
/// an all-1s pattern representing that the length is stored separately.
///
/// `c` represents whether the span context is zero (and then it is not stored as a separate varint)
/// for direct span encodings, and whether the offset is absolute or relative otherwise (zero for
/// absolute).
///
/// d bits represent the kind of span we are storing (local, foreign, partial, indirect).
#[derive(Encodable, Decodable, Copy, Clone)]
struct SpanTag(u8);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum SpanKind {
    Local = 0b00,
    Foreign = 0b01,
    Partial = 0b10,
    // Indicates the actual span contents are elsewhere.
    // If this is the kind, then the span context bit represents whether it is a relative or
    // absolute offset.
    Indirect = 0b11,
}

impl SpanTag {
    fn new(kind: SpanKind, context: rustc_span::SyntaxContext, length: usize) -> SpanTag {
        let mut data = 0u8;
        data |= kind as u8;
        if context.is_root() {
            data |= 0b100;
        }
        let all_1s_len = (0xffu8 << 3) >> 3;
        // strictly less than - all 1s pattern is a sentinel for storage being out of band.
        if length < all_1s_len as usize {
            data |= (length as u8) << 3;
        } else {
            data |= all_1s_len << 3;
        }

        SpanTag(data)
    }

    fn indirect(relative: bool, length_bytes: u8) -> SpanTag {
        let mut tag = SpanTag(SpanKind::Indirect as u8);
        if relative {
            tag.0 |= 0b100;
        }
        assert!(length_bytes <= 8);
        tag.0 |= length_bytes << 3;
        tag
    }

    fn kind(self) -> SpanKind {
        let masked = self.0 & 0b11;
        match masked {
            0b00 => SpanKind::Local,
            0b01 => SpanKind::Foreign,
            0b10 => SpanKind::Partial,
            0b11 => SpanKind::Indirect,
            _ => unreachable!(),
        }
    }

    fn is_relative_offset(self) -> bool {
        debug_assert_eq!(self.kind(), SpanKind::Indirect);
        self.0 & 0b100 != 0
    }

    fn context(self) -> Option<rustc_span::SyntaxContext> {
        if self.0 & 0b100 != 0 { Some(rustc_span::SyntaxContext::root()) } else { None }
    }

    fn length(self) -> Option<rustc_span::BytePos> {
        let all_1s_len = (0xffu8 << 3) >> 3;
        let len = self.0 >> 3;
        if len != all_1s_len { Some(rustc_span::BytePos(u32::from(len))) } else { None }
    }
}

// Tags for encoding Symbol's
const SYMBOL_STR: u8 = 0;
const SYMBOL_OFFSET: u8 = 1;
const SYMBOL_PREDEFINED: u8 = 2;

pub fn provide(providers: &mut Providers) {
    encoder::provide(&mut providers.queries);
    decoder::provide(providers);
}
