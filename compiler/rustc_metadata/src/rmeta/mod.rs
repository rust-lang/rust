use crate::creader::CrateMetadataRef;
use decoder::Metadata;
use def_path_hash_map::DefPathHashMapRef;
use table::TableBuilder;

use rustc_ast as ast;
use rustc_attr as attr;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::MetadataRef;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, DefPathHash, StableCrateId};
use rustc_hir::definitions::DefKey;
use rustc_hir::lang_items;
use rustc_index::{bit_set::FiniteBitSet, vec::IndexVec};
use rustc_middle::metadata::ModChild;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo};
use rustc_middle::mir;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, ReprOptions, Ty};
use rustc_middle::ty::{GeneratorDiagnosticData, ParameterizedOverTcx, TyCtxt};
use rustc_serialize::opaque::FileEncoder;
use rustc_session::config::SymbolManglingVersion;
use rustc_session::cstore::{CrateDepKind, ForeignModule, LinkagePreference, NativeLib};
use rustc_span::edition::Edition;
use rustc_span::hygiene::{ExpnIndex, MacroKind};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{self, ExpnData, ExpnHash, ExpnId, Span};
use rustc_target::spec::{PanicStrategy, TargetTriple};

use std::marker::PhantomData;
use std::num::NonZeroUsize;

pub use decoder::provide_extern;
use decoder::DecodeContext;
pub(crate) use decoder::{CrateMetadata, CrateNumMap, MetadataBlob};
use encoder::EncodeContext;
pub use encoder::{encode_metadata, EncodedMetadata};
use rustc_span::hygiene::SyntaxContextData;

mod decoder;
mod def_path_hash_map;
mod encoder;
mod table;

pub(crate) fn rustc_version() -> String {
    format!("rustc {}", option_env!("CFG_VERSION").unwrap_or("unknown version"))
}

/// Metadata encoding version.
/// N.B., increment this if you change the format of metadata such that
/// the rustc version can't be found to compare with `rustc_version()`.
const METADATA_VERSION: u8 = 6;

/// Metadata header which includes `METADATA_VERSION`.
///
/// This header is followed by the position of the `CrateRoot`,
/// which is encoded as a 32-bit big-endian unsigned integer,
/// and further followed by the rustc version string.
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
    position: NonZeroUsize,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for LazyValue<T> {
    type Value<'tcx> = LazyValue<T::Value<'tcx>>;
}

impl<T> LazyValue<T> {
    fn from_position(position: NonZeroUsize) -> LazyValue<T> {
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
    position: NonZeroUsize,
    num_elems: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ParameterizedOverTcx> ParameterizedOverTcx for LazyArray<T> {
    type Value<'tcx> = LazyArray<T::Value<'tcx>>;
}

impl<T> LazyArray<T> {
    fn from_position_and_num_elems(position: NonZeroUsize, num_elems: usize) -> LazyArray<T> {
        LazyArray { position, num_elems, _marker: PhantomData }
    }

    fn empty() -> LazyArray<T> {
        LazyArray::from_position_and_num_elems(NonZeroUsize::new(1).unwrap(), 0)
    }
}

/// A list of lazily-decoded values, with the added capability of random access.
///
/// Random-access table (i.e. offering constant-time `get`/`set`), similar to
/// `LazyArray<T>`, but without requiring encoding or decoding all the values
/// eagerly and in-order.
struct LazyTable<I, T> {
    position: NonZeroUsize,
    encoded_size: usize,
    _marker: PhantomData<fn(I) -> T>,
}

impl<I: 'static, T: ParameterizedOverTcx> ParameterizedOverTcx for LazyTable<I, T> {
    type Value<'tcx> = LazyTable<I, T::Value<'tcx>>;
}

impl<I, T> LazyTable<I, T> {
    fn from_position_and_encoded_size(
        position: NonZeroUsize,
        encoded_size: usize,
    ) -> LazyTable<I, T> {
        LazyTable { position, encoded_size, _marker: PhantomData }
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
    NodeStart(NonZeroUsize),

    /// Inside a metadata node, with a previous `Lazy`s.
    /// The position is where that previous `Lazy` would start.
    Previous(NonZeroUsize),
}

type SyntaxContextTable = LazyTable<u32, LazyValue<SyntaxContextData>>;
type ExpnDataTable = LazyTable<ExpnIndex, LazyValue<ExpnData>>;
type ExpnHashTable = LazyTable<ExpnIndex, LazyValue<ExpnHash>>;

#[derive(MetadataEncodable, MetadataDecodable)]
pub(crate) struct ProcMacroData {
    proc_macro_decls_static: DefIndex,
    stability: Option<attr::Stability>,
    macros: LazyArray<DefIndex>,
}

/// Serialized metadata for a crate.
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
#[derive(MetadataEncodable, MetadataDecodable)]
pub(crate) struct CrateRoot {
    name: Symbol,
    triple: TargetTriple,
    extra_filename: String,
    hash: Svh,
    stable_crate_id: StableCrateId,
    required_panic_strategy: Option<PanicStrategy>,
    panic_in_drop_strategy: PanicStrategy,
    edition: Edition,
    has_global_allocator: bool,
    has_panic_handler: bool,
    has_default_lib_allocator: bool,

    crate_deps: LazyArray<CrateDep>,
    dylib_dependency_formats: LazyArray<Option<LinkagePreference>>,
    lib_features: LazyArray<(Symbol, Option<Symbol>)>,
    stability_implications: LazyArray<(Symbol, Symbol)>,
    lang_items: LazyArray<(DefIndex, usize)>,
    lang_items_missing: LazyArray<lang_items::LangItem>,
    diagnostic_items: LazyArray<(Symbol, DefIndex)>,
    native_libraries: LazyArray<NativeLib>,
    foreign_modules: LazyArray<ForeignModule>,
    traits: LazyArray<DefIndex>,
    impls: LazyArray<TraitImpls>,
    incoherent_impls: LazyArray<IncoherentImpls>,
    interpret_alloc_index: LazyArray<u32>,
    proc_macro_data: Option<ProcMacroData>,

    tables: LazyTables,
    debugger_visualizers: LazyArray<rustc_span::DebuggerVisualizerFile>,

    exported_symbols: LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>,

    syntax_contexts: SyntaxContextTable,
    expn_data: ExpnDataTable,
    expn_hashes: ExpnHashTable,

    def_path_hash_map: LazyValue<DefPathHashMapRef<'static>>,

    source_map: LazyTable<u32, LazyValue<rustc_span::SourceFile>>,

    compiler_builtins: bool,
    needs_allocator: bool,
    needs_panic_runtime: bool,
    no_builtins: bool,
    panic_runtime: bool,
    profiler_runtime: bool,
    symbol_mangling_version: SymbolManglingVersion,
}

/// On-disk representation of `DefId`.
/// This creates a type-safe way to enforce that we remap the CrateNum between the on-disk
/// representation and the compilation session.
#[derive(Copy, Clone)]
pub(crate) struct RawDefId {
    krate: u32,
    index: u32,
}

impl Into<RawDefId> for DefId {
    fn into(self) -> RawDefId {
        RawDefId { krate: self.krate.as_u32(), index: self.index.as_u32() }
    }
}

impl RawDefId {
    /// This exists so that `provide_one!` is happy
    fn decode(self, meta: (CrateMetadataRef<'_>, TyCtxt<'_>)) -> DefId {
        self.decode_from_cdata(meta.0)
    }

    fn decode_from_cdata(self, cdata: CrateMetadataRef<'_>) -> DefId {
        let krate = CrateNum::from_u32(self.krate);
        let krate = cdata.map_encoded_cnum_to_current(krate);
        DefId { krate, index: DefIndex::from_u32(self.index) }
    }
}

#[derive(Encodable, Decodable)]
pub(crate) struct CrateDep {
    pub name: Symbol,
    pub hash: Svh,
    pub host_hash: Option<Svh>,
    pub kind: CrateDepKind,
    pub extra_filename: String,
}

#[derive(MetadataEncodable, MetadataDecodable)]
pub(crate) struct TraitImpls {
    trait_id: (u32, DefIndex),
    impls: LazyArray<(DefIndex, Option<SimplifiedType>)>,
}

#[derive(MetadataEncodable, MetadataDecodable)]
pub(crate) struct IncoherentImpls {
    self_ty: SimplifiedType,
    impls: LazyArray<DefIndex>,
}

/// Define `LazyTables` and `TableBuilders` at the same time.
macro_rules! define_tables {
    ($($name:ident: Table<$IDX:ty, $T:ty>),+ $(,)?) => {
        #[derive(MetadataEncodable, MetadataDecodable)]
        pub(crate) struct LazyTables {
            $($name: LazyTable<$IDX, $T>),+
        }

        #[derive(Default)]
        struct TableBuilders {
            $($name: TableBuilder<$IDX, $T>),+
        }

        impl TableBuilders {
            fn encode(&self, buf: &mut FileEncoder) -> LazyTables {
                LazyTables {
                    $($name: self.$name.encode(buf)),+
                }
            }
        }
    }
}

define_tables! {
    kind: Table<DefIndex, LazyValue<EntryKind>>,
    attributes: Table<DefIndex, LazyArray<ast::Attribute>>,
    children: Table<DefIndex, LazyArray<DefIndex>>,

    opt_def_kind: Table<DefIndex, DefKind>,
    visibility: Table<DefIndex, LazyValue<ty::Visibility>>,
    def_span: Table<DefIndex, LazyValue<Span>>,
    def_ident_span: Table<DefIndex, LazyValue<Span>>,
    lookup_stability: Table<DefIndex, LazyValue<attr::Stability>>,
    lookup_const_stability: Table<DefIndex, LazyValue<attr::ConstStability>>,
    lookup_default_body_stability: Table<DefIndex, LazyValue<attr::DefaultBodyStability>>,
    lookup_deprecation_entry: Table<DefIndex, LazyValue<attr::Deprecation>>,
    // As an optimization, a missing entry indicates an empty `&[]`.
    explicit_item_bounds: Table<DefIndex, LazyArray<(ty::Predicate<'static>, Span)>>,
    explicit_predicates_of: Table<DefIndex, LazyValue<ty::GenericPredicates<'static>>>,
    generics_of: Table<DefIndex, LazyValue<ty::Generics>>,
    // As an optimization, a missing entry indicates an empty `&[]`.
    inferred_outlives_of: Table<DefIndex, LazyArray<(ty::Predicate<'static>, Span)>>,
    super_predicates_of: Table<DefIndex, LazyValue<ty::GenericPredicates<'static>>>,
    type_of: Table<DefIndex, LazyValue<Ty<'static>>>,
    variances_of: Table<DefIndex, LazyArray<ty::Variance>>,
    fn_sig: Table<DefIndex, LazyValue<ty::PolyFnSig<'static>>>,
    codegen_fn_attrs: Table<DefIndex, LazyValue<CodegenFnAttrs>>,
    impl_trait_ref: Table<DefIndex, LazyValue<ty::TraitRef<'static>>>,
    const_param_default: Table<DefIndex, LazyValue<rustc_middle::ty::Const<'static>>>,
    optimized_mir: Table<DefIndex, LazyValue<mir::Body<'static>>>,
    mir_for_ctfe: Table<DefIndex, LazyValue<mir::Body<'static>>>,
    promoted_mir: Table<DefIndex, LazyValue<IndexVec<mir::Promoted, mir::Body<'static>>>>,
    // FIXME(compiler-errors): Why isn't this a LazyArray?
    thir_abstract_const: Table<DefIndex, LazyValue<&'static [ty::abstract_const::Node<'static>]>>,
    impl_parent: Table<DefIndex, RawDefId>,
    impl_polarity: Table<DefIndex, ty::ImplPolarity>,
    constness: Table<DefIndex, hir::Constness>,
    is_intrinsic: Table<DefIndex, ()>,
    impl_defaultness: Table<DefIndex, hir::Defaultness>,
    // FIXME(eddyb) perhaps compute this on the fly if cheap enough?
    coerce_unsized_info: Table<DefIndex, LazyValue<ty::adjustment::CoerceUnsizedInfo>>,
    mir_const_qualif: Table<DefIndex, LazyValue<mir::ConstQualifs>>,
    rendered_const: Table<DefIndex, LazyValue<String>>,
    asyncness: Table<DefIndex, hir::IsAsync>,
    fn_arg_names: Table<DefIndex, LazyArray<Ident>>,
    generator_kind: Table<DefIndex, LazyValue<hir::GeneratorKind>>,
    trait_def: Table<DefIndex, LazyValue<ty::TraitDef>>,

    trait_item_def_id: Table<DefIndex, RawDefId>,
    inherent_impls: Table<DefIndex, LazyArray<DefIndex>>,
    expn_that_defined: Table<DefIndex, LazyValue<ExpnId>>,
    unused_generic_params: Table<DefIndex, LazyValue<FiniteBitSet<u32>>>,
    repr_options: Table<DefIndex, LazyValue<ReprOptions>>,
    // `def_keys` and `def_path_hashes` represent a lazy version of a
    // `DefPathTable`. This allows us to avoid deserializing an entire
    // `DefPathTable` up front, since we may only ever use a few
    // definitions from any given crate.
    def_keys: Table<DefIndex, LazyValue<DefKey>>,
    def_path_hashes: Table<DefIndex, DefPathHash>,
    proc_macro_quoted_spans: Table<usize, LazyValue<Span>>,
    generator_diagnostic_data: Table<DefIndex, LazyValue<GeneratorDiagnosticData<'static>>>,
    may_have_doc_links: Table<DefIndex, ()>,
}

#[derive(Copy, Clone, MetadataEncodable, MetadataDecodable)]
enum EntryKind {
    AnonConst,
    Const,
    Static,
    ForeignStatic,
    ForeignMod,
    ForeignType,
    GlobalAsm,
    Type,
    TypeParam,
    ConstParam,
    OpaqueTy,
    Enum,
    Field,
    Variant(LazyValue<VariantData>),
    Struct(LazyValue<VariantData>),
    Union(LazyValue<VariantData>),
    Fn,
    ForeignFn,
    Mod(LazyArray<ModChild>),
    MacroDef(LazyValue<ast::MacArgs>, /*macro_rules*/ bool),
    ProcMacro(MacroKind),
    Closure,
    Generator,
    Trait,
    Impl,
    AssocFn { container: ty::AssocItemContainer, has_self: bool },
    AssocType(ty::AssocItemContainer),
    AssocConst(ty::AssocItemContainer),
    TraitAlias,
}

#[derive(TyEncodable, TyDecodable)]
struct VariantData {
    ctor_kind: CtorKind,
    discr: ty::VariantDiscr,
    /// If this is unit or tuple-variant/struct, then this is the index of the ctor id.
    ctor: Option<DefIndex>,
    is_non_exhaustive: bool,
}

#[derive(TyEncodable, TyDecodable)]
struct GeneratorData<'tcx> {
    layout: mir::GeneratorLayout<'tcx>,
}

// Tags used for encoding Spans:
const TAG_VALID_SPAN_LOCAL: u8 = 0;
const TAG_VALID_SPAN_FOREIGN: u8 = 1;
const TAG_PARTIAL_SPAN: u8 = 2;

// Tags for encoding Symbol's
const SYMBOL_STR: u8 = 0;
const SYMBOL_OFFSET: u8 = 1;
const SYMBOL_PREINTERNED: u8 = 2;

pub fn provide(providers: &mut Providers) {
    encoder::provide(providers);
    decoder::provide(providers);
}

trivially_parameterized_over_tcx! {
    VariantData,
    EntryKind,
    RawDefId,
    TraitImpls,
    IncoherentImpls,
    CrateRoot,
    CrateDep,
}
