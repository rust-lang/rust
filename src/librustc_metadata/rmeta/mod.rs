use decoder::Metadata;
use table::{Table, TableBuilder};

use rustc_ast::ast::{self, MacroDef};
use rustc_attr as attr;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::MetadataRef;
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::{DefId, DefIndex};
use rustc_hir::lang_items;
use rustc_index::vec::IndexVec;
use rustc_middle::hir::exports::Export;
use rustc_middle::middle::cstore::{DepKind, ForeignModule, LinkagePreference, NativeLib};
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportLevel};
use rustc_middle::mir;
use rustc_middle::ty::{self, ReprOptions, Ty};
use rustc_serialize::opaque::Encoder;
use rustc_session::config::SymbolManglingVersion;
use rustc_session::CrateDisambiguator;
use rustc_span::edition::Edition;
use rustc_span::symbol::Symbol;
use rustc_span::{self, Span};
use rustc_target::spec::{PanicStrategy, TargetTriple};

use std::marker::PhantomData;
use std::num::NonZeroUsize;

pub use decoder::{provide, provide_extern};
crate use decoder::{CrateMetadata, CrateNumMap, MetadataBlob};

mod decoder;
mod encoder;
mod table;

crate fn rustc_version() -> String {
    format!("rustc {}", option_env!("CFG_VERSION").unwrap_or("unknown version"))
}

/// Metadata encoding version.
/// N.B., increment this if you change the format of metadata such that
/// the rustc version can't be found to compare with `rustc_version()`.
const METADATA_VERSION: u8 = 5;

/// Metadata header which includes `METADATA_VERSION`.
///
/// This header is followed by the position of the `CrateRoot`,
/// which is encoded as a 32-bit big-endian unsigned integer,
/// and further followed by the rustc version string.
crate const METADATA_HEADER: &[u8; 8] = &[b'r', b'u', b's', b't', 0, 0, 0, METADATA_VERSION];

/// Additional metadata for a `Lazy<T>` where `T` may not be `Sized`,
/// e.g. for `Lazy<[T]>`, this is the length (count of `T` values).
trait LazyMeta {
    type Meta: Copy + 'static;

    /// Returns the minimum encoded size.
    // FIXME(eddyb) Give better estimates for certain types.
    fn min_size(meta: Self::Meta) -> usize;
}

impl<T> LazyMeta for T {
    type Meta = ();

    fn min_size(_: ()) -> usize {
        assert_ne!(std::mem::size_of::<T>(), 0);
        1
    }
}

impl<T> LazyMeta for [T] {
    type Meta = usize;

    fn min_size(len: usize) -> usize {
        len * T::min_size(())
    }
}

/// A value of type T referred to by its absolute position
/// in the metadata, and which can be decoded lazily.
///
/// Metadata is effective a tree, encoded in post-order,
/// and with the root's position written next to the header.
/// That means every single `Lazy` points to some previous
/// location in the metadata and is part of a larger node.
///
/// The first `Lazy` in a node is encoded as the backwards
/// distance from the position where the containing node
/// starts and where the `Lazy` points to, while the rest
/// use the forward distance from the previous `Lazy`.
/// Distances start at 1, as 0-byte nodes are invalid.
/// Also invalid are nodes being referred in a different
/// order than they were encoded in.
///
/// # Sequences (`Lazy<[T]>`)
///
/// Unlike `Lazy<Vec<T>>`, the length is encoded next to the
/// position, not at the position, which means that the length
/// doesn't need to be known before encoding all the elements.
///
/// If the length is 0, no position is encoded, but otherwise,
/// the encoding is that of `Lazy`, with the distinction that
/// the minimal distance the length of the sequence, i.e.
/// it's assumed there's no 0-byte element in the sequence.
#[must_use]
// FIXME(#59875) the `Meta` parameter only exists to dodge
// invariance wrt `T` (coming from the `meta: T::Meta` field).
struct Lazy<T, Meta = <T as LazyMeta>::Meta>
where
    T: ?Sized + LazyMeta<Meta = Meta>,
    Meta: 'static + Copy,
{
    position: NonZeroUsize,
    meta: Meta,
    _marker: PhantomData<T>,
}

impl<T: ?Sized + LazyMeta> Lazy<T> {
    fn from_position_and_meta(position: NonZeroUsize, meta: T::Meta) -> Lazy<T> {
        Lazy { position, meta, _marker: PhantomData }
    }
}

impl<T> Lazy<T> {
    fn from_position(position: NonZeroUsize) -> Lazy<T> {
        Lazy::from_position_and_meta(position, ())
    }
}

impl<T> Lazy<[T]> {
    fn empty() -> Lazy<[T]> {
        Lazy::from_position_and_meta(NonZeroUsize::new(1).unwrap(), 0)
    }
}

impl<T: ?Sized + LazyMeta> Copy for Lazy<T> {}
impl<T: ?Sized + LazyMeta> Clone for Lazy<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized + LazyMeta> rustc_serialize::UseSpecializedEncodable for Lazy<T> {}
impl<T: ?Sized + LazyMeta> rustc_serialize::UseSpecializedDecodable for Lazy<T> {}

/// Encoding / decoding state for `Lazy`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum LazyState {
    /// Outside of a metadata node.
    NoNode,

    /// Inside a metadata node, and before any `Lazy`.
    /// The position is that of the node itself.
    NodeStart(NonZeroUsize),

    /// Inside a metadata node, with a previous `Lazy`.
    /// The position is a conservative estimate of where that
    /// previous `Lazy` would end (see their comments).
    Previous(NonZeroUsize),
}

// FIXME(#59875) `Lazy!(T)` replaces `Lazy<T>`, passing the `Meta` parameter
// manually, instead of relying on the default, to get the correct variance.
// Only needed when `T` itself contains a parameter (e.g. `'tcx`).
macro_rules! Lazy {
    (Table<$I:ty, $T:ty>) => {Lazy<Table<$I, $T>, usize>};
    ([$T:ty]) => {Lazy<[$T], usize>};
    ($T:ty) => {Lazy<$T, ()>};
}

#[derive(RustcEncodable, RustcDecodable)]
crate struct CrateRoot<'tcx> {
    name: Symbol,
    triple: TargetTriple,
    extra_filename: String,
    hash: Svh,
    disambiguator: CrateDisambiguator,
    panic_strategy: PanicStrategy,
    edition: Edition,
    has_global_allocator: bool,
    has_panic_handler: bool,
    has_default_lib_allocator: bool,
    plugin_registrar_fn: Option<DefIndex>,
    proc_macro_decls_static: Option<DefIndex>,
    proc_macro_stability: Option<attr::Stability>,

    crate_deps: Lazy<[CrateDep]>,
    dylib_dependency_formats: Lazy<[Option<LinkagePreference>]>,
    lib_features: Lazy<[(Symbol, Option<Symbol>)]>,
    lang_items: Lazy<[(DefIndex, usize)]>,
    lang_items_missing: Lazy<[lang_items::LangItem]>,
    diagnostic_items: Lazy<[(Symbol, DefIndex)]>,
    native_libraries: Lazy<[NativeLib]>,
    foreign_modules: Lazy<[ForeignModule]>,
    source_map: Lazy<[rustc_span::SourceFile]>,
    def_path_table: Lazy<rustc_hir::definitions::DefPathTable>,
    impls: Lazy<[TraitImpls]>,
    interpret_alloc_index: Lazy<[u32]>,

    tables: LazyTables<'tcx>,

    /// The DefIndex's of any proc macros declared by this crate.
    proc_macro_data: Option<Lazy<[DefIndex]>>,

    exported_symbols: Lazy!([(ExportedSymbol<'tcx>, SymbolExportLevel)]),

    compiler_builtins: bool,
    needs_allocator: bool,
    needs_panic_runtime: bool,
    no_builtins: bool,
    panic_runtime: bool,
    profiler_runtime: bool,
    symbol_mangling_version: SymbolManglingVersion,
}

#[derive(RustcEncodable, RustcDecodable)]
crate struct CrateDep {
    pub name: Symbol,
    pub hash: Svh,
    pub host_hash: Option<Svh>,
    pub kind: DepKind,
    pub extra_filename: String,
}

#[derive(RustcEncodable, RustcDecodable)]
crate struct TraitImpls {
    trait_id: (u32, DefIndex),
    impls: Lazy<[DefIndex]>,
}

/// Define `LazyTables` and `TableBuilders` at the same time.
macro_rules! define_tables {
    ($($name:ident: Table<DefIndex, $T:ty>),+ $(,)?) => {
        #[derive(RustcEncodable, RustcDecodable)]
        crate struct LazyTables<'tcx> {
            $($name: Lazy!(Table<DefIndex, $T>)),+
        }

        #[derive(Default)]
        struct TableBuilders<'tcx> {
            $($name: TableBuilder<DefIndex, $T>),+
        }

        impl TableBuilders<'tcx> {
            fn encode(&self, buf: &mut Encoder) -> LazyTables<'tcx> {
                LazyTables {
                    $($name: self.$name.encode(buf)),+
                }
            }
        }
    }
}

define_tables! {
    kind: Table<DefIndex, Lazy<EntryKind>>,
    visibility: Table<DefIndex, Lazy<ty::Visibility>>,
    span: Table<DefIndex, Lazy<Span>>,
    ident_span: Table<DefIndex, Lazy<Span>>,
    attributes: Table<DefIndex, Lazy<[ast::Attribute]>>,
    children: Table<DefIndex, Lazy<[DefIndex]>>,
    stability: Table<DefIndex, Lazy<attr::Stability>>,
    const_stability: Table<DefIndex, Lazy<attr::ConstStability>>,
    deprecation: Table<DefIndex, Lazy<attr::Deprecation>>,
    ty: Table<DefIndex, Lazy!(Ty<'tcx>)>,
    fn_sig: Table<DefIndex, Lazy!(ty::PolyFnSig<'tcx>)>,
    impl_trait_ref: Table<DefIndex, Lazy!(ty::TraitRef<'tcx>)>,
    inherent_impls: Table<DefIndex, Lazy<[DefIndex]>>,
    variances: Table<DefIndex, Lazy<[ty::Variance]>>,
    generics: Table<DefIndex, Lazy<ty::Generics>>,
    explicit_predicates: Table<DefIndex, Lazy!(ty::GenericPredicates<'tcx>)>,
    // FIXME(eddyb) this would ideally be `Lazy<[...]>` but `ty::Predicate`
    // doesn't handle shorthands in its own (de)serialization impls,
    // as it's an `enum` for which we want to derive (de)serialization,
    // so the `ty::codec` APIs handle the whole `&'tcx [...]` at once.
    // Also, as an optimization, a missing entry indicates an empty `&[]`.
    inferred_outlives: Table<DefIndex, Lazy!(&'tcx [(ty::Predicate<'tcx>, Span)])>,
    super_predicates: Table<DefIndex, Lazy!(ty::GenericPredicates<'tcx>)>,
    mir: Table<DefIndex, Lazy!(mir::Body<'tcx>)>,
    promoted_mir: Table<DefIndex, Lazy!(IndexVec<mir::Promoted, mir::Body<'tcx>>)>,
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
enum EntryKind {
    Const(mir::ConstQualifs, Lazy<RenderedConst>),
    ImmStatic,
    MutStatic,
    ForeignImmStatic,
    ForeignMutStatic,
    ForeignMod,
    ForeignType,
    GlobalAsm,
    Type,
    TypeParam,
    ConstParam,
    OpaqueTy,
    Enum(ReprOptions),
    Field,
    Variant(Lazy<VariantData>),
    Struct(Lazy<VariantData>, ReprOptions),
    Union(Lazy<VariantData>, ReprOptions),
    Fn(Lazy<FnData>),
    ForeignFn(Lazy<FnData>),
    Mod(Lazy<ModData>),
    MacroDef(Lazy<MacroDef>),
    Closure,
    Generator(hir::GeneratorKind),
    Trait(Lazy<TraitData>),
    Impl(Lazy<ImplData>),
    AssocFn(Lazy<AssocFnData>),
    AssocType(AssocContainer),
    AssocOpaqueTy(AssocContainer),
    AssocConst(AssocContainer, mir::ConstQualifs, Lazy<RenderedConst>),
    TraitAlias,
}

/// Contains a constant which has been rendered to a String.
/// Used by rustdoc.
#[derive(RustcEncodable, RustcDecodable)]
struct RenderedConst(String);

#[derive(RustcEncodable, RustcDecodable)]
struct ModData {
    reexports: Lazy<[Export<hir::HirId>]>,
}

#[derive(RustcEncodable, RustcDecodable)]
struct FnData {
    asyncness: hir::IsAsync,
    constness: hir::Constness,
    param_names: Lazy<[Symbol]>,
}

#[derive(RustcEncodable, RustcDecodable)]
struct VariantData {
    ctor_kind: CtorKind,
    discr: ty::VariantDiscr,
    /// If this is unit or tuple-variant/struct, then this is the index of the ctor id.
    ctor: Option<DefIndex>,
}

#[derive(RustcEncodable, RustcDecodable)]
struct TraitData {
    unsafety: hir::Unsafety,
    paren_sugar: bool,
    has_auto_impl: bool,
    is_marker: bool,
    specialization_kind: ty::trait_def::TraitSpecializationKind,
}

#[derive(RustcEncodable, RustcDecodable)]
struct ImplData {
    polarity: ty::ImplPolarity,
    defaultness: hir::Defaultness,
    parent_impl: Option<DefId>,

    /// This is `Some` only for impls of `CoerceUnsized`.
    // FIXME(eddyb) perhaps compute this on the fly if cheap enough?
    coerce_unsized_info: Option<ty::adjustment::CoerceUnsizedInfo>,
}

/// Describes whether the container of an associated item
/// is a trait or an impl and whether, in a trait, it has
/// a default, or an in impl, whether it's marked "default".
#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
enum AssocContainer {
    TraitRequired,
    TraitWithDefault,
    ImplDefault,
    ImplFinal,
}

impl AssocContainer {
    fn with_def_id(&self, def_id: DefId) -> ty::AssocItemContainer {
        match *self {
            AssocContainer::TraitRequired | AssocContainer::TraitWithDefault => {
                ty::TraitContainer(def_id)
            }

            AssocContainer::ImplDefault | AssocContainer::ImplFinal => ty::ImplContainer(def_id),
        }
    }

    fn defaultness(&self) -> hir::Defaultness {
        match *self {
            AssocContainer::TraitRequired => hir::Defaultness::Default { has_value: false },

            AssocContainer::TraitWithDefault | AssocContainer::ImplDefault => {
                hir::Defaultness::Default { has_value: true }
            }

            AssocContainer::ImplFinal => hir::Defaultness::Final,
        }
    }
}

#[derive(RustcEncodable, RustcDecodable)]
struct AssocFnData {
    fn_data: FnData,
    container: AssocContainer,
    has_self: bool,
}

#[derive(RustcEncodable, RustcDecodable)]
struct GeneratorData<'tcx> {
    layout: mir::GeneratorLayout<'tcx>,
}

// Tags used for encoding Spans:
const TAG_VALID_SPAN_LOCAL: u8 = 0;
const TAG_VALID_SPAN_FOREIGN: u8 = 1;
const TAG_INVALID_SPAN: u8 = 2;
