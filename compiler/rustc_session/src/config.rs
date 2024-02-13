//! Contains infrastructure for configuring the compiler, including parsing
//! command-line options.

pub use crate::options::*;

use crate::errors::FileWriteFail;
use crate::search_paths::SearchPath;
use crate::utils::{CanonicalizedPath, NativeLib, NativeLibKind};
use crate::{lint, HashStableContext};
use crate::{EarlyDiagCtxt, Session};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::stable_hasher::{StableOrd, ToStableHashKey};
use rustc_errors::emitter::HumanReadableErrorType;
use rustc_errors::{ColorConfig, DiagCtxtFlags, DiagnosticArgValue, IntoDiagnosticArg};
use rustc_feature::UnstableFeatures;
use rustc_span::edition::{Edition, DEFAULT_EDITION, EDITION_NAME_LIST, LATEST_STABLE_EDITION};
use rustc_span::source_map::FilePathMapping;
use rustc_span::symbol::{sym, Symbol};
use rustc_span::{FileName, FileNameDisplayPreference, RealFileName, SourceFileHashAlgorithm};
use rustc_target::abi::Align;
use rustc_target::spec::LinkSelfContainedComponents;
use rustc_target::spec::{PanicStrategy, RelocModel, SanitizerSet, SplitDebuginfo};
use rustc_target::spec::{Target, TargetTriple, TargetWarnings, TARGETS};
use std::collections::btree_map::{
    Iter as BTreeMapIter, Keys as BTreeMapKeysIter, Values as BTreeMapValuesIter,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fmt;
use std::fs;
use std::hash::Hash;
use std::iter;
use std::path::{Path, PathBuf};
use std::str::{self, FromStr};
use std::sync::LazyLock;

pub mod sigpipe;

/// The different settings that the `-C strip` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum Strip {
    /// Do not strip at all.
    None,

    /// Strip debuginfo.
    Debuginfo,

    /// Strip all symbols.
    Symbols,
}

/// The different settings that the `-C control-flow-guard` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum CFGuard {
    /// Do not emit Control Flow Guard metadata or checks.
    Disabled,

    /// Emit Control Flow Guard metadata but no checks.
    NoChecks,

    /// Emit Control Flow Guard metadata and checks.
    Checks,
}

/// The different settings that the `-Z cf-protection` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum CFProtection {
    /// Do not enable control-flow protection
    None,

    /// Emit control-flow protection for branches (enables indirect branch tracking).
    Branch,

    /// Emit control-flow protection for returns.
    Return,

    /// Emit control-flow protection for both branches and returns.
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, HashStable_Generic)]
pub enum OptLevel {
    No,         // -O0
    Less,       // -O1
    Default,    // -O2
    Aggressive, // -O3
    Size,       // -Os
    SizeMin,    // -Oz
}

/// This is what the `LtoCli` values get mapped to after resolving defaults and
/// and taking other command line options into account.
///
/// Note that linker plugin-based LTO is a different mechanism entirely.
#[derive(Clone, PartialEq)]
pub enum Lto {
    /// Don't do any LTO whatsoever.
    No,

    /// Do a full-crate-graph (inter-crate) LTO with ThinLTO.
    Thin,

    /// Do a local ThinLTO (intra-crate, over the CodeGen Units of the local crate only). This is
    /// only relevant if multiple CGUs are used.
    ThinLocal,

    /// Do a full-crate-graph (inter-crate) LTO with "fat" LTO.
    Fat,
}

/// The different settings that the `-C lto` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum LtoCli {
    /// `-C lto=no`
    No,
    /// `-C lto=yes`
    Yes,
    /// `-C lto`
    NoParam,
    /// `-C lto=thin`
    Thin,
    /// `-C lto=fat`
    Fat,
    /// No `-C lto` flag passed
    Unspecified,
}

/// The different settings that the `-C instrument-coverage` flag can have.
///
/// Coverage instrumentation now supports combining `-C instrument-coverage`
/// with compiler and linker optimization (enabled with `-O` or `-C opt-level=1`
/// and higher). Nevertheless, there are many variables, depending on options
/// selected, code structure, and enabled attributes. If errors are encountered,
/// either while compiling or when generating `llvm-cov show` reports, consider
/// lowering the optimization level, including or excluding `-C link-dead-code`,
/// or using `-Zunstable-options -C instrument-coverage=except-unused-functions`
/// or `-Zunstable-options -C instrument-coverage=except-unused-generics`.
///
/// Note that `ExceptUnusedFunctions` means: When `mapgen.rs` generates the
/// coverage map, it will not attempt to generate synthetic functions for unused
/// (and not code-generated) functions (whether they are generic or not). As a
/// result, non-codegenned functions will not be included in the coverage map,
/// and will not appear, as covered or uncovered, in coverage reports.
///
/// `ExceptUnusedGenerics` will add synthetic functions to the coverage map,
/// unless the function has type parameters.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum InstrumentCoverage {
    /// Default `-C instrument-coverage` or `-C instrument-coverage=statement`
    All,
    /// Additionally, instrument branches and output branch coverage.
    /// `-Zunstable-options -C instrument-coverage=branch`
    Branch,
    /// `-Zunstable-options -C instrument-coverage=except-unused-generics`
    ExceptUnusedGenerics,
    /// `-Zunstable-options -C instrument-coverage=except-unused-functions`
    ExceptUnusedFunctions,
    /// `-C instrument-coverage=off` (or `no`, etc.)
    Off,
}

/// Settings for `-Z instrument-xray` flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct InstrumentXRay {
    /// `-Z instrument-xray=always`, force instrumentation
    pub always: bool,
    /// `-Z instrument-xray=never`, disable instrumentation
    pub never: bool,
    /// `-Z instrument-xray=ignore-loops`, ignore presence of loops,
    /// instrument functions based only on instruction count
    pub ignore_loops: bool,
    /// `-Z instrument-xray=instruction-threshold=N`, explicitly set instruction threshold
    /// for instrumentation, or `None` to use compiler's default
    pub instruction_threshold: Option<usize>,
    /// `-Z instrument-xray=skip-entry`, do not instrument function entry
    pub skip_entry: bool,
    /// `-Z instrument-xray=skip-exit`, do not instrument function exit
    pub skip_exit: bool,
}

#[derive(Clone, PartialEq, Hash, Debug)]
pub enum LinkerPluginLto {
    LinkerPlugin(PathBuf),
    LinkerPluginAuto,
    Disabled,
}

impl LinkerPluginLto {
    pub fn enabled(&self) -> bool {
        match *self {
            LinkerPluginLto::LinkerPlugin(_) | LinkerPluginLto::LinkerPluginAuto => true,
            LinkerPluginLto::Disabled => false,
        }
    }
}

/// The different values `-C link-self-contained` can take: a list of individually enabled or
/// disabled components used during linking, coming from the rustc distribution, instead of being
/// found somewhere on the host system.
///
/// They can be set in bulk via `-C link-self-contained=yes|y|on` or `-C
/// link-self-contained=no|n|off`, and those boolean values are the historical defaults.
///
/// But each component is fine-grained, and can be unstably targeted, to use:
/// - some CRT objects
/// - the libc static library
/// - libgcc/libunwind libraries
/// - a linker we distribute
/// - some sanitizer runtime libraries
/// - all other MinGW libraries and Windows import libs
///
#[derive(Default, Clone, PartialEq, Debug)]
pub struct LinkSelfContained {
    /// Whether the user explicitly set `-C link-self-contained` on or off, the historical values.
    /// Used for compatibility with the existing opt-in and target inference.
    pub explicitly_set: Option<bool>,

    /// The components that are enabled on the CLI, using the `+component` syntax or one of the
    /// `true` shorcuts.
    enabled_components: LinkSelfContainedComponents,

    /// The components that are disabled on the CLI, using the `-component` syntax or one of the
    /// `false` shortcuts.
    disabled_components: LinkSelfContainedComponents,
}

impl LinkSelfContained {
    /// Incorporates an enabled or disabled component as specified on the CLI, if possible.
    /// For example: `+linker`, and `-crto`.
    pub(crate) fn handle_cli_component(&mut self, component: &str) -> Option<()> {
        // Note that for example `-Cself-contained=y -Cself-contained=-linker` is not an explicit
        // set of all values like `y` or `n` used to be. Therefore, if this flag had previously been
        // set in bulk with its historical values, then manually setting a component clears that
        // `explicitly_set` state.
        if let Some(component_to_enable) = component.strip_prefix('+') {
            self.explicitly_set = None;
            self.enabled_components
                .insert(LinkSelfContainedComponents::from_str(component_to_enable)?);
            Some(())
        } else if let Some(component_to_disable) = component.strip_prefix('-') {
            self.explicitly_set = None;
            self.disabled_components
                .insert(LinkSelfContainedComponents::from_str(component_to_disable)?);
            Some(())
        } else {
            None
        }
    }

    /// Turns all components on or off and records that this was done explicitly for compatibility
    /// purposes.
    pub(crate) fn set_all_explicitly(&mut self, enabled: bool) {
        self.explicitly_set = Some(enabled);

        if enabled {
            self.enabled_components = LinkSelfContainedComponents::all();
            self.disabled_components = LinkSelfContainedComponents::empty();
        } else {
            self.enabled_components = LinkSelfContainedComponents::empty();
            self.disabled_components = LinkSelfContainedComponents::all();
        }
    }

    /// Helper creating a fully enabled `LinkSelfContained` instance. Used in tests.
    pub fn on() -> Self {
        let mut on = LinkSelfContained::default();
        on.set_all_explicitly(true);
        on
    }

    /// To help checking CLI usage while some of the values are unstable: returns whether one of the
    /// components was set individually. This would also require the `-Zunstable-options` flag, to
    /// be allowed.
    fn are_unstable_variants_set(&self) -> bool {
        let any_component_set =
            !self.enabled_components.is_empty() || !self.disabled_components.is_empty();
        self.explicitly_set.is_none() && any_component_set
    }

    /// Returns whether the self-contained linker component was enabled on the CLI, using the
    /// `-C link-self-contained=+linker` syntax, or one of the `true` shorcuts.
    pub fn is_linker_enabled(&self) -> bool {
        self.enabled_components.contains(LinkSelfContainedComponents::LINKER)
    }

    /// Returns whether the self-contained linker component was disabled on the CLI, using the
    /// `-C link-self-contained=-linker` syntax, or one of the `false` shorcuts.
    pub fn is_linker_disabled(&self) -> bool {
        self.disabled_components.contains(LinkSelfContainedComponents::LINKER)
    }

    /// Returns CLI inconsistencies to emit errors: individual components were both enabled and
    /// disabled.
    fn check_consistency(&self) -> Option<LinkSelfContainedComponents> {
        if self.explicitly_set.is_some() {
            None
        } else {
            let common = self.enabled_components.intersection(self.disabled_components);
            if common.is_empty() { None } else { Some(common) }
        }
    }
}

/// Used with `-Z assert-incr-state`.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum IncrementalStateAssertion {
    /// Found and loaded an existing session directory.
    ///
    /// Note that this says nothing about whether any particular query
    /// will be found to be red or green.
    Loaded,
    /// Did not load an existing session directory.
    NotLoaded,
}

/// The different settings that can be enabled via the `-Z location-detail` flag.
#[derive(Copy, Clone, PartialEq, Hash, Debug)]
pub struct LocationDetail {
    pub file: bool,
    pub line: bool,
    pub column: bool,
}

impl LocationDetail {
    pub fn all() -> Self {
        Self { file: true, line: true, column: true }
    }
}

#[derive(Clone, PartialEq, Hash, Debug)]
pub enum SwitchWithOptPath {
    Enabled(Option<PathBuf>),
    Disabled,
}

impl SwitchWithOptPath {
    pub fn enabled(&self) -> bool {
        match *self {
            SwitchWithOptPath::Enabled(_) => true,
            SwitchWithOptPath::Disabled => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable_Generic)]
#[derive(Encodable, Decodable)]
pub enum SymbolManglingVersion {
    Legacy,
    V0,
    Hashed,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum DebugInfo {
    None,
    LineDirectivesOnly,
    LineTablesOnly,
    Limited,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum DebugInfoCompression {
    None,
    Zlib,
    Zstd,
}

impl ToString for DebugInfoCompression {
    fn to_string(&self) -> String {
        match self {
            DebugInfoCompression::None => "none",
            DebugInfoCompression::Zlib => "zlib",
            DebugInfoCompression::Zstd => "zstd",
        }
        .to_owned()
    }
}

/// Split debug-information is enabled by `-C split-debuginfo`, this enum is only used if split
/// debug-information is enabled (in either `Packed` or `Unpacked` modes), and the platform
/// uses DWARF for debug-information.
///
/// Some debug-information requires link-time relocation and some does not. LLVM can partition
/// the debuginfo into sections depending on whether or not it requires link-time relocation. Split
/// DWARF provides a mechanism which allows the linker to skip the sections which don't require
/// link-time relocation - either by putting those sections in DWARF object files, or by keeping
/// them in the object file in such a way that the linker will skip them.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum SplitDwarfKind {
    /// Sections which do not require relocation are written into object file but ignored by the
    /// linker.
    Single,
    /// Sections which do not require relocation are written into a DWARF object (`.dwo`) file
    /// which is ignored by the linker.
    Split,
}

impl FromStr for SplitDwarfKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        Ok(match s {
            "single" => SplitDwarfKind::Single,
            "split" => SplitDwarfKind::Split,
            _ => return Err(()),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, HashStable_Generic)]
#[derive(Encodable, Decodable)]
pub enum OutputType {
    Bitcode,
    Assembly,
    LlvmAssembly,
    Mir,
    Metadata,
    Object,
    Exe,
    DepInfo,
}

// Safety: Trivial C-Style enums have a stable sort order across compilation sessions.
unsafe impl StableOrd for OutputType {
    const CAN_USE_UNSTABLE_SORT: bool = true;
}

impl<HCX: HashStableContext> ToStableHashKey<HCX> for OutputType {
    type KeyType = Self;

    fn to_stable_hash_key(&self, _: &HCX) -> Self::KeyType {
        *self
    }
}

impl OutputType {
    fn is_compatible_with_codegen_units_and_single_output_file(&self) -> bool {
        match *self {
            OutputType::Exe | OutputType::DepInfo | OutputType::Metadata => true,
            OutputType::Bitcode
            | OutputType::Assembly
            | OutputType::LlvmAssembly
            | OutputType::Mir
            | OutputType::Object => false,
        }
    }

    pub fn shorthand(&self) -> &'static str {
        match *self {
            OutputType::Bitcode => "llvm-bc",
            OutputType::Assembly => "asm",
            OutputType::LlvmAssembly => "llvm-ir",
            OutputType::Mir => "mir",
            OutputType::Object => "obj",
            OutputType::Metadata => "metadata",
            OutputType::Exe => "link",
            OutputType::DepInfo => "dep-info",
        }
    }

    fn from_shorthand(shorthand: &str) -> Option<Self> {
        Some(match shorthand {
            "asm" => OutputType::Assembly,
            "llvm-ir" => OutputType::LlvmAssembly,
            "mir" => OutputType::Mir,
            "llvm-bc" => OutputType::Bitcode,
            "obj" => OutputType::Object,
            "metadata" => OutputType::Metadata,
            "link" => OutputType::Exe,
            "dep-info" => OutputType::DepInfo,
            _ => return None,
        })
    }

    fn shorthands_display() -> String {
        format!(
            "`{}`, `{}`, `{}`, `{}`, `{}`, `{}`, `{}`, `{}`",
            OutputType::Bitcode.shorthand(),
            OutputType::Assembly.shorthand(),
            OutputType::LlvmAssembly.shorthand(),
            OutputType::Mir.shorthand(),
            OutputType::Object.shorthand(),
            OutputType::Metadata.shorthand(),
            OutputType::Exe.shorthand(),
            OutputType::DepInfo.shorthand(),
        )
    }

    pub fn extension(&self) -> &'static str {
        match *self {
            OutputType::Bitcode => "bc",
            OutputType::Assembly => "s",
            OutputType::LlvmAssembly => "ll",
            OutputType::Mir => "mir",
            OutputType::Object => "o",
            OutputType::Metadata => "rmeta",
            OutputType::DepInfo => "d",
            OutputType::Exe => "",
        }
    }

    pub fn is_text_output(&self) -> bool {
        match *self {
            OutputType::Assembly
            | OutputType::LlvmAssembly
            | OutputType::Mir
            | OutputType::DepInfo => true,
            OutputType::Bitcode | OutputType::Object | OutputType::Metadata | OutputType::Exe => {
                false
            }
        }
    }
}

/// The type of diagnostics output to generate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorOutputType {
    /// Output meant for the consumption of humans.
    HumanReadable(HumanReadableErrorType),
    /// Output that's consumed by other tools such as `rustfix` or the `RLS`.
    Json {
        /// Render the JSON in a human readable way (with indents and newlines).
        pretty: bool,
        /// The JSON output includes a `rendered` field that includes the rendered
        /// human output.
        json_rendered: HumanReadableErrorType,
    },
}

impl Default for ErrorOutputType {
    fn default() -> Self {
        Self::HumanReadable(HumanReadableErrorType::Default(ColorConfig::Auto))
    }
}

#[derive(Clone, Hash, Debug)]
pub enum ResolveDocLinks {
    /// Do not resolve doc links.
    None,
    /// Resolve doc links on exported items only for crate types that have metadata.
    ExportedMetadata,
    /// Resolve doc links on exported items.
    Exported,
    /// Resolve doc links on all items.
    All,
}

/// Use tree-based collections to cheaply get a deterministic `Hash` implementation.
/// *Do not* switch `BTreeMap` out for an unsorted container type! That would break
/// dependency tracking for command-line arguments. Also only hash keys, since tracking
/// should only depend on the output types, not the paths they're written to.
#[derive(Clone, Debug, Hash, HashStable_Generic, Encodable, Decodable)]
pub struct OutputTypes(BTreeMap<OutputType, Option<OutFileName>>);

impl OutputTypes {
    pub fn new(entries: &[(OutputType, Option<OutFileName>)]) -> OutputTypes {
        OutputTypes(BTreeMap::from_iter(entries.iter().map(|&(k, ref v)| (k, v.clone()))))
    }

    pub fn get(&self, key: &OutputType) -> Option<&Option<OutFileName>> {
        self.0.get(key)
    }

    pub fn contains_key(&self, key: &OutputType) -> bool {
        self.0.contains_key(key)
    }

    pub fn iter(&self) -> BTreeMapIter<'_, OutputType, Option<OutFileName>> {
        self.0.iter()
    }

    pub fn keys(&self) -> BTreeMapKeysIter<'_, OutputType, Option<OutFileName>> {
        self.0.keys()
    }

    pub fn values(&self) -> BTreeMapValuesIter<'_, OutputType, Option<OutFileName>> {
        self.0.values()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if any of the output types require codegen or linking.
    pub fn should_codegen(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::Bitcode
            | OutputType::Assembly
            | OutputType::LlvmAssembly
            | OutputType::Mir
            | OutputType::Object
            | OutputType::Exe => true,
            OutputType::Metadata | OutputType::DepInfo => false,
        })
    }

    /// Returns `true` if any of the output types require linking.
    pub fn should_link(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::Bitcode
            | OutputType::Assembly
            | OutputType::LlvmAssembly
            | OutputType::Mir
            | OutputType::Metadata
            | OutputType::Object
            | OutputType::DepInfo => false,
            OutputType::Exe => true,
        })
    }
}

/// Use tree-based collections to cheaply get a deterministic `Hash` implementation.
/// *Do not* switch `BTreeMap` or `BTreeSet` out for an unsorted container type! That
/// would break dependency tracking for command-line arguments.
#[derive(Clone)]
pub struct Externs(BTreeMap<String, ExternEntry>);

#[derive(Clone, Debug)]
pub struct ExternEntry {
    pub location: ExternLocation,
    /// Indicates this is a "private" dependency for the
    /// `exported_private_dependencies` lint.
    ///
    /// This can be set with the `priv` option like
    /// `--extern priv:name=foo.rlib`.
    pub is_private_dep: bool,
    /// Add the extern entry to the extern prelude.
    ///
    /// This can be disabled with the `noprelude` option like
    /// `--extern noprelude:name`.
    pub add_prelude: bool,
    /// The extern entry shouldn't be considered for unused dependency warnings.
    ///
    /// `--extern nounused:std=/path/to/lib/libstd.rlib`. This is used to
    /// suppress `unused-crate-dependencies` warnings.
    pub nounused_dep: bool,
    /// If the extern entry is not referenced in the crate, force it to be resolved anyway.
    ///
    /// Allows a dependency satisfying, for instance, a missing panic handler to be injected
    /// without modifying source:
    /// `--extern force:extras=/path/to/lib/libstd.rlib`
    pub force: bool,
}

#[derive(Clone, Debug)]
pub enum ExternLocation {
    /// Indicates to look for the library in the search paths.
    ///
    /// Added via `--extern name`.
    FoundInLibrarySearchDirectories,
    /// The locations where this extern entry must be found.
    ///
    /// The `CrateLoader` is responsible for loading these and figuring out
    /// which one to use.
    ///
    /// Added via `--extern prelude_name=some_file.rlib`
    ExactPaths(BTreeSet<CanonicalizedPath>),
}

impl Externs {
    /// Used for testing.
    pub fn new(data: BTreeMap<String, ExternEntry>) -> Externs {
        Externs(data)
    }

    pub fn get(&self, key: &str) -> Option<&ExternEntry> {
        self.0.get(key)
    }

    pub fn iter(&self) -> BTreeMapIter<'_, String, ExternEntry> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl ExternEntry {
    fn new(location: ExternLocation) -> ExternEntry {
        ExternEntry {
            location,
            is_private_dep: false,
            add_prelude: false,
            nounused_dep: false,
            force: false,
        }
    }

    pub fn files(&self) -> Option<impl Iterator<Item = &CanonicalizedPath>> {
        match &self.location {
            ExternLocation::ExactPaths(set) => Some(set.iter()),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct PrintRequest {
    pub kind: PrintKind,
    pub out: OutFileName,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PrintKind {
    FileNames,
    Sysroot,
    TargetLibdir,
    CrateName,
    Cfg,
    CallingConventions,
    TargetList,
    TargetCPUs,
    TargetFeatures,
    RelocationModels,
    CodeModels,
    TlsModels,
    TargetSpec,
    AllTargetSpecs,
    NativeStaticLibs,
    StackProtectorStrategies,
    LinkArgs,
    SplitDebuginfo,
    DeploymentTarget,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct NextSolverConfig {
    /// Whether the new trait solver should be enabled in coherence.
    pub coherence: bool,
    /// Whether the new trait solver should be enabled everywhere.
    /// This is only `true` if `coherence` is also enabled.
    pub globally: bool,
    /// Whether to dump proof trees after computing a proof tree.
    pub dump_tree: DumpSolverProofTree,
}

#[derive(Default, Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DumpSolverProofTree {
    Always,
    OnError,
    #[default]
    Never,
}

pub enum Input {
    /// Load source code from a file.
    File(PathBuf),
    /// Load source code from a string.
    Str {
        /// A string that is shown in place of a filename.
        name: FileName,
        /// An anonymous string containing the source code.
        input: String,
    },
}

impl Input {
    pub fn filestem(&self) -> &str {
        match *self {
            Input::File(ref ifile) => ifile.file_stem().unwrap().to_str().unwrap(),
            Input::Str { .. } => "rust_out",
        }
    }

    pub fn source_name(&self) -> FileName {
        match *self {
            Input::File(ref ifile) => ifile.clone().into(),
            Input::Str { ref name, .. } => name.clone(),
        }
    }

    pub fn opt_path(&self) -> Option<&Path> {
        match self {
            Input::File(file) => Some(file),
            Input::Str { name, .. } => match name {
                FileName::Real(real) => real.local_path(),
                FileName::QuoteExpansion(_) => None,
                FileName::Anon(_) => None,
                FileName::MacroExpansion(_) => None,
                FileName::ProcMacroSourceCode(_) => None,
                FileName::CliCrateAttr(_) => None,
                FileName::Custom(_) => None,
                FileName::DocTest(path, _) => Some(path),
                FileName::InlineAsm(_) => None,
            },
        }
    }
}

#[derive(Clone, Hash, Debug, HashStable_Generic, PartialEq, Encodable, Decodable)]
pub enum OutFileName {
    Real(PathBuf),
    Stdout,
}

impl OutFileName {
    pub fn parent(&self) -> Option<&Path> {
        match *self {
            OutFileName::Real(ref path) => path.parent(),
            OutFileName::Stdout => None,
        }
    }

    pub fn filestem(&self) -> Option<&OsStr> {
        match *self {
            OutFileName::Real(ref path) => path.file_stem(),
            OutFileName::Stdout => Some(OsStr::new("stdout")),
        }
    }

    pub fn is_stdout(&self) -> bool {
        match *self {
            OutFileName::Real(_) => false,
            OutFileName::Stdout => true,
        }
    }

    pub fn is_tty(&self) -> bool {
        use std::io::IsTerminal;
        match *self {
            OutFileName::Real(_) => false,
            OutFileName::Stdout => std::io::stdout().is_terminal(),
        }
    }

    pub fn as_path(&self) -> &Path {
        match *self {
            OutFileName::Real(ref path) => path.as_ref(),
            OutFileName::Stdout => Path::new("stdout"),
        }
    }

    /// For a given output filename, return the actual name of the file that
    /// can be used to write codegen data of type `flavor`. For real-path
    /// output filenames, this would be trivial as we can just use the path.
    /// Otherwise for stdout, return a temporary path so that the codegen data
    /// may be later copied to stdout.
    pub fn file_for_writing(
        &self,
        outputs: &OutputFilenames,
        flavor: OutputType,
        codegen_unit_name: Option<&str>,
    ) -> PathBuf {
        match *self {
            OutFileName::Real(ref path) => path.clone(),
            OutFileName::Stdout => outputs.temp_path(flavor, codegen_unit_name),
        }
    }

    pub fn overwrite(&self, content: &str, sess: &Session) {
        match self {
            OutFileName::Stdout => print!("{content}"),
            OutFileName::Real(path) => {
                if let Err(e) = fs::write(path, content) {
                    sess.dcx().emit_fatal(FileWriteFail { path, err: e.to_string() });
                }
            }
        }
    }
}

#[derive(Clone, Hash, Debug, HashStable_Generic, Encodable, Decodable)]
pub struct OutputFilenames {
    pub out_directory: PathBuf,
    /// Crate name. Never contains '-'.
    crate_stem: String,
    /// Typically based on `.rs` input file name. Any '-' is preserved.
    filestem: String,
    pub single_output_file: Option<OutFileName>,
    pub temps_directory: Option<PathBuf>,
    pub outputs: OutputTypes,
}

pub const RLINK_EXT: &str = "rlink";
pub const RUST_CGU_EXT: &str = "rcgu";
pub const DWARF_OBJECT_EXT: &str = "dwo";

impl OutputFilenames {
    pub fn new(
        out_directory: PathBuf,
        out_crate_name: String,
        out_filestem: String,
        single_output_file: Option<OutFileName>,
        temps_directory: Option<PathBuf>,
        extra: String,
        outputs: OutputTypes,
    ) -> Self {
        OutputFilenames {
            out_directory,
            single_output_file,
            temps_directory,
            outputs,
            crate_stem: format!("{out_crate_name}{extra}"),
            filestem: format!("{out_filestem}{extra}"),
        }
    }

    pub fn path(&self, flavor: OutputType) -> OutFileName {
        self.outputs
            .get(&flavor)
            .and_then(|p| p.to_owned())
            .or_else(|| self.single_output_file.clone())
            .unwrap_or_else(|| OutFileName::Real(self.output_path(flavor)))
    }

    /// Gets the output path where a compilation artifact of the given type
    /// should be placed on disk.
    pub fn output_path(&self, flavor: OutputType) -> PathBuf {
        let extension = flavor.extension();
        match flavor {
            OutputType::Metadata => {
                self.out_directory.join(format!("lib{}.{}", self.crate_stem, extension))
            }
            _ => self.with_directory_and_extension(&self.out_directory, extension),
        }
    }

    /// Gets the path where a compilation artifact of the given type for the
    /// given codegen unit should be placed on disk. If codegen_unit_name is
    /// None, a path distinct from those of any codegen unit will be generated.
    pub fn temp_path(&self, flavor: OutputType, codegen_unit_name: Option<&str>) -> PathBuf {
        let extension = flavor.extension();
        self.temp_path_ext(extension, codegen_unit_name)
    }

    /// Like `temp_path`, but specifically for dwarf objects.
    pub fn temp_path_dwo(&self, codegen_unit_name: Option<&str>) -> PathBuf {
        self.temp_path_ext(DWARF_OBJECT_EXT, codegen_unit_name)
    }

    /// Like `temp_path`, but also supports things where there is no corresponding
    /// OutputType, like noopt-bitcode or lto-bitcode.
    pub fn temp_path_ext(&self, ext: &str, codegen_unit_name: Option<&str>) -> PathBuf {
        let mut extension = String::new();

        if let Some(codegen_unit_name) = codegen_unit_name {
            extension.push_str(codegen_unit_name);
        }

        if !ext.is_empty() {
            if !extension.is_empty() {
                extension.push('.');
                extension.push_str(RUST_CGU_EXT);
                extension.push('.');
            }

            extension.push_str(ext);
        }

        let temps_directory = self.temps_directory.as_ref().unwrap_or(&self.out_directory);

        self.with_directory_and_extension(temps_directory, &extension)
    }

    pub fn with_extension(&self, extension: &str) -> PathBuf {
        self.with_directory_and_extension(&self.out_directory, extension)
    }

    fn with_directory_and_extension(&self, directory: &PathBuf, extension: &str) -> PathBuf {
        let mut path = directory.join(&self.filestem);
        path.set_extension(extension);
        path
    }

    /// Returns the path for the Split DWARF file - this can differ depending on which Split DWARF
    /// mode is being used, which is the logic that this function is intended to encapsulate.
    pub fn split_dwarf_path(
        &self,
        split_debuginfo_kind: SplitDebuginfo,
        split_dwarf_kind: SplitDwarfKind,
        cgu_name: Option<&str>,
    ) -> Option<PathBuf> {
        let obj_out = self.temp_path(OutputType::Object, cgu_name);
        let dwo_out = self.temp_path_dwo(cgu_name);
        match (split_debuginfo_kind, split_dwarf_kind) {
            (SplitDebuginfo::Off, SplitDwarfKind::Single | SplitDwarfKind::Split) => None,
            // Single mode doesn't change how DWARF is emitted, but does add Split DWARF attributes
            // (pointing at the path which is being determined here). Use the path to the current
            // object file.
            (SplitDebuginfo::Packed | SplitDebuginfo::Unpacked, SplitDwarfKind::Single) => {
                Some(obj_out)
            }
            // Split mode emits the DWARF into a different file, use that path.
            (SplitDebuginfo::Packed | SplitDebuginfo::Unpacked, SplitDwarfKind::Split) => {
                Some(dwo_out)
            }
        }
    }
}

bitflags::bitflags! {
    /// Scopes used to determined if it need to apply to --remap-path-prefix
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct RemapPathScopeComponents: u8 {
        /// Apply remappings to the expansion of std::file!() macro
        const MACRO = 1 << 0;
        /// Apply remappings to printed compiler diagnostics
        const DIAGNOSTICS = 1 << 1;
        /// Apply remappings to debug information only when they are written to
        /// compiled executables or libraries, but not when they are in split
        /// debuginfo files
        const UNSPLIT_DEBUGINFO = 1 << 2;
        /// Apply remappings to debug information only when they are written to
        /// split debug information files, but not in compiled executables or
        /// libraries
        const SPLIT_DEBUGINFO = 1 << 3;
        /// Apply remappings to the paths pointing to split debug information
        /// files. Does nothing when these files are not generated.
        const SPLIT_DEBUGINFO_PATH = 1 << 4;

        /// An alias for macro,unsplit-debuginfo,split-debuginfo-path. This
        /// ensures all paths in compiled executables or libraries are remapped
        /// but not elsewhere.
        const OBJECT = Self::MACRO.bits() | Self::UNSPLIT_DEBUGINFO.bits() | Self::SPLIT_DEBUGINFO_PATH.bits();
    }
}

pub fn host_triple() -> &'static str {
    // Get the host triple out of the build environment. This ensures that our
    // idea of the host triple is the same as for the set of libraries we've
    // actually built. We can't just take LLVM's host triple because they
    // normalize all ix86 architectures to i386.
    //
    // Instead of grabbing the host triple (for the current host), we grab (at
    // compile time) the target triple that this rustc is built with and
    // calling that (at runtime) the host triple.
    (option_env!("CFG_COMPILER_HOST_TRIPLE")).expect("CFG_COMPILER_HOST_TRIPLE")
}

fn file_path_mapping(
    remap_path_prefix: Vec<(PathBuf, PathBuf)>,
    unstable_opts: &UnstableOptions,
) -> FilePathMapping {
    FilePathMapping::new(
        remap_path_prefix.clone(),
        if unstable_opts.remap_path_scope.contains(RemapPathScopeComponents::DIAGNOSTICS)
            && !remap_path_prefix.is_empty()
        {
            FileNameDisplayPreference::Remapped
        } else {
            FileNameDisplayPreference::Local
        },
    )
}

impl Default for Options {
    fn default() -> Options {
        Options {
            assert_incr_state: None,
            crate_types: Vec::new(),
            optimize: OptLevel::No,
            debuginfo: DebugInfo::None,
            debuginfo_compression: DebugInfoCompression::None,
            lint_opts: Vec::new(),
            lint_cap: None,
            describe_lints: false,
            output_types: OutputTypes(BTreeMap::new()),
            search_paths: vec![],
            maybe_sysroot: None,
            target_triple: TargetTriple::from_triple(host_triple()),
            test: false,
            incremental: None,
            untracked_state_hash: Default::default(),
            unstable_opts: Default::default(),
            prints: Vec::new(),
            cg: Default::default(),
            error_format: ErrorOutputType::default(),
            diagnostic_width: None,
            externs: Externs(BTreeMap::new()),
            crate_name: None,
            libs: Vec::new(),
            unstable_features: UnstableFeatures::Disallow,
            debug_assertions: true,
            actually_rustdoc: false,
            resolve_doc_links: ResolveDocLinks::None,
            trimmed_def_paths: false,
            cli_forced_codegen_units: None,
            cli_forced_local_thinlto_off: false,
            remap_path_prefix: Vec::new(),
            real_rust_source_base_dir: None,
            edition: DEFAULT_EDITION,
            json_artifact_notifications: false,
            json_unused_externs: JsonUnusedExterns::No,
            json_future_incompat: false,
            pretty: None,
            working_dir: RealFileName::LocalPath(std::env::current_dir().unwrap()),
            color: ColorConfig::Auto,
            logical_env: FxIndexMap::default(),
            verbose: false,
        }
    }
}

impl Options {
    /// Returns `true` if there is a reason to build the dep graph.
    pub fn build_dep_graph(&self) -> bool {
        self.incremental.is_some()
            || self.unstable_opts.dump_dep_graph
            || self.unstable_opts.query_dep_graph
    }

    pub fn file_path_mapping(&self) -> FilePathMapping {
        file_path_mapping(self.remap_path_prefix.clone(), &self.unstable_opts)
    }

    /// Returns `true` if there will be an output file generated.
    pub fn will_create_output_file(&self) -> bool {
        !self.unstable_opts.parse_only && // The file is just being parsed
            self.unstable_opts.ls.is_empty() // The file is just being queried
    }

    #[inline]
    pub fn share_generics(&self) -> bool {
        match self.unstable_opts.share_generics {
            Some(setting) => setting,
            None => match self.optimize {
                OptLevel::No | OptLevel::Less | OptLevel::Size | OptLevel::SizeMin => true,
                OptLevel::Default | OptLevel::Aggressive => false,
            },
        }
    }

    pub fn get_symbol_mangling_version(&self) -> SymbolManglingVersion {
        self.cg.symbol_mangling_version.unwrap_or(SymbolManglingVersion::Legacy)
    }
}

impl UnstableOptions {
    pub fn dcx_flags(&self, can_emit_warnings: bool) -> DiagCtxtFlags {
        DiagCtxtFlags {
            can_emit_warnings,
            treat_err_as_bug: self.treat_err_as_bug,
            eagerly_emit_delayed_bugs: self.eagerly_emit_delayed_bugs,
            macro_backtrace: self.macro_backtrace,
            deduplicate_diagnostics: self.deduplicate_diagnostics,
            track_diagnostics: self.track_diagnostics,
        }
    }
}

// The type of entry function, so users can have their own entry functions
#[derive(Copy, Clone, PartialEq, Hash, Debug, HashStable_Generic)]
pub enum EntryFnType {
    Main {
        /// Specifies what to do with `SIGPIPE` before calling `fn main()`.
        ///
        /// What values that are valid and what they mean must be in sync
        /// across rustc and libstd, but we don't want it public in libstd,
        /// so we take a bit of an unusual approach with simple constants
        /// and an `include!()`.
        sigpipe: u8,
    },
    Start,
}

#[derive(Copy, PartialEq, PartialOrd, Clone, Ord, Eq, Hash, Debug, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub enum CrateType {
    Executable,
    Dylib,
    Rlib,
    Staticlib,
    Cdylib,
    ProcMacro,
}

impl CrateType {
    pub fn has_metadata(self) -> bool {
        match self {
            CrateType::Rlib | CrateType::Dylib | CrateType::ProcMacro => true,
            CrateType::Executable | CrateType::Cdylib | CrateType::Staticlib => false,
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum Passes {
    Some(Vec<String>),
    All,
}

impl Passes {
    pub fn is_empty(&self) -> bool {
        match *self {
            Passes::Some(ref v) => v.is_empty(),
            Passes::All => false,
        }
    }

    pub fn extend(&mut self, passes: impl IntoIterator<Item = String>) {
        match *self {
            Passes::Some(ref mut v) => v.extend(passes),
            Passes::All => {}
        }
    }
}

#[derive(Clone, Copy, Hash, Debug, PartialEq)]
pub enum PAuthKey {
    A,
    B,
}

#[derive(Clone, Copy, Hash, Debug, PartialEq)]
pub struct PacRet {
    pub leaf: bool,
    pub key: PAuthKey,
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Default)]
pub struct BranchProtection {
    pub bti: bool,
    pub pac_ret: Option<PacRet>,
}

pub const fn default_lib_output() -> CrateType {
    CrateType::Rlib
}

fn default_configuration(sess: &Session) -> Cfg {
    let mut ret = Cfg::default();

    macro_rules! ins_none {
        ($key:expr) => {
            ret.insert(($key, None));
        };
    }
    macro_rules! ins_str {
        ($key:expr, $val_str:expr) => {
            ret.insert(($key, Some(Symbol::intern($val_str))));
        };
    }
    macro_rules! ins_sym {
        ($key:expr, $val_sym:expr) => {
            ret.insert(($key, Some($val_sym)));
        };
    }

    // Symbols are inserted in alphabetical order as much as possible.
    // The exceptions are where control flow forces things out of order.
    //
    // Run `rustc --print cfg` to see the configuration in practice.
    //
    // NOTE: These insertions should be kept in sync with
    // `CheckCfg::fill_well_known` below.

    if sess.opts.debug_assertions {
        ins_none!(sym::debug_assertions);
    }

    if sess.overflow_checks() {
        ins_none!(sym::overflow_checks);
    }

    ins_sym!(sym::panic, sess.panic_strategy().desc_symbol());

    // JUSTIFICATION: before wrapper fn is available
    #[allow(rustc::bad_opt_access)]
    if sess.opts.crate_types.contains(&CrateType::ProcMacro) {
        ins_none!(sym::proc_macro);
    }

    if sess.is_nightly_build() {
        ins_sym!(sym::relocation_model, sess.target.relocation_model.desc_symbol());
    }

    for mut s in sess.opts.unstable_opts.sanitizer {
        // KASAN is still ASAN under the hood, so it uses the same attribute.
        if s == SanitizerSet::KERNELADDRESS {
            s = SanitizerSet::ADDRESS;
        }
        ins_str!(sym::sanitize, &s.to_string());
    }

    if sess.is_sanitizer_cfi_generalize_pointers_enabled() {
        ins_none!(sym::sanitizer_cfi_generalize_pointers);
    }
    if sess.is_sanitizer_cfi_normalize_integers_enabled() {
        ins_none!(sym::sanitizer_cfi_normalize_integers);
    }

    ins_str!(sym::target_abi, &sess.target.abi);
    ins_str!(sym::target_arch, &sess.target.arch);
    ins_str!(sym::target_endian, sess.target.endian.as_str());
    ins_str!(sym::target_env, &sess.target.env);

    for family in sess.target.families.as_ref() {
        ins_str!(sym::target_family, family);
        if family == "windows" {
            ins_none!(sym::windows);
        } else if family == "unix" {
            ins_none!(sym::unix);
        }
    }

    // `target_has_atomic*`
    let layout = sess.target.parse_data_layout().unwrap_or_else(|err| {
        sess.dcx().emit_fatal(err);
    });
    let mut has_atomic = false;
    for (i, align) in [
        (8, layout.i8_align.abi),
        (16, layout.i16_align.abi),
        (32, layout.i32_align.abi),
        (64, layout.i64_align.abi),
        (128, layout.i128_align.abi),
    ] {
        if i >= sess.target.min_atomic_width() && i <= sess.target.max_atomic_width() {
            if !has_atomic {
                has_atomic = true;
                if sess.is_nightly_build() {
                    if sess.target.atomic_cas {
                        ins_none!(sym::target_has_atomic);
                    }
                    ins_none!(sym::target_has_atomic_load_store);
                }
            }
            let mut insert_atomic = |sym, align: Align| {
                if sess.target.atomic_cas {
                    ins_sym!(sym::target_has_atomic, sym);
                }
                if align.bits() == i {
                    ins_sym!(sym::target_has_atomic_equal_alignment, sym);
                }
                ins_sym!(sym::target_has_atomic_load_store, sym);
            };
            insert_atomic(sym::integer(i), align);
            if sess.target.pointer_width as u64 == i {
                insert_atomic(sym::ptr, layout.pointer_align.abi);
            }
        }
    }

    ins_str!(sym::target_os, &sess.target.os);
    ins_sym!(sym::target_pointer_width, sym::integer(sess.target.pointer_width));

    if sess.opts.unstable_opts.has_thread_local.unwrap_or(sess.target.has_thread_local) {
        ins_none!(sym::target_thread_local);
    }

    ins_str!(sym::target_vendor, &sess.target.vendor);

    // If the user wants a test runner, then add the test cfg.
    if sess.is_test_crate() {
        ins_none!(sym::test);
    }

    ret
}

/// The parsed `--cfg` options that define the compilation environment of the
/// crate, used to drive conditional compilation.
///
/// An `FxIndexSet` is used to ensure deterministic ordering of error messages
/// relating to `--cfg`.
pub type Cfg = FxIndexSet<(Symbol, Option<Symbol>)>;

/// The parsed `--check-cfg` options.
#[derive(Default)]
pub struct CheckCfg {
    /// Is well known names activated
    pub exhaustive_names: bool,
    /// Is well known values activated
    pub exhaustive_values: bool,
    /// All the expected values for a config name
    pub expecteds: FxHashMap<Symbol, ExpectedValues<Symbol>>,
    /// Well known names (only used for diagnostics purposes)
    pub well_known_names: FxHashSet<Symbol>,
}

pub enum ExpectedValues<T> {
    Some(FxHashSet<Option<T>>),
    Any,
}

impl<T: Eq + Hash> ExpectedValues<T> {
    fn insert(&mut self, value: T) -> bool {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.insert(Some(value)),
            ExpectedValues::Any => false,
        }
    }
}

impl<T: Eq + Hash> Extend<T> for ExpectedValues<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.extend(iter.into_iter().map(Some)),
            ExpectedValues::Any => {}
        }
    }
}

impl<'a, T: Eq + Hash + Copy + 'a> Extend<&'a T> for ExpectedValues<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.extend(iter.into_iter().map(|a| Some(*a))),
            ExpectedValues::Any => {}
        }
    }
}

impl CheckCfg {
    pub fn fill_well_known(&mut self, current_target: &Target) {
        if !self.exhaustive_values && !self.exhaustive_names {
            return;
        }

        let no_values = || {
            let mut values = FxHashSet::default();
            values.insert(None);
            ExpectedValues::Some(values)
        };

        let empty_values = || {
            let values = FxHashSet::default();
            ExpectedValues::Some(values)
        };

        macro_rules! ins {
            ($name:expr, $values:expr) => {{
                self.well_known_names.insert($name);
                self.expecteds.entry($name).or_insert_with($values)
            }};
        }

        // Symbols are inserted in alphabetical order as much as possible.
        // The exceptions are where control flow forces things out of order.
        //
        // NOTE: This should be kept in sync with `default_configuration`.
        // Note that symbols inserted conditionally in `default_configuration`
        // are inserted unconditionally here.
        //
        // When adding a new config here you should also update
        // `tests/ui/check-cfg/well-known-values.rs`.

        ins!(sym::debug_assertions, no_values);

        // These three are never set by rustc, but we set them anyway: they
        // should not trigger a lint because `cargo doc`, `cargo test`, and
        // `cargo miri run` (respectively) can set them.
        ins!(sym::doc, no_values);
        ins!(sym::doctest, no_values);
        ins!(sym::miri, no_values);

        ins!(sym::overflow_checks, no_values);

        ins!(sym::panic, empty_values).extend(&PanicStrategy::all());

        ins!(sym::proc_macro, no_values);

        ins!(sym::relocation_model, empty_values).extend(RelocModel::all());

        let sanitize_values = SanitizerSet::all()
            .into_iter()
            .map(|sanitizer| Symbol::intern(sanitizer.as_str().unwrap()));
        ins!(sym::sanitize, empty_values).extend(sanitize_values);

        ins!(sym::sanitizer_cfi_generalize_pointers, no_values);
        ins!(sym::sanitizer_cfi_normalize_integers, no_values);

        ins!(sym::target_feature, empty_values).extend(
            rustc_target::target_features::all_known_features()
                .map(|(f, _sb)| f)
                .chain(rustc_target::target_features::RUSTC_SPECIFIC_FEATURES.iter().cloned())
                .map(Symbol::intern),
        );

        // sym::target_*
        {
            const VALUES: [&Symbol; 8] = [
                &sym::target_abi,
                &sym::target_arch,
                &sym::target_endian,
                &sym::target_env,
                &sym::target_family,
                &sym::target_os,
                &sym::target_pointer_width,
                &sym::target_vendor,
            ];

            // Initialize (if not already initialized)
            for &e in VALUES {
                if !self.exhaustive_values {
                    ins!(e, || ExpectedValues::Any);
                } else {
                    ins!(e, empty_values);
                }
            }

            if self.exhaustive_values {
                // Get all values map at once otherwise it would be costly.
                // (8 values * 220 targets ~= 1760 times, at the time of writing this comment).
                let [
                    values_target_abi,
                    values_target_arch,
                    values_target_endian,
                    values_target_env,
                    values_target_family,
                    values_target_os,
                    values_target_pointer_width,
                    values_target_vendor,
                ] = self
                    .expecteds
                    .get_many_mut(VALUES)
                    .expect("unable to get all the check-cfg values buckets");

                for target in TARGETS
                    .iter()
                    .map(|target| Target::expect_builtin(&TargetTriple::from_triple(target)))
                    .chain(iter::once(current_target.clone()))
                {
                    values_target_abi.insert(Symbol::intern(&target.options.abi));
                    values_target_arch.insert(Symbol::intern(&target.arch));
                    values_target_endian.insert(Symbol::intern(target.options.endian.as_str()));
                    values_target_env.insert(Symbol::intern(&target.options.env));
                    values_target_family.extend(
                        target.options.families.iter().map(|family| Symbol::intern(family)),
                    );
                    values_target_os.insert(Symbol::intern(&target.options.os));
                    values_target_pointer_width.insert(sym::integer(target.pointer_width));
                    values_target_vendor.insert(Symbol::intern(&target.options.vendor));
                }
            }
        }

        let atomic_values = &[
            sym::ptr,
            sym::integer(8usize),
            sym::integer(16usize),
            sym::integer(32usize),
            sym::integer(64usize),
            sym::integer(128usize),
        ];
        for sym in [
            sym::target_has_atomic,
            sym::target_has_atomic_equal_alignment,
            sym::target_has_atomic_load_store,
        ] {
            ins!(sym, no_values).extend(atomic_values);
        }

        ins!(sym::target_thread_local, no_values);

        ins!(sym::test, no_values);

        ins!(sym::unix, no_values);
        ins!(sym::windows, no_values);
    }
}

pub fn build_configuration(sess: &Session, mut user_cfg: Cfg) -> Cfg {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items.
    user_cfg.extend(default_configuration(sess));
    user_cfg
}

pub(super) fn build_target_config(
    early_dcx: &EarlyDiagCtxt,
    opts: &Options,
    target_override: Option<Target>,
    sysroot: &Path,
) -> Target {
    let target_result = target_override.map_or_else(
        || Target::search(&opts.target_triple, sysroot),
        |t| Ok((t, TargetWarnings::empty())),
    );
    let (target, target_warnings) = target_result.unwrap_or_else(|e| {
        early_dcx.early_fatal(format!(
            "Error loading target specification: {e}. \
                 Run `rustc --print target-list` for a list of built-in targets"
        ))
    });
    for warning in target_warnings.warning_messages() {
        early_dcx.early_warn(warning)
    }

    if !matches!(target.pointer_width, 16 | 32 | 64) {
        early_dcx.early_fatal(format!(
            "target specification was invalid: unrecognized target-pointer-width {}",
            target.pointer_width
        ))
    }

    target
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OptionStability {
    Stable,
    Unstable,
}

pub struct RustcOptGroup {
    pub apply: Box<dyn Fn(&mut getopts::Options) -> &mut getopts::Options>,
    pub name: &'static str,
    pub stability: OptionStability,
}

impl RustcOptGroup {
    pub fn is_stable(&self) -> bool {
        self.stability == OptionStability::Stable
    }

    pub fn stable<F>(name: &'static str, f: F) -> RustcOptGroup
    where
        F: Fn(&mut getopts::Options) -> &mut getopts::Options + 'static,
    {
        RustcOptGroup { name, apply: Box::new(f), stability: OptionStability::Stable }
    }

    pub fn unstable<F>(name: &'static str, f: F) -> RustcOptGroup
    where
        F: Fn(&mut getopts::Options) -> &mut getopts::Options + 'static,
    {
        RustcOptGroup { name, apply: Box::new(f), stability: OptionStability::Unstable }
    }
}

// The `opt` local module holds wrappers around the `getopts` API that
// adds extra rustc-specific metadata to each option; such metadata
// is exposed by . The public
// functions below ending with `_u` are the functions that return
// *unstable* options, i.e., options that are only enabled when the
// user also passes the `-Z unstable-options` debugging flag.
mod opt {
    // The `fn flag*` etc below are written so that we can use them
    // in the future; do not warn about them not being used right now.
    #![allow(dead_code)]

    use super::RustcOptGroup;

    pub type R = RustcOptGroup;
    pub type S = &'static str;

    fn stable<F>(name: S, f: F) -> R
    where
        F: Fn(&mut getopts::Options) -> &mut getopts::Options + 'static,
    {
        RustcOptGroup::stable(name, f)
    }

    fn unstable<F>(name: S, f: F) -> R
    where
        F: Fn(&mut getopts::Options) -> &mut getopts::Options + 'static,
    {
        RustcOptGroup::unstable(name, f)
    }

    fn longer(a: S, b: S) -> S {
        if a.len() > b.len() { a } else { b }
    }

    pub fn opt_s(a: S, b: S, c: S, d: S) -> R {
        stable(longer(a, b), move |opts| opts.optopt(a, b, c, d))
    }
    pub fn multi_s(a: S, b: S, c: S, d: S) -> R {
        stable(longer(a, b), move |opts| opts.optmulti(a, b, c, d))
    }
    pub fn flag_s(a: S, b: S, c: S) -> R {
        stable(longer(a, b), move |opts| opts.optflag(a, b, c))
    }
    pub fn flagmulti_s(a: S, b: S, c: S) -> R {
        stable(longer(a, b), move |opts| opts.optflagmulti(a, b, c))
    }

    pub fn opt(a: S, b: S, c: S, d: S) -> R {
        unstable(longer(a, b), move |opts| opts.optopt(a, b, c, d))
    }
    pub fn multi(a: S, b: S, c: S, d: S) -> R {
        unstable(longer(a, b), move |opts| opts.optmulti(a, b, c, d))
    }
}
static EDITION_STRING: LazyLock<String> = LazyLock::new(|| {
    format!(
        "Specify which edition of the compiler to use when compiling code. \
The default is {DEFAULT_EDITION} and the latest stable edition is {LATEST_STABLE_EDITION}."
    )
});
/// Returns the "short" subset of the rustc command line options,
/// including metadata for each option, such as whether the option is
/// part of the stable long-term interface for rustc.
pub fn rustc_short_optgroups() -> Vec<RustcOptGroup> {
    vec![
        opt::flag_s("h", "help", "Display this message"),
        opt::multi_s("", "cfg", "Configure the compilation environment.
                             SPEC supports the syntax `NAME[=\"VALUE\"]`.", "SPEC"),
        opt::multi("", "check-cfg", "Provide list of valid cfg options for checking", "SPEC"),
        opt::multi_s(
            "L",
            "",
            "Add a directory to the library search path. The
                             optional KIND can be one of dependency, crate, native,
                             framework, or all (the default).",
            "[KIND=]PATH",
        ),
        opt::multi_s(
            "l",
            "",
            "Link the generated crate(s) to the specified native
                             library NAME. The optional KIND can be one of
                             static, framework, or dylib (the default).
                             Optional comma separated MODIFIERS (bundle|verbatim|whole-archive|as-needed)
                             may be specified each with a prefix of either '+' to
                             enable or '-' to disable.",
            "[KIND[:MODIFIERS]=]NAME[:RENAME]",
        ),
        make_crate_type_option(),
        opt::opt_s("", "crate-name", "Specify the name of the crate being built", "NAME"),
        opt::opt_s(
            "",
            "edition",
            &EDITION_STRING,
            EDITION_NAME_LIST,
        ),
        opt::multi_s(
            "",
            "emit",
            "Comma separated list of types of output for \
             the compiler to emit",
            "[asm|llvm-bc|llvm-ir|obj|metadata|link|dep-info|mir]",
        ),
        opt::multi_s(
            "",
            "print",
            "Compiler information to print on stdout",
            "[crate-name|file-names|sysroot|target-libdir|cfg|calling-conventions|\
             target-list|target-cpus|target-features|relocation-models|code-models|\
             tls-models|target-spec-json|all-target-specs-json|native-static-libs|\
             stack-protector-strategies|link-args|deployment-target]",
        ),
        opt::flagmulti_s("g", "", "Equivalent to -C debuginfo=2"),
        opt::flagmulti_s("O", "", "Equivalent to -C opt-level=2"),
        opt::opt_s("o", "", "Write output to <filename>", "FILENAME"),
        opt::opt_s(
            "",
            "out-dir",
            "Write output to compiler-chosen filename \
             in <dir>",
            "DIR",
        ),
        opt::opt_s(
            "",
            "explain",
            "Provide a detailed explanation of an error \
             message",
            "OPT",
        ),
        opt::flag_s("", "test", "Build a test harness"),
        opt::opt_s("", "target", "Target triple for which the code is compiled", "TARGET"),
        opt::multi_s("A", "allow", "Set lint allowed", "LINT"),
        opt::multi_s("W", "warn", "Set lint warnings", "LINT"),
        opt::multi_s("", "force-warn", "Set lint force-warn", "LINT"),
        opt::multi_s("D", "deny", "Set lint denied", "LINT"),
        opt::multi_s("F", "forbid", "Set lint forbidden", "LINT"),
        opt::multi_s(
            "",
            "cap-lints",
            "Set the most restrictive lint level. \
             More restrictive lints are capped at this \
             level",
            "LEVEL",
        ),
        opt::multi_s("C", "codegen", "Set a codegen option", "OPT[=VALUE]"),
        opt::flag_s("V", "version", "Print version info and exit"),
        opt::flag_s("v", "verbose", "Use verbose output"),
    ]
}

/// Returns all rustc command line options, including metadata for
/// each option, such as whether the option is part of the stable
/// long-term interface for rustc.
pub fn rustc_optgroups() -> Vec<RustcOptGroup> {
    let mut opts = rustc_short_optgroups();
    // FIXME: none of these descriptions are actually used
    opts.extend(vec![
        opt::multi_s(
            "",
            "extern",
            "Specify where an external rust library is located",
            "NAME[=PATH]",
        ),
        opt::opt_s("", "sysroot", "Override the system root", "PATH"),
        opt::multi("Z", "", "Set unstable / perma-unstable options", "FLAG"),
        opt::opt_s(
            "",
            "error-format",
            "How errors and other messages are produced",
            "human|json|short",
        ),
        opt::multi_s("", "json", "Configure the JSON output of the compiler", "CONFIG"),
        opt::opt_s(
            "",
            "color",
            "Configure coloring of output:
                                 auto   = colorize, if output goes to a tty (default);
                                 always = always colorize output;
                                 never  = never colorize output",
            "auto|always|never",
        ),
        opt::opt_s(
            "",
            "diagnostic-width",
            "Inform rustc of the width of the output so that diagnostics can be truncated to fit",
            "WIDTH",
        ),
        opt::multi_s(
            "",
            "remap-path-prefix",
            "Remap source names in all output (compiler messages and output files)",
            "FROM=TO",
        ),
        opt::multi("", "env-set", "Inject an environment variable", "VAR=VALUE"),
    ]);
    opts
}

pub fn get_cmd_lint_options(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
) -> (Vec<(String, lint::Level)>, bool, Option<lint::Level>) {
    let mut lint_opts_with_position = vec![];
    let mut describe_lints = false;

    for level in [lint::Allow, lint::Warn, lint::ForceWarn(None), lint::Deny, lint::Forbid] {
        for (arg_pos, lint_name) in matches.opt_strs_pos(level.as_str()) {
            if lint_name == "help" {
                describe_lints = true;
            } else {
                lint_opts_with_position.push((arg_pos, lint_name.replace('-', "_"), level));
            }
        }
    }

    lint_opts_with_position.sort_by_key(|x| x.0);
    let lint_opts = lint_opts_with_position
        .iter()
        .cloned()
        .map(|(_, lint_name, level)| (lint_name, level))
        .collect();

    let lint_cap = matches.opt_str("cap-lints").map(|cap| {
        lint::Level::from_str(&cap)
            .unwrap_or_else(|| early_dcx.early_fatal(format!("unknown lint level: `{cap}`")))
    });

    (lint_opts, describe_lints, lint_cap)
}

/// Parses the `--color` flag.
pub fn parse_color(early_dcx: &EarlyDiagCtxt, matches: &getopts::Matches) -> ColorConfig {
    match matches.opt_str("color").as_deref() {
        Some("auto") => ColorConfig::Auto,
        Some("always") => ColorConfig::Always,
        Some("never") => ColorConfig::Never,

        None => ColorConfig::Auto,

        Some(arg) => early_dcx.early_fatal(format!(
            "argument for `--color` must be auto, \
                 always or never (instead was `{arg}`)"
        )),
    }
}

/// Possible json config files
pub struct JsonConfig {
    pub json_rendered: HumanReadableErrorType,
    pub json_artifact_notifications: bool,
    pub json_unused_externs: JsonUnusedExterns,
    pub json_future_incompat: bool,
}

/// Report unused externs in event stream
#[derive(Copy, Clone)]
pub enum JsonUnusedExterns {
    /// Do not
    No,
    /// Report, but do not exit with failure status for deny/forbid
    Silent,
    /// Report, and also exit with failure status for deny/forbid
    Loud,
}

impl JsonUnusedExterns {
    pub fn is_enabled(&self) -> bool {
        match self {
            JsonUnusedExterns::No => false,
            JsonUnusedExterns::Loud | JsonUnusedExterns::Silent => true,
        }
    }

    pub fn is_loud(&self) -> bool {
        match self {
            JsonUnusedExterns::No | JsonUnusedExterns::Silent => false,
            JsonUnusedExterns::Loud => true,
        }
    }
}

/// Parse the `--json` flag.
///
/// The first value returned is how to render JSON diagnostics, and the second
/// is whether or not artifact notifications are enabled.
pub fn parse_json(early_dcx: &EarlyDiagCtxt, matches: &getopts::Matches) -> JsonConfig {
    let mut json_rendered: fn(ColorConfig) -> HumanReadableErrorType =
        HumanReadableErrorType::Default;
    let mut json_color = ColorConfig::Never;
    let mut json_artifact_notifications = false;
    let mut json_unused_externs = JsonUnusedExterns::No;
    let mut json_future_incompat = false;
    for option in matches.opt_strs("json") {
        // For now conservatively forbid `--color` with `--json` since `--json`
        // won't actually be emitting any colors and anything colorized is
        // embedded in a diagnostic message anyway.
        if matches.opt_str("color").is_some() {
            early_dcx.early_fatal("cannot specify the `--color` option with `--json`");
        }

        for sub_option in option.split(',') {
            match sub_option {
                "diagnostic-short" => json_rendered = HumanReadableErrorType::Short,
                "diagnostic-rendered-ansi" => json_color = ColorConfig::Always,
                "artifacts" => json_artifact_notifications = true,
                "unused-externs" => json_unused_externs = JsonUnusedExterns::Loud,
                "unused-externs-silent" => json_unused_externs = JsonUnusedExterns::Silent,
                "future-incompat" => json_future_incompat = true,
                s => early_dcx.early_fatal(format!("unknown `--json` option `{s}`")),
            }
        }
    }

    JsonConfig {
        json_rendered: json_rendered(json_color),
        json_artifact_notifications,
        json_unused_externs,
        json_future_incompat,
    }
}

/// Parses the `--error-format` flag.
pub fn parse_error_format(
    early_dcx: &mut EarlyDiagCtxt,
    matches: &getopts::Matches,
    color: ColorConfig,
    json_rendered: HumanReadableErrorType,
) -> ErrorOutputType {
    // We need the `opts_present` check because the driver will send us Matches
    // with only stable options if no unstable options are used. Since error-format
    // is unstable, it will not be present. We have to use `opts_present` not
    // `opt_present` because the latter will panic.
    let error_format = if matches.opts_present(&["error-format".to_owned()]) {
        match matches.opt_str("error-format").as_deref() {
            None | Some("human") => {
                ErrorOutputType::HumanReadable(HumanReadableErrorType::Default(color))
            }
            Some("human-annotate-rs") => {
                ErrorOutputType::HumanReadable(HumanReadableErrorType::AnnotateSnippet(color))
            }
            Some("json") => ErrorOutputType::Json { pretty: false, json_rendered },
            Some("pretty-json") => ErrorOutputType::Json { pretty: true, json_rendered },
            Some("short") => ErrorOutputType::HumanReadable(HumanReadableErrorType::Short(color)),

            Some(arg) => {
                early_dcx.abort_if_error_and_set_error_format(ErrorOutputType::HumanReadable(
                    HumanReadableErrorType::Default(color),
                ));
                early_dcx.early_fatal(format!(
                    "argument for `--error-format` must be `human`, `json` or \
                     `short` (instead was `{arg}`)"
                ))
            }
        }
    } else {
        ErrorOutputType::HumanReadable(HumanReadableErrorType::Default(color))
    };

    match error_format {
        ErrorOutputType::Json { .. } => {}

        // Conservatively require that the `--json` argument is coupled with
        // `--error-format=json`. This means that `--json` is specified we
        // should actually be emitting JSON blobs.
        _ if !matches.opt_strs("json").is_empty() => {
            early_dcx.early_fatal("using `--json` requires also using `--error-format=json`");
        }

        _ => {}
    }

    error_format
}

pub fn parse_crate_edition(early_dcx: &EarlyDiagCtxt, matches: &getopts::Matches) -> Edition {
    let edition = match matches.opt_str("edition") {
        Some(arg) => Edition::from_str(&arg).unwrap_or_else(|_| {
            early_dcx.early_fatal(format!(
                "argument for `--edition` must be one of: \
                     {EDITION_NAME_LIST}. (instead was `{arg}`)"
            ))
        }),
        None => DEFAULT_EDITION,
    };

    if !edition.is_stable() && !nightly_options::is_unstable_enabled(matches) {
        let is_nightly = nightly_options::match_is_nightly_build(matches);
        let msg = if !is_nightly {
            format!(
                "the crate requires edition {edition}, but the latest edition supported by this Rust version is {LATEST_STABLE_EDITION}"
            )
        } else {
            format!("edition {edition} is unstable and only available with -Z unstable-options")
        };
        early_dcx.early_fatal(msg)
    }

    edition
}

fn check_error_format_stability(
    early_dcx: &mut EarlyDiagCtxt,
    unstable_opts: &UnstableOptions,
    error_format: ErrorOutputType,
) {
    if !unstable_opts.unstable_options {
        if let ErrorOutputType::Json { pretty: true, .. } = error_format {
            early_dcx.early_fatal("`--error-format=pretty-json` is unstable");
        }
        if let ErrorOutputType::HumanReadable(HumanReadableErrorType::AnnotateSnippet(_)) =
            error_format
        {
            early_dcx.early_fatal("`--error-format=human-annotate-rs` is unstable");
        }
    }
}

fn parse_output_types(
    early_dcx: &EarlyDiagCtxt,
    unstable_opts: &UnstableOptions,
    matches: &getopts::Matches,
) -> OutputTypes {
    let mut output_types = BTreeMap::new();
    if !unstable_opts.parse_only {
        for list in matches.opt_strs("emit") {
            for output_type in list.split(',') {
                let (shorthand, path) = split_out_file_name(output_type);
                let output_type = OutputType::from_shorthand(shorthand).unwrap_or_else(|| {
                    early_dcx.early_fatal(format!(
                        "unknown emission type: `{shorthand}` - expected one of: {display}",
                        display = OutputType::shorthands_display(),
                    ))
                });
                output_types.insert(output_type, path);
            }
        }
    };
    if output_types.is_empty() {
        output_types.insert(OutputType::Exe, None);
    }
    OutputTypes(output_types)
}

fn split_out_file_name(arg: &str) -> (&str, Option<OutFileName>) {
    match arg.split_once('=') {
        None => (arg, None),
        Some((kind, "-")) => (kind, Some(OutFileName::Stdout)),
        Some((kind, path)) => (kind, Some(OutFileName::Real(PathBuf::from(path)))),
    }
}

fn should_override_cgus_and_disable_thinlto(
    early_dcx: &EarlyDiagCtxt,
    output_types: &OutputTypes,
    matches: &getopts::Matches,
    mut codegen_units: Option<usize>,
) -> (bool, Option<usize>) {
    let mut disable_local_thinlto = false;
    // Issue #30063: if user requests LLVM-related output to one
    // particular path, disable codegen-units.
    let incompatible: Vec<_> = output_types
        .0
        .iter()
        .map(|ot_path| ot_path.0)
        .filter(|ot| !ot.is_compatible_with_codegen_units_and_single_output_file())
        .map(|ot| ot.shorthand())
        .collect();
    if !incompatible.is_empty() {
        match codegen_units {
            Some(n) if n > 1 => {
                if matches.opt_present("o") {
                    for ot in &incompatible {
                        early_dcx.early_warn(format!(
                            "`--emit={ot}` with `-o` incompatible with \
                                 `-C codegen-units=N` for N > 1",
                        ));
                    }
                    early_dcx.early_warn("resetting to default -C codegen-units=1");
                    codegen_units = Some(1);
                    disable_local_thinlto = true;
                }
            }
            _ => {
                codegen_units = Some(1);
                disable_local_thinlto = true;
            }
        }
    }

    if codegen_units == Some(0) {
        early_dcx.early_fatal("value for codegen units must be a positive non-zero integer");
    }

    (disable_local_thinlto, codegen_units)
}

fn collect_print_requests(
    early_dcx: &EarlyDiagCtxt,
    cg: &mut CodegenOptions,
    unstable_opts: &mut UnstableOptions,
    matches: &getopts::Matches,
) -> Vec<PrintRequest> {
    let mut prints = Vec::<PrintRequest>::new();
    if cg.target_cpu.as_ref().is_some_and(|s| s == "help") {
        prints.push(PrintRequest { kind: PrintKind::TargetCPUs, out: OutFileName::Stdout });
        cg.target_cpu = None;
    };
    if cg.target_feature == "help" {
        prints.push(PrintRequest { kind: PrintKind::TargetFeatures, out: OutFileName::Stdout });
        cg.target_feature = String::new();
    }

    const PRINT_KINDS: &[(&str, PrintKind)] = &[
        // tidy-alphabetical-start
        ("all-target-specs-json", PrintKind::AllTargetSpecs),
        ("calling-conventions", PrintKind::CallingConventions),
        ("cfg", PrintKind::Cfg),
        ("code-models", PrintKind::CodeModels),
        ("crate-name", PrintKind::CrateName),
        ("deployment-target", PrintKind::DeploymentTarget),
        ("file-names", PrintKind::FileNames),
        ("link-args", PrintKind::LinkArgs),
        ("native-static-libs", PrintKind::NativeStaticLibs),
        ("relocation-models", PrintKind::RelocationModels),
        ("split-debuginfo", PrintKind::SplitDebuginfo),
        ("stack-protector-strategies", PrintKind::StackProtectorStrategies),
        ("sysroot", PrintKind::Sysroot),
        ("target-cpus", PrintKind::TargetCPUs),
        ("target-features", PrintKind::TargetFeatures),
        ("target-libdir", PrintKind::TargetLibdir),
        ("target-list", PrintKind::TargetList),
        ("target-spec-json", PrintKind::TargetSpec),
        ("tls-models", PrintKind::TlsModels),
        // tidy-alphabetical-end
    ];

    // We disallow reusing the same path in multiple prints, such as `--print
    // cfg=output.txt --print link-args=output.txt`, because outputs are printed
    // by disparate pieces of the compiler, and keeping track of which files
    // need to be overwritten vs appended to is annoying.
    let mut printed_paths = FxHashSet::default();

    prints.extend(matches.opt_strs("print").into_iter().map(|req| {
        let (req, out) = split_out_file_name(&req);

        let kind = match PRINT_KINDS.iter().find(|&&(name, _)| name == req) {
            Some((_, PrintKind::TargetSpec)) => {
                if unstable_opts.unstable_options {
                    PrintKind::TargetSpec
                } else {
                    early_dcx.early_fatal(
                        "the `-Z unstable-options` flag must also be passed to \
                         enable the target-spec-json print option",
                    );
                }
            }
            Some((_, PrintKind::AllTargetSpecs)) => {
                if unstable_opts.unstable_options {
                    PrintKind::AllTargetSpecs
                } else {
                    early_dcx.early_fatal(
                        "the `-Z unstable-options` flag must also be passed to \
                         enable the all-target-specs-json print option",
                    );
                }
            }
            Some(&(_, print_kind)) => print_kind,
            None => {
                let prints =
                    PRINT_KINDS.iter().map(|(name, _)| format!("`{name}`")).collect::<Vec<_>>();
                let prints = prints.join(", ");
                early_dcx.early_fatal(format!(
                    "unknown print request `{req}`. Valid print requests are: {prints}"
                ));
            }
        };

        let out = out.unwrap_or(OutFileName::Stdout);
        if let OutFileName::Real(path) = &out {
            if !printed_paths.insert(path.clone()) {
                early_dcx.early_fatal(format!(
                    "cannot print multiple outputs to the same path: {}",
                    path.display(),
                ));
            }
        }

        PrintRequest { kind, out }
    }));

    prints
}

pub fn parse_target_triple(early_dcx: &EarlyDiagCtxt, matches: &getopts::Matches) -> TargetTriple {
    match matches.opt_str("target") {
        Some(target) if target.ends_with(".json") => {
            let path = Path::new(&target);
            TargetTriple::from_path(path).unwrap_or_else(|_| {
                early_dcx.early_fatal(format!("target file {path:?} does not exist"))
            })
        }
        Some(target) => TargetTriple::TargetTriple(target),
        _ => TargetTriple::from_triple(host_triple()),
    }
}

fn parse_opt_level(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
    cg: &CodegenOptions,
) -> OptLevel {
    // The `-O` and `-C opt-level` flags specify the same setting, so we want to be able
    // to use them interchangeably. However, because they're technically different flags,
    // we need to work out manually which should take precedence if both are supplied (i.e.
    // the rightmost flag). We do this by finding the (rightmost) position of both flags and
    // comparing them. Note that if a flag is not found, its position will be `None`, which
    // always compared less than `Some(_)`.
    let max_o = matches.opt_positions("O").into_iter().max();
    let max_c = matches
        .opt_strs_pos("C")
        .into_iter()
        .flat_map(|(i, s)| {
            // NB: This can match a string without `=`.
            if let Some("opt-level") = s.split('=').next() { Some(i) } else { None }
        })
        .max();
    if max_o > max_c {
        OptLevel::Default
    } else {
        match cg.opt_level.as_ref() {
            "0" => OptLevel::No,
            "1" => OptLevel::Less,
            "2" => OptLevel::Default,
            "3" => OptLevel::Aggressive,
            "s" => OptLevel::Size,
            "z" => OptLevel::SizeMin,
            arg => {
                early_dcx.early_fatal(format!(
                    "optimization level needs to be \
                            between 0-3, s or z (instead was `{arg}`)"
                ));
            }
        }
    }
}

fn select_debuginfo(matches: &getopts::Matches, cg: &CodegenOptions) -> DebugInfo {
    let max_g = matches.opt_positions("g").into_iter().max();
    let max_c = matches
        .opt_strs_pos("C")
        .into_iter()
        .flat_map(|(i, s)| {
            // NB: This can match a string without `=`.
            if let Some("debuginfo") = s.split('=').next() { Some(i) } else { None }
        })
        .max();
    if max_g > max_c { DebugInfo::Full } else { cg.debuginfo }
}

fn parse_assert_incr_state(
    early_dcx: &EarlyDiagCtxt,
    opt_assertion: &Option<String>,
) -> Option<IncrementalStateAssertion> {
    match opt_assertion {
        Some(s) if s.as_str() == "loaded" => Some(IncrementalStateAssertion::Loaded),
        Some(s) if s.as_str() == "not-loaded" => Some(IncrementalStateAssertion::NotLoaded),
        Some(s) => {
            early_dcx.early_fatal(format!("unexpected incremental state assertion value: {s}"))
        }
        None => None,
    }
}

fn parse_native_lib_kind(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
    kind: &str,
) -> (NativeLibKind, Option<bool>) {
    let (kind, modifiers) = match kind.split_once(':') {
        None => (kind, None),
        Some((kind, modifiers)) => (kind, Some(modifiers)),
    };

    let kind = match kind {
        "static" => NativeLibKind::Static { bundle: None, whole_archive: None },
        "dylib" => NativeLibKind::Dylib { as_needed: None },
        "framework" => NativeLibKind::Framework { as_needed: None },
        "link-arg" => {
            if !nightly_options::is_unstable_enabled(matches) {
                let why = if nightly_options::match_is_nightly_build(matches) {
                    " and only accepted on the nightly compiler"
                } else {
                    ", the `-Z unstable-options` flag must also be passed to use it"
                };
                early_dcx.early_fatal(format!("library kind `link-arg` is unstable{why}"))
            }
            NativeLibKind::LinkArg
        }
        _ => early_dcx.early_fatal(format!(
            "unknown library kind `{kind}`, expected one of: static, dylib, framework, link-arg"
        )),
    };
    match modifiers {
        None => (kind, None),
        Some(modifiers) => parse_native_lib_modifiers(early_dcx, kind, modifiers, matches),
    }
}

fn parse_native_lib_modifiers(
    early_dcx: &EarlyDiagCtxt,
    mut kind: NativeLibKind,
    modifiers: &str,
    matches: &getopts::Matches,
) -> (NativeLibKind, Option<bool>) {
    let mut verbatim = None;
    for modifier in modifiers.split(',') {
        let (modifier, value) = match modifier.strip_prefix(['+', '-']) {
            Some(m) => (m, modifier.starts_with('+')),
            None => early_dcx.early_fatal(
                "invalid linking modifier syntax, expected '+' or '-' prefix \
                 before one of: bundle, verbatim, whole-archive, as-needed",
            ),
        };

        let report_unstable_modifier = || {
            if !nightly_options::is_unstable_enabled(matches) {
                let why = if nightly_options::match_is_nightly_build(matches) {
                    " and only accepted on the nightly compiler"
                } else {
                    ", the `-Z unstable-options` flag must also be passed to use it"
                };
                early_dcx.early_fatal(format!("linking modifier `{modifier}` is unstable{why}"))
            }
        };
        let assign_modifier = |dst: &mut Option<bool>| {
            if dst.is_some() {
                let msg = format!("multiple `{modifier}` modifiers in a single `-l` option");
                early_dcx.early_fatal(msg)
            } else {
                *dst = Some(value);
            }
        };
        match (modifier, &mut kind) {
            ("bundle", NativeLibKind::Static { bundle, .. }) => assign_modifier(bundle),
            ("bundle", _) => early_dcx.early_fatal(
                "linking modifier `bundle` is only compatible with `static` linking kind",
            ),

            ("verbatim", _) => assign_modifier(&mut verbatim),

            ("whole-archive", NativeLibKind::Static { whole_archive, .. }) => {
                assign_modifier(whole_archive)
            }
            ("whole-archive", _) => early_dcx.early_fatal(
                "linking modifier `whole-archive` is only compatible with `static` linking kind",
            ),

            ("as-needed", NativeLibKind::Dylib { as_needed })
            | ("as-needed", NativeLibKind::Framework { as_needed }) => {
                report_unstable_modifier();
                assign_modifier(as_needed)
            }
            ("as-needed", _) => early_dcx.early_fatal(
                "linking modifier `as-needed` is only compatible with \
                 `dylib` and `framework` linking kinds",
            ),

            // Note: this error also excludes the case with empty modifier
            // string, like `modifiers = ""`.
            _ => early_dcx.early_fatal(format!(
                "unknown linking modifier `{modifier}`, expected one \
                     of: bundle, verbatim, whole-archive, as-needed"
            )),
        }
    }

    (kind, verbatim)
}

fn parse_libs(early_dcx: &EarlyDiagCtxt, matches: &getopts::Matches) -> Vec<NativeLib> {
    matches
        .opt_strs("l")
        .into_iter()
        .map(|s| {
            // Parse string of the form "[KIND[:MODIFIERS]=]lib[:new_name]",
            // where KIND is one of "dylib", "framework", "static", "link-arg" and
            // where MODIFIERS are a comma separated list of supported modifiers
            // (bundle, verbatim, whole-archive, as-needed). Each modifier is prefixed
            // with either + or - to indicate whether it is enabled or disabled.
            // The last value specified for a given modifier wins.
            let (name, kind, verbatim) = match s.split_once('=') {
                None => (s, NativeLibKind::Unspecified, None),
                Some((kind, name)) => {
                    let (kind, verbatim) = parse_native_lib_kind(early_dcx, matches, kind);
                    (name.to_string(), kind, verbatim)
                }
            };

            let (name, new_name) = match name.split_once(':') {
                None => (name, None),
                Some((name, new_name)) => (name.to_string(), Some(new_name.to_owned())),
            };
            if name.is_empty() {
                early_dcx.early_fatal("library name must not be empty");
            }
            NativeLib { name, new_name, kind, verbatim }
        })
        .collect()
}

pub fn parse_externs(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
    unstable_opts: &UnstableOptions,
) -> Externs {
    fn is_ascii_ident(string: &str) -> bool {
        let mut chars = string.chars();
        if let Some(start) = chars.next()
            && (start.is_ascii_alphabetic() || start == '_')
        {
            chars.all(|char| char.is_ascii_alphanumeric() || char == '_')
        } else {
            false
        }
    }

    let is_unstable_enabled = unstable_opts.unstable_options;
    let mut externs: BTreeMap<String, ExternEntry> = BTreeMap::new();
    for arg in matches.opt_strs("extern") {
        let (name, path) = match arg.split_once('=') {
            None => (arg, None),
            Some((name, path)) => (name.to_string(), Some(Path::new(path))),
        };
        let (options, name) = match name.split_once(':') {
            None => (None, name),
            Some((opts, name)) => (Some(opts), name.to_string()),
        };

        if !is_ascii_ident(&name) {
            let mut error = early_dcx.early_struct_fatal(format!(
                "crate name `{name}` passed to `--extern` is not a valid ASCII identifier"
            ));
            let adjusted_name = name.replace('-', "_");
            if is_ascii_ident(&adjusted_name) {
                error.help(format!(
                    "consider replacing the dashes with underscores: `{adjusted_name}`"
                ));
            }
            error.emit();
        }

        let path = path.map(|p| CanonicalizedPath::new(p));

        let entry = externs.entry(name.to_owned());

        use std::collections::btree_map::Entry;

        let entry = if let Some(path) = path {
            // --extern prelude_name=some_file.rlib
            match entry {
                Entry::Vacant(vacant) => {
                    let files = BTreeSet::from_iter(iter::once(path));
                    vacant.insert(ExternEntry::new(ExternLocation::ExactPaths(files)))
                }
                Entry::Occupied(occupied) => {
                    let ext_ent = occupied.into_mut();
                    match ext_ent {
                        ExternEntry { location: ExternLocation::ExactPaths(files), .. } => {
                            files.insert(path);
                        }
                        ExternEntry {
                            location: location @ ExternLocation::FoundInLibrarySearchDirectories,
                            ..
                        } => {
                            // Exact paths take precedence over search directories.
                            let files = BTreeSet::from_iter(iter::once(path));
                            *location = ExternLocation::ExactPaths(files);
                        }
                    }
                    ext_ent
                }
            }
        } else {
            // --extern prelude_name
            match entry {
                Entry::Vacant(vacant) => {
                    vacant.insert(ExternEntry::new(ExternLocation::FoundInLibrarySearchDirectories))
                }
                Entry::Occupied(occupied) => {
                    // Ignore if already specified.
                    occupied.into_mut()
                }
            }
        };

        let mut is_private_dep = false;
        let mut add_prelude = true;
        let mut nounused_dep = false;
        let mut force = false;
        if let Some(opts) = options {
            if !is_unstable_enabled {
                early_dcx.early_fatal(
                    "the `-Z unstable-options` flag must also be passed to \
                     enable `--extern` options",
                );
            }
            for opt in opts.split(',') {
                match opt {
                    "priv" => is_private_dep = true,
                    "noprelude" => {
                        if let ExternLocation::ExactPaths(_) = &entry.location {
                            add_prelude = false;
                        } else {
                            early_dcx.early_fatal(
                                "the `noprelude` --extern option requires a file path",
                            );
                        }
                    }
                    "nounused" => nounused_dep = true,
                    "force" => force = true,
                    _ => early_dcx.early_fatal(format!("unknown --extern option `{opt}`")),
                }
            }
        }

        // Crates start out being not private, and go to being private `priv`
        // is specified.
        entry.is_private_dep |= is_private_dep;
        // likewise `nounused`
        entry.nounused_dep |= nounused_dep;
        // and `force`
        entry.force |= force;
        // If any flag is missing `noprelude`, then add to the prelude.
        entry.add_prelude |= add_prelude;
    }
    Externs(externs)
}

fn parse_remap_path_prefix(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
    unstable_opts: &UnstableOptions,
) -> Vec<(PathBuf, PathBuf)> {
    let mut mapping: Vec<(PathBuf, PathBuf)> = matches
        .opt_strs("remap-path-prefix")
        .into_iter()
        .map(|remap| match remap.rsplit_once('=') {
            None => {
                early_dcx.early_fatal("--remap-path-prefix must contain '=' between FROM and TO")
            }
            Some((from, to)) => (PathBuf::from(from), PathBuf::from(to)),
        })
        .collect();
    match &unstable_opts.remap_cwd_prefix {
        Some(to) => match std::env::current_dir() {
            Ok(cwd) => mapping.push((cwd, to.clone())),
            Err(_) => (),
        },
        None => (),
    };
    mapping
}

fn parse_logical_env(
    early_dcx: &mut EarlyDiagCtxt,
    matches: &getopts::Matches,
) -> FxIndexMap<String, String> {
    let mut vars = FxIndexMap::default();

    for arg in matches.opt_strs("env-set") {
        if let Some((name, val)) = arg.split_once('=') {
            vars.insert(name.to_string(), val.to_string());
        } else {
            early_dcx.early_fatal(format!("`--env-set`: specify value for variable `{arg}`"));
        }
    }

    vars
}

// JUSTIFICATION: before wrapper fn is available
#[allow(rustc::bad_opt_access)]
pub fn build_session_options(early_dcx: &mut EarlyDiagCtxt, matches: &getopts::Matches) -> Options {
    let color = parse_color(early_dcx, matches);

    let edition = parse_crate_edition(early_dcx, matches);

    let JsonConfig {
        json_rendered,
        json_artifact_notifications,
        json_unused_externs,
        json_future_incompat,
    } = parse_json(early_dcx, matches);

    let error_format = parse_error_format(early_dcx, matches, color, json_rendered);

    early_dcx.abort_if_error_and_set_error_format(error_format);

    let diagnostic_width = matches.opt_get("diagnostic-width").unwrap_or_else(|_| {
        early_dcx.early_fatal("`--diagnostic-width` must be an positive integer");
    });

    let unparsed_crate_types = matches.opt_strs("crate-type");
    let crate_types = parse_crate_types_from_list(unparsed_crate_types)
        .unwrap_or_else(|e| early_dcx.early_fatal(e));

    let mut unstable_opts = UnstableOptions::build(early_dcx, matches);
    let (lint_opts, describe_lints, lint_cap) = get_cmd_lint_options(early_dcx, matches);

    check_error_format_stability(early_dcx, &unstable_opts, error_format);

    if !unstable_opts.unstable_options && json_unused_externs.is_enabled() {
        early_dcx.early_fatal(
            "the `-Z unstable-options` flag must also be passed to enable \
            the flag `--json=unused-externs`",
        );
    }

    let output_types = parse_output_types(early_dcx, &unstable_opts, matches);

    let mut cg = CodegenOptions::build(early_dcx, matches);
    let (disable_local_thinlto, mut codegen_units) = should_override_cgus_and_disable_thinlto(
        early_dcx,
        &output_types,
        matches,
        cg.codegen_units,
    );

    if unstable_opts.threads == 0 {
        early_dcx.early_fatal("value for threads must be a positive non-zero integer");
    }

    let fuel = unstable_opts.fuel.is_some() || unstable_opts.print_fuel.is_some();
    if fuel && unstable_opts.threads > 1 {
        early_dcx.early_fatal("optimization fuel is incompatible with multiple threads");
    }
    if fuel && cg.incremental.is_some() {
        early_dcx.early_fatal("optimization fuel is incompatible with incremental compilation");
    }

    let incremental = cg.incremental.as_ref().map(PathBuf::from);

    let assert_incr_state = parse_assert_incr_state(early_dcx, &unstable_opts.assert_incr_state);

    if unstable_opts.profile && incremental.is_some() {
        early_dcx.early_fatal("can't instrument with gcov profiling when compiling incrementally");
    }
    if unstable_opts.profile {
        match codegen_units {
            Some(1) => {}
            None => codegen_units = Some(1),
            Some(_) => early_dcx
                .early_fatal("can't instrument with gcov profiling with multiple codegen units"),
        }
    }

    if cg.profile_generate.enabled() && cg.profile_use.is_some() {
        early_dcx.early_fatal("options `-C profile-generate` and `-C profile-use` are exclusive");
    }

    if unstable_opts.profile_sample_use.is_some()
        && (cg.profile_generate.enabled() || cg.profile_use.is_some())
    {
        early_dcx.early_fatal(
            "option `-Z profile-sample-use` cannot be used with `-C profile-generate` or `-C profile-use`",
        );
    }

    // Check for unstable values of `-C symbol-mangling-version`.
    // This is what prevents them from being used on stable compilers.
    match cg.symbol_mangling_version {
        // Stable values:
        None | Some(SymbolManglingVersion::V0) => {}

        // Unstable values:
        Some(SymbolManglingVersion::Legacy) => {
            if !unstable_opts.unstable_options {
                early_dcx.early_fatal(
                    "`-C symbol-mangling-version=legacy` requires `-Z unstable-options`",
                );
            }
        }
        Some(SymbolManglingVersion::Hashed) => {
            if !unstable_opts.unstable_options {
                early_dcx.early_fatal(
                    "`-C symbol-mangling-version=hashed` requires `-Z unstable-options`",
                );
            }
        }
    }

    // Check for unstable values of `-C instrument-coverage`.
    // This is what prevents them from being used on stable compilers.
    match cg.instrument_coverage {
        // Stable values:
        InstrumentCoverage::All | InstrumentCoverage::Off => {}
        // Unstable values:
        InstrumentCoverage::Branch
        | InstrumentCoverage::ExceptUnusedFunctions
        | InstrumentCoverage::ExceptUnusedGenerics => {
            if !unstable_opts.unstable_options {
                early_dcx.early_fatal(
                    "`-C instrument-coverage=branch` and `-C instrument-coverage=except-*` \
                    require `-Z unstable-options`",
                );
            }
        }
    }

    if cg.instrument_coverage != InstrumentCoverage::Off {
        if cg.profile_generate.enabled() || cg.profile_use.is_some() {
            early_dcx.early_fatal(
                "option `-C instrument-coverage` is not compatible with either `-C profile-use` \
                or `-C profile-generate`",
            );
        }

        // `-C instrument-coverage` implies `-C symbol-mangling-version=v0` - to ensure consistent
        // and reversible name mangling. Note, LLVM coverage tools can analyze coverage over
        // multiple runs, including some changes to source code; so mangled names must be consistent
        // across compilations.
        match cg.symbol_mangling_version {
            None => cg.symbol_mangling_version = Some(SymbolManglingVersion::V0),
            Some(SymbolManglingVersion::Legacy) => {
                early_dcx.early_warn(
                    "-C instrument-coverage requires symbol mangling version `v0`, \
                    but `-C symbol-mangling-version=legacy` was specified",
                );
            }
            Some(SymbolManglingVersion::V0) => {}
            Some(SymbolManglingVersion::Hashed) => {
                early_dcx.early_warn(
                    "-C instrument-coverage requires symbol mangling version `v0`, \
                    but `-C symbol-mangling-version=hashed` was specified",
                );
            }
        }
    }

    if let Ok(graphviz_font) = std::env::var("RUSTC_GRAPHVIZ_FONT") {
        unstable_opts.graphviz_font = graphviz_font;
    }

    if !cg.embed_bitcode {
        match cg.lto {
            LtoCli::No | LtoCli::Unspecified => {}
            LtoCli::Yes | LtoCli::NoParam | LtoCli::Thin | LtoCli::Fat => {
                early_dcx.early_fatal("options `-C embed-bitcode=no` and `-C lto` are incompatible")
            }
        }
    }

    // For testing purposes, until we have more feedback about these options: ensure `-Z
    // unstable-options` is required when using the unstable `-C link-self-contained` and `-C
    // linker-flavor` options.
    if !nightly_options::is_unstable_enabled(matches) {
        let uses_unstable_self_contained_option =
            cg.link_self_contained.are_unstable_variants_set();
        if uses_unstable_self_contained_option {
            early_dcx.early_fatal(
                "only `-C link-self-contained` values `y`/`yes`/`on`/`n`/`no`/`off` are stable, \
                the `-Z unstable-options` flag must also be passed to use the unstable values",
            );
        }

        if let Some(flavor) = cg.linker_flavor {
            if flavor.is_unstable() {
                early_dcx.early_fatal(format!(
                    "the linker flavor `{}` is unstable, the `-Z unstable-options` \
                        flag must also be passed to use the unstable values",
                    flavor.desc()
                ));
            }
        }
    }

    // Check `-C link-self-contained` for consistency: individual components cannot be both enabled
    // and disabled at the same time.
    if let Some(erroneous_components) = cg.link_self_contained.check_consistency() {
        let names: String = erroneous_components
            .into_iter()
            .map(|c| c.as_str().unwrap())
            .intersperse(", ")
            .collect();
        early_dcx.early_fatal(format!(
            "some `-C link-self-contained` components were both enabled and disabled: {names}"
        ));
    }

    let prints = collect_print_requests(early_dcx, &mut cg, &mut unstable_opts, matches);

    let cg = cg;

    let sysroot_opt = matches.opt_str("sysroot").map(|m| PathBuf::from(&m));
    let target_triple = parse_target_triple(early_dcx, matches);
    let opt_level = parse_opt_level(early_dcx, matches, &cg);
    // The `-g` and `-C debuginfo` flags specify the same setting, so we want to be able
    // to use them interchangeably. See the note above (regarding `-O` and `-C opt-level`)
    // for more details.
    let debug_assertions = cg.debug_assertions.unwrap_or(opt_level == OptLevel::No);
    let debuginfo = select_debuginfo(matches, &cg);
    let debuginfo_compression = unstable_opts.debuginfo_compression;

    let mut search_paths = vec![];
    for s in &matches.opt_strs("L") {
        search_paths.push(SearchPath::from_cli_opt(early_dcx, s));
    }

    let libs = parse_libs(early_dcx, matches);

    let test = matches.opt_present("test");

    if !cg.remark.is_empty() && debuginfo == DebugInfo::None {
        early_dcx.early_warn("-C remark requires \"-C debuginfo=n\" to show source locations");
    }

    if cg.remark.is_empty() && unstable_opts.remark_dir.is_some() {
        early_dcx
            .early_warn("using -Z remark-dir without enabling remarks using e.g. -C remark=all");
    }

    let externs = parse_externs(early_dcx, matches, &unstable_opts);

    let crate_name = matches.opt_str("crate-name");

    let remap_path_prefix = parse_remap_path_prefix(early_dcx, matches, &unstable_opts);

    let pretty = parse_pretty(early_dcx, &unstable_opts);

    // query-dep-graph is required if dump-dep-graph is given #106736
    if unstable_opts.dump_dep_graph && !unstable_opts.query_dep_graph {
        early_dcx.early_fatal("can't dump dependency graph without `-Z query-dep-graph`");
    }

    let logical_env = parse_logical_env(early_dcx, matches);

    // Try to find a directory containing the Rust `src`, for more details see
    // the doc comment on the `real_rust_source_base_dir` field.
    let tmp_buf;
    let sysroot = match &sysroot_opt {
        Some(s) => s,
        None => {
            tmp_buf = crate::filesearch::get_or_default_sysroot().expect("Failed finding sysroot");
            &tmp_buf
        }
    };
    let real_rust_source_base_dir = {
        // This is the location used by the `rust-src` `rustup` component.
        let mut candidate = sysroot.join("lib/rustlib/src/rust");
        if let Ok(metadata) = candidate.symlink_metadata() {
            // Replace the symlink rustbuild creates, with its destination.
            // We could try to use `fs::canonicalize` instead, but that might
            // produce unnecessarily verbose path.
            if metadata.file_type().is_symlink() {
                if let Ok(symlink_dest) = std::fs::read_link(&candidate) {
                    candidate = symlink_dest;
                }
            }
        }

        // Only use this directory if it has a file we can expect to always find.
        candidate.join("library/std/src/lib.rs").is_file().then_some(candidate)
    };

    let working_dir = std::env::current_dir().unwrap_or_else(|e| {
        early_dcx.early_fatal(format!("Current directory is invalid: {e}"));
    });

    let remap = file_path_mapping(remap_path_prefix.clone(), &unstable_opts);
    let (path, remapped) = remap.map_prefix(&working_dir);
    let working_dir = if remapped {
        RealFileName::Remapped { virtual_name: path.into_owned(), local_path: Some(working_dir) }
    } else {
        RealFileName::LocalPath(path.into_owned())
    };

    let verbose = matches.opt_present("verbose") || unstable_opts.verbose_internals;

    Options {
        assert_incr_state,
        crate_types,
        optimize: opt_level,
        debuginfo,
        debuginfo_compression,
        lint_opts,
        lint_cap,
        describe_lints,
        output_types,
        search_paths,
        maybe_sysroot: sysroot_opt,
        target_triple,
        test,
        incremental,
        untracked_state_hash: Default::default(),
        unstable_opts,
        prints,
        cg,
        error_format,
        diagnostic_width,
        externs,
        unstable_features: UnstableFeatures::from_environment(crate_name.as_deref()),
        crate_name,
        libs,
        debug_assertions,
        actually_rustdoc: false,
        resolve_doc_links: ResolveDocLinks::ExportedMetadata,
        trimmed_def_paths: false,
        cli_forced_codegen_units: codegen_units,
        cli_forced_local_thinlto_off: disable_local_thinlto,
        remap_path_prefix,
        real_rust_source_base_dir,
        edition,
        json_artifact_notifications,
        json_unused_externs,
        json_future_incompat,
        pretty,
        working_dir,
        color,
        logical_env,
        verbose,
    }
}

fn parse_pretty(early_dcx: &EarlyDiagCtxt, unstable_opts: &UnstableOptions) -> Option<PpMode> {
    use PpMode::*;

    let first = match unstable_opts.unpretty.as_deref()? {
        "normal" => Source(PpSourceMode::Normal),
        "identified" => Source(PpSourceMode::Identified),
        "expanded" => Source(PpSourceMode::Expanded),
        "expanded,identified" => Source(PpSourceMode::ExpandedIdentified),
        "expanded,hygiene" => Source(PpSourceMode::ExpandedHygiene),
        "ast-tree" => AstTree,
        "ast-tree,expanded" => AstTreeExpanded,
        "hir" => Hir(PpHirMode::Normal),
        "hir,identified" => Hir(PpHirMode::Identified),
        "hir,typed" => Hir(PpHirMode::Typed),
        "hir-tree" => HirTree,
        "thir-tree" => ThirTree,
        "thir-flat" => ThirFlat,
        "mir" => Mir,
        "stable-mir" => StableMir,
        "mir-cfg" => MirCFG,
        name => early_dcx.early_fatal(format!(
            "argument to `unpretty` must be one of `normal`, `identified`, \
                            `expanded`, `expanded,identified`, `expanded,hygiene`, \
                            `ast-tree`, `ast-tree,expanded`, `hir`, `hir,identified`, \
                            `hir,typed`, `hir-tree`, `thir-tree`, `thir-flat`, `mir`, `stable-mir`, or \
                            `mir-cfg`; got {name}"
        )),
    };
    debug!("got unpretty option: {first:?}");
    Some(first)
}

pub fn make_crate_type_option() -> RustcOptGroup {
    opt::multi_s(
        "",
        "crate-type",
        "Comma separated list of types of crates
                                for the compiler to emit",
        "[bin|lib|rlib|dylib|cdylib|staticlib|proc-macro]",
    )
}

pub fn parse_crate_types_from_list(list_list: Vec<String>) -> Result<Vec<CrateType>, String> {
    let mut crate_types: Vec<CrateType> = Vec::new();
    for unparsed_crate_type in &list_list {
        for part in unparsed_crate_type.split(',') {
            let new_part = match part {
                "lib" => default_lib_output(),
                "rlib" => CrateType::Rlib,
                "staticlib" => CrateType::Staticlib,
                "dylib" => CrateType::Dylib,
                "cdylib" => CrateType::Cdylib,
                "bin" => CrateType::Executable,
                "proc-macro" => CrateType::ProcMacro,
                _ => return Err(format!("unknown crate type: `{part}`")),
            };
            if !crate_types.contains(&new_part) {
                crate_types.push(new_part)
            }
        }
    }

    Ok(crate_types)
}

pub mod nightly_options {
    use super::{OptionStability, RustcOptGroup};
    use crate::EarlyDiagCtxt;
    use rustc_feature::UnstableFeatures;

    pub fn is_unstable_enabled(matches: &getopts::Matches) -> bool {
        match_is_nightly_build(matches)
            && matches.opt_strs("Z").iter().any(|x| *x == "unstable-options")
    }

    pub fn match_is_nightly_build(matches: &getopts::Matches) -> bool {
        is_nightly_build(matches.opt_str("crate-name").as_deref())
    }

    pub fn is_nightly_build(krate: Option<&str>) -> bool {
        UnstableFeatures::from_environment(krate).is_nightly_build()
    }

    pub fn check_nightly_options(
        early_dcx: &EarlyDiagCtxt,
        matches: &getopts::Matches,
        flags: &[RustcOptGroup],
    ) {
        let has_z_unstable_option = matches.opt_strs("Z").iter().any(|x| *x == "unstable-options");
        let really_allows_unstable_options = match_is_nightly_build(matches);
        let mut nightly_options_on_stable = 0;

        for opt in flags.iter() {
            if opt.stability == OptionStability::Stable {
                continue;
            }
            if !matches.opt_present(opt.name) {
                continue;
            }
            if opt.name != "Z" && !has_z_unstable_option {
                early_dcx.early_fatal(format!(
                    "the `-Z unstable-options` flag must also be passed to enable \
                         the flag `{}`",
                    opt.name
                ));
            }
            if really_allows_unstable_options {
                continue;
            }
            match opt.stability {
                OptionStability::Unstable => {
                    nightly_options_on_stable += 1;
                    let msg = format!(
                        "the option `{}` is only accepted on the nightly compiler",
                        opt.name
                    );
                    let _ = early_dcx.early_err(msg);
                }
                OptionStability::Stable => {}
            }
        }
        if nightly_options_on_stable > 0 {
            early_dcx
                .early_help("consider switching to a nightly toolchain: `rustup default nightly`");
            early_dcx.early_note("selecting a toolchain with `+toolchain` arguments require a rustup proxy; see <https://rust-lang.github.io/rustup/concepts/index.html>");
            early_dcx.early_note("for more information about Rust's stability policy, see <https://doc.rust-lang.org/book/appendix-07-nightly-rust.html#unstable-features>");
            early_dcx.early_fatal(format!(
                "{} nightly option{} were parsed",
                nightly_options_on_stable,
                if nightly_options_on_stable > 1 { "s" } else { "" }
            ));
        }
    }
}

impl fmt::Display for CrateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            CrateType::Executable => "bin".fmt(f),
            CrateType::Dylib => "dylib".fmt(f),
            CrateType::Rlib => "rlib".fmt(f),
            CrateType::Staticlib => "staticlib".fmt(f),
            CrateType::Cdylib => "cdylib".fmt(f),
            CrateType::ProcMacro => "proc-macro".fmt(f),
        }
    }
}

impl IntoDiagnosticArg for CrateType {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue {
        self.to_string().into_diagnostic_arg()
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpSourceMode {
    /// `-Zunpretty=normal`
    Normal,
    /// `-Zunpretty=expanded`
    Expanded,
    /// `-Zunpretty=identified`
    Identified,
    /// `-Zunpretty=expanded,identified`
    ExpandedIdentified,
    /// `-Zunpretty=expanded,hygiene`
    ExpandedHygiene,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpHirMode {
    /// `-Zunpretty=hir`
    Normal,
    /// `-Zunpretty=hir,identified`
    Identified,
    /// `-Zunpretty=hir,typed`
    Typed,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpMode {
    /// Options that print the source code, i.e.
    /// `-Zunpretty=normal` and `-Zunpretty=expanded`
    Source(PpSourceMode),
    /// `-Zunpretty=ast-tree`
    AstTree,
    /// `-Zunpretty=ast-tree,expanded`
    AstTreeExpanded,
    /// Options that print the HIR, i.e. `-Zunpretty=hir`
    Hir(PpHirMode),
    /// `-Zunpretty=hir-tree`
    HirTree,
    /// `-Zunpretty=thir-tree`
    ThirTree,
    /// `-Zunpretty=thir-flat`
    ThirFlat,
    /// `-Zunpretty=mir`
    Mir,
    /// `-Zunpretty=mir-cfg`
    MirCFG,
    /// `-Zunpretty=stable-mir`
    StableMir,
}

impl PpMode {
    pub fn needs_ast_map(&self) -> bool {
        use PpMode::*;
        use PpSourceMode::*;
        match *self {
            Source(Normal | Identified) | AstTree => false,

            Source(Expanded | ExpandedIdentified | ExpandedHygiene)
            | AstTreeExpanded
            | Hir(_)
            | HirTree
            | ThirTree
            | ThirFlat
            | Mir
            | MirCFG
            | StableMir => true,
        }
    }
    pub fn needs_hir(&self) -> bool {
        use PpMode::*;
        match *self {
            Source(_) | AstTree | AstTreeExpanded => false,

            Hir(_) | HirTree | ThirTree | ThirFlat | Mir | MirCFG | StableMir => true,
        }
    }

    pub fn needs_analysis(&self) -> bool {
        use PpMode::*;
        matches!(*self, Hir(PpHirMode::Typed) | Mir | StableMir | MirCFG | ThirTree | ThirFlat)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum WasiExecModel {
    Command,
    Reactor,
}

/// Command-line arguments passed to the compiler have to be incorporated with
/// the dependency tracking system for incremental compilation. This module
/// provides some utilities to make this more convenient.
///
/// The values of all command-line arguments that are relevant for dependency
/// tracking are hashed into a single value that determines whether the
/// incremental compilation cache can be re-used or not. This hashing is done
/// via the `DepTrackingHash` trait defined below, since the standard `Hash`
/// implementation might not be suitable (e.g., arguments are stored in a `Vec`,
/// the hash of which is order dependent, but we might not want the order of
/// arguments to make a difference for the hash).
///
/// However, since the value provided by `Hash::hash` often *is* suitable,
/// especially for primitive types, there is the
/// `impl_dep_tracking_hash_via_hash!()` macro that allows to simply reuse the
/// `Hash` implementation for `DepTrackingHash`. It's important though that
/// we have an opt-in scheme here, so one is hopefully forced to think about
/// how the hash should be calculated when adding a new command-line argument.
pub(crate) mod dep_tracking {
    use super::{
        BranchProtection, CFGuard, CFProtection, CollapseMacroDebuginfo, CrateType, DebugInfo,
        DebugInfoCompression, ErrorOutputType, FunctionReturn, InliningThreshold,
        InstrumentCoverage, InstrumentXRay, LinkerPluginLto, LocationDetail, LtoCli,
        NextSolverConfig, OomStrategy, OptLevel, OutFileName, OutputType, OutputTypes, Polonius,
        RemapPathScopeComponents, ResolveDocLinks, SourceFileHashAlgorithm, SplitDwarfKind,
        SwitchWithOptPath, SymbolManglingVersion, WasiExecModel,
    };
    use crate::lint;
    use crate::utils::NativeLib;
    use rustc_data_structures::fx::FxIndexMap;
    use rustc_data_structures::stable_hasher::Hash64;
    use rustc_errors::LanguageIdentifier;
    use rustc_feature::UnstableFeatures;
    use rustc_span::edition::Edition;
    use rustc_span::RealFileName;
    use rustc_target::spec::{CodeModel, MergeFunctions, PanicStrategy, RelocModel};
    use rustc_target::spec::{
        RelroLevel, SanitizerSet, SplitDebuginfo, StackProtector, TargetTriple, TlsModel,
    };
    use std::collections::BTreeMap;
    use std::hash::{DefaultHasher, Hash};
    use std::num::NonZero;
    use std::path::PathBuf;

    pub trait DepTrackingHash {
        fn hash(
            &self,
            hasher: &mut DefaultHasher,
            error_format: ErrorOutputType,
            for_crate_hash: bool,
        );
    }

    macro_rules! impl_dep_tracking_hash_via_hash {
        ($($t:ty),+ $(,)?) => {$(
            impl DepTrackingHash for $t {
                fn hash(&self, hasher: &mut DefaultHasher, _: ErrorOutputType, _for_crate_hash: bool) {
                    Hash::hash(self, hasher);
                }
            }
        )+};
    }

    impl<T: DepTrackingHash> DepTrackingHash for Option<T> {
        fn hash(
            &self,
            hasher: &mut DefaultHasher,
            error_format: ErrorOutputType,
            for_crate_hash: bool,
        ) {
            match self {
                Some(x) => {
                    Hash::hash(&1, hasher);
                    DepTrackingHash::hash(x, hasher, error_format, for_crate_hash);
                }
                None => Hash::hash(&0, hasher),
            }
        }
    }

    impl_dep_tracking_hash_via_hash!(
        bool,
        usize,
        NonZero<usize>,
        u64,
        Hash64,
        String,
        PathBuf,
        lint::Level,
        WasiExecModel,
        u32,
        RelocModel,
        CodeModel,
        TlsModel,
        InstrumentCoverage,
        InstrumentXRay,
        CrateType,
        MergeFunctions,
        PanicStrategy,
        RelroLevel,
        OptLevel,
        LtoCli,
        DebugInfo,
        DebugInfoCompression,
        CollapseMacroDebuginfo,
        UnstableFeatures,
        NativeLib,
        SanitizerSet,
        CFGuard,
        CFProtection,
        TargetTriple,
        Edition,
        LinkerPluginLto,
        ResolveDocLinks,
        SplitDebuginfo,
        SplitDwarfKind,
        StackProtector,
        SwitchWithOptPath,
        SymbolManglingVersion,
        RemapPathScopeComponents,
        SourceFileHashAlgorithm,
        OutFileName,
        OutputType,
        RealFileName,
        LocationDetail,
        BranchProtection,
        OomStrategy,
        LanguageIdentifier,
        NextSolverConfig,
        Polonius,
        InliningThreshold,
        FunctionReturn,
    );

    impl<T1, T2> DepTrackingHash for (T1, T2)
    where
        T1: DepTrackingHash,
        T2: DepTrackingHash,
    {
        fn hash(
            &self,
            hasher: &mut DefaultHasher,
            error_format: ErrorOutputType,
            for_crate_hash: bool,
        ) {
            Hash::hash(&0, hasher);
            DepTrackingHash::hash(&self.0, hasher, error_format, for_crate_hash);
            Hash::hash(&1, hasher);
            DepTrackingHash::hash(&self.1, hasher, error_format, for_crate_hash);
        }
    }

    impl<T1, T2, T3> DepTrackingHash for (T1, T2, T3)
    where
        T1: DepTrackingHash,
        T2: DepTrackingHash,
        T3: DepTrackingHash,
    {
        fn hash(
            &self,
            hasher: &mut DefaultHasher,
            error_format: ErrorOutputType,
            for_crate_hash: bool,
        ) {
            Hash::hash(&0, hasher);
            DepTrackingHash::hash(&self.0, hasher, error_format, for_crate_hash);
            Hash::hash(&1, hasher);
            DepTrackingHash::hash(&self.1, hasher, error_format, for_crate_hash);
            Hash::hash(&2, hasher);
            DepTrackingHash::hash(&self.2, hasher, error_format, for_crate_hash);
        }
    }

    impl<T: DepTrackingHash> DepTrackingHash for Vec<T> {
        fn hash(
            &self,
            hasher: &mut DefaultHasher,
            error_format: ErrorOutputType,
            for_crate_hash: bool,
        ) {
            Hash::hash(&self.len(), hasher);
            for (index, elem) in self.iter().enumerate() {
                Hash::hash(&index, hasher);
                DepTrackingHash::hash(elem, hasher, error_format, for_crate_hash);
            }
        }
    }

    impl<T: DepTrackingHash, V: DepTrackingHash> DepTrackingHash for FxIndexMap<T, V> {
        fn hash(
            &self,
            hasher: &mut DefaultHasher,
            error_format: ErrorOutputType,
            for_crate_hash: bool,
        ) {
            Hash::hash(&self.len(), hasher);
            for (key, value) in self.iter() {
                DepTrackingHash::hash(key, hasher, error_format, for_crate_hash);
                DepTrackingHash::hash(value, hasher, error_format, for_crate_hash);
            }
        }
    }

    impl DepTrackingHash for OutputTypes {
        fn hash(
            &self,
            hasher: &mut DefaultHasher,
            error_format: ErrorOutputType,
            for_crate_hash: bool,
        ) {
            Hash::hash(&self.0.len(), hasher);
            for (key, val) in &self.0 {
                DepTrackingHash::hash(key, hasher, error_format, for_crate_hash);
                if !for_crate_hash {
                    DepTrackingHash::hash(val, hasher, error_format, for_crate_hash);
                }
            }
        }
    }

    // This is a stable hash because BTreeMap is a sorted container
    pub(crate) fn stable_hash(
        sub_hashes: BTreeMap<&'static str, &dyn DepTrackingHash>,
        hasher: &mut DefaultHasher,
        error_format: ErrorOutputType,
        for_crate_hash: bool,
    ) {
        for (key, sub_hash) in sub_hashes {
            // Using Hash::hash() instead of DepTrackingHash::hash() is fine for
            // the keys, as they are just plain strings
            Hash::hash(&key.len(), hasher);
            Hash::hash(key, hasher);
            sub_hash.hash(hasher, error_format, for_crate_hash);
        }
    }
}

/// Default behavior to use in out-of-memory situations.
#[derive(Clone, Copy, PartialEq, Hash, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum OomStrategy {
    /// Generate a panic that can be caught by `catch_unwind`.
    Panic,

    /// Abort the process immediately.
    Abort,
}

impl OomStrategy {
    pub const SYMBOL: &'static str = "__rust_alloc_error_handler_should_panic";

    pub fn should_panic(self) -> u8 {
        match self {
            OomStrategy::Panic => 1,
            OomStrategy::Abort => 0,
        }
    }
}

/// How to run proc-macro code when building this crate
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum ProcMacroExecutionStrategy {
    /// Run the proc-macro code on the same thread as the server.
    SameThread,

    /// Run the proc-macro code on a different thread.
    CrossThread,
}

/// How to perform collapse macros debug info
/// if-ext - if macro from different crate (related to callsite code)
/// | cmd \ attr    | no  | (unspecified) | external | yes |
/// | no            | no  | no            | no       | no  |
/// | (unspecified) | no  | no            | if-ext   | yes |
/// | external      | no  | if-ext        | if-ext   | yes |
/// | yes           | yes | yes           | yes      | yes |
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum CollapseMacroDebuginfo {
    /// Don't collapse debuginfo for the macro
    No = 0,
    /// Unspecified value
    Unspecified = 1,
    /// Collapse debuginfo if the macro comes from a different crate
    External = 2,
    /// Collapse debuginfo for the macro
    Yes = 3,
}

/// Which format to use for `-Z dump-mono-stats`
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum DumpMonoStatsFormat {
    /// Pretty-print a markdown table
    Markdown,
    /// Emit structured JSON
    Json,
}

impl DumpMonoStatsFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Markdown => "md",
            Self::Json => "json",
        }
    }
}

/// `-Zpolonius` values, enabling the borrow checker polonius analysis, and which version: legacy,
/// or future prototype.
#[derive(Clone, Copy, PartialEq, Hash, Debug, Default)]
pub enum Polonius {
    /// The default value: disabled.
    #[default]
    Off,

    /// Legacy version, using datalog and the `polonius-engine` crate. Historical value for `-Zpolonius`.
    Legacy,

    /// In-tree prototype, extending the NLL infrastructure.
    Next,
}

impl Polonius {
    /// Returns whether the legacy version of polonius is enabled
    pub fn is_legacy_enabled(&self) -> bool {
        matches!(self, Polonius::Legacy)
    }

    /// Returns whether the "next" version of polonius is enabled
    pub fn is_next_enabled(&self) -> bool {
        matches!(self, Polonius::Next)
    }
}

#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum InliningThreshold {
    Always,
    Sometimes(usize),
    Never,
}

impl Default for InliningThreshold {
    fn default() -> Self {
        Self::Sometimes(100)
    }
}

/// The different settings that the `-Zfunction-return` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug, Default)]
pub enum FunctionReturn {
    /// Keep the function return unmodified.
    #[default]
    Keep,

    /// Replace returns with jumps to thunk, without emitting the thunk.
    ThunkExtern,
}
