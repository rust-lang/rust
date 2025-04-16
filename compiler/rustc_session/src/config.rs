//! Contains infrastructure for configuring the compiler, including parsing
//! command-line options.

#![allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable

use std::collections::btree_map::{
    Iter as BTreeMapIter, Keys as BTreeMapKeysIter, Values as BTreeMapValuesIter,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::str::{self, FromStr};
use std::sync::LazyLock;
use std::{cmp, fmt, fs, iter};

use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::stable_hasher::{StableOrd, ToStableHashKey};
use rustc_errors::emitter::HumanReadableErrorType;
use rustc_errors::{ColorConfig, DiagArgValue, DiagCtxtFlags, IntoDiagArg};
use rustc_feature::UnstableFeatures;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::edition::{DEFAULT_EDITION, EDITION_NAME_LIST, Edition, LATEST_STABLE_EDITION};
use rustc_span::source_map::FilePathMapping;
use rustc_span::{
    FileName, FileNameDisplayPreference, RealFileName, SourceFileHashAlgorithm, Symbol, sym,
};
use rustc_target::spec::{
    FramePointer, LinkSelfContainedComponents, LinkerFeatures, SplitDebuginfo, Target, TargetTuple,
};
use tracing::debug;

pub use crate::config::cfg::{Cfg, CheckCfg, ExpectedValues};
use crate::config::native_libs::parse_native_libs;
use crate::errors::FileWriteFail;
pub use crate::options::*;
use crate::search_paths::SearchPath;
use crate::utils::CanonicalizedPath;
use crate::{EarlyDiagCtxt, HashStableContext, Session, filesearch, lint};

mod cfg;
mod native_libs;
pub mod sigpipe;

pub const PRINT_KINDS: &[(&str, PrintKind)] = &[
    // tidy-alphabetical-start
    ("all-target-specs-json", PrintKind::AllTargetSpecsJson),
    ("calling-conventions", PrintKind::CallingConventions),
    ("cfg", PrintKind::Cfg),
    ("check-cfg", PrintKind::CheckCfg),
    ("code-models", PrintKind::CodeModels),
    ("crate-name", PrintKind::CrateName),
    ("crate-root-lint-levels", PrintKind::CrateRootLintLevels),
    ("deployment-target", PrintKind::DeploymentTarget),
    ("file-names", PrintKind::FileNames),
    ("host-tuple", PrintKind::HostTuple),
    ("link-args", PrintKind::LinkArgs),
    ("native-static-libs", PrintKind::NativeStaticLibs),
    ("relocation-models", PrintKind::RelocationModels),
    ("split-debuginfo", PrintKind::SplitDebuginfo),
    ("stack-protector-strategies", PrintKind::StackProtectorStrategies),
    ("supported-crate-types", PrintKind::SupportedCrateTypes),
    ("sysroot", PrintKind::Sysroot),
    ("target-cpus", PrintKind::TargetCPUs),
    ("target-features", PrintKind::TargetFeatures),
    ("target-libdir", PrintKind::TargetLibdir),
    ("target-list", PrintKind::TargetList),
    ("target-spec-json", PrintKind::TargetSpecJson),
    ("tls-models", PrintKind::TlsModels),
    // tidy-alphabetical-end
];

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
    /// `-Copt-level=0`
    No,
    /// `-Copt-level=1`
    Less,
    /// `-Copt-level=2`
    More,
    /// `-Copt-level=3` / `-O`
    Aggressive,
    /// `-Copt-level=s`
    Size,
    /// `-Copt-level=z`
    SizeMin,
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
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum InstrumentCoverage {
    /// `-C instrument-coverage=no` (or `off`, `false` etc.)
    No,
    /// `-C instrument-coverage` or `-C instrument-coverage=yes`
    Yes,
}

/// Individual flag values controlled by `-Zcoverage-options`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct CoverageOptions {
    pub level: CoverageLevel,

    /// `-Zcoverage-options=no-mir-spans`: Don't extract block coverage spans
    /// from MIR statements/terminators, making it easier to inspect/debug
    /// branch and MC/DC coverage mappings.
    ///
    /// For internal debugging only. If other code changes would make it hard
    /// to keep supporting this flag, remove it.
    pub no_mir_spans: bool,

    /// `-Zcoverage-options=discard-all-spans-in-codegen`: During codgen,
    /// discard all coverage spans as though they were invalid. Needed by
    /// regression tests for #133606, because we don't have an easy way to
    /// reproduce it from actual source code.
    pub discard_all_spans_in_codegen: bool,
}

/// Controls whether branch coverage or MC/DC coverage is enabled.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum CoverageLevel {
    /// Instrument for coverage at the MIR block level.
    #[default]
    Block,
    /// Also instrument branch points (includes block coverage).
    Branch,
    /// Same as branch coverage, but also adds branch instrumentation for
    /// certain boolean expressions that are not directly used for branching.
    ///
    /// For example, in the following code, `b` does not directly participate
    /// in a branch, but condition coverage will instrument it as its own
    /// artificial branch:
    /// ```
    /// # let (a, b) = (false, true);
    /// let x = a && b;
    /// //           ^ last operand
    /// ```
    ///
    /// This level is mainly intended to be a stepping-stone towards full MC/DC
    /// instrumentation, so it might be removed in the future when MC/DC is
    /// sufficiently complete, or if it is making MC/DC changes difficult.
    Condition,
    /// Instrument for MC/DC. Mostly a superset of condition coverage, but might
    /// differ in some corner cases.
    Mcdc,
}

/// The different settings that the `-Z autodiff` flag can have.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum AutoDiff {
    /// Enable the autodiff opt pipeline
    Enable,

    /// Print TypeAnalysis information
    PrintTA,
    /// Print ActivityAnalysis Information
    PrintAA,
    /// Print Performance Warnings from Enzyme
    PrintPerf,
    /// Print intermediate IR generation steps
    PrintSteps,
    /// Print the module, before running autodiff.
    PrintModBefore,
    /// Print the module after running autodiff.
    PrintModAfter,
    /// Print the module after running autodiff and optimizations.
    PrintModFinal,

    /// Enzyme's loose type debug helper (can cause incorrect gradients!!)
    /// Usable in cases where Enzyme errors with `can not deduce type of X`.
    LooseTypes,
    /// Runs Enzyme's aggressive inlining
    Inline,
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
    /// `true` shortcuts.
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
    /// `-C link-self-contained=+linker` syntax, or one of the `true` shortcuts.
    pub fn is_linker_enabled(&self) -> bool {
        self.enabled_components.contains(LinkSelfContainedComponents::LINKER)
    }

    /// Returns whether the self-contained linker component was disabled on the CLI, using the
    /// `-C link-self-contained=-linker` syntax, or one of the `false` shortcuts.
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

/// The different values that `-Z linker-features` can take on the CLI: a list of individually
/// enabled or disabled features used during linking.
///
/// There is no need to enable or disable them in bulk. Each feature is fine-grained, and can be
/// used to turn `LinkerFeatures` on or off, without needing to change the linker flavor:
/// - using the system lld, or the self-contained `rust-lld` linker
/// - using a C/C++ compiler to drive the linker (not yet exposed on the CLI)
/// - etc.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct LinkerFeaturesCli {
    /// The linker features that are enabled on the CLI, using the `+feature` syntax.
    pub enabled: LinkerFeatures,

    /// The linker features that are disabled on the CLI, using the `-feature` syntax.
    pub disabled: LinkerFeatures,
}

impl LinkerFeaturesCli {
    /// Accumulates an enabled or disabled feature as specified on the CLI, if possible.
    /// For example: `+lld`, and `-lld`.
    pub(crate) fn handle_cli_feature(&mut self, feature: &str) -> Option<()> {
        // Duplicate flags are reduced as we go, the last occurrence wins:
        // `+feature,-feature,+feature` only enables the feature, and does not record it as both
        // enabled and disabled on the CLI.
        // We also only expose `+/-lld` at the moment, as it's currently the only implemented linker
        // feature and toggling `LinkerFeatures::CC` would be a noop.
        match feature {
            "+lld" => {
                self.enabled.insert(LinkerFeatures::LLD);
                self.disabled.remove(LinkerFeatures::LLD);
                Some(())
            }
            "-lld" => {
                self.disabled.insert(LinkerFeatures::LLD);
                self.enabled.remove(LinkerFeatures::LLD);
                Some(())
            }
            _ => None,
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
    pub(crate) fn all() -> Self {
        Self { file: true, line: true, column: true }
    }
}

/// Values for the `-Z fmt-debug` flag.
#[derive(Copy, Clone, PartialEq, Hash, Debug)]
pub enum FmtDebug {
    /// Derive fully-featured implementation
    Full,
    /// Print only type name, without fields
    Shallow,
    /// `#[derive(Debug)]` and `{:?}` are no-ops
    None,
}

impl FmtDebug {
    pub(crate) fn all() -> [Symbol; 3] {
        [sym::full, sym::none, sym::shallow]
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

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum MirStripDebugInfo {
    None,
    LocalsInTinyFunctions,
    AllLocals,
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
    /// This is the optimized bitcode, which could be either pre-LTO or non-LTO bitcode,
    /// depending on the specific request type.
    Bitcode,
    /// This is the summary or index data part of the ThinLTO bitcode.
    ThinLinkBitcode,
    Assembly,
    LlvmAssembly,
    Mir,
    Metadata,
    Object,
    Exe,
    DepInfo,
}

impl StableOrd for OutputType {
    const CAN_USE_UNSTABLE_SORT: bool = true;

    // Trivial C-Style enums have a stable sort order across compilation sessions.
    const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
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
            | OutputType::ThinLinkBitcode
            | OutputType::Assembly
            | OutputType::LlvmAssembly
            | OutputType::Mir
            | OutputType::Object => false,
        }
    }

    pub fn shorthand(&self) -> &'static str {
        match *self {
            OutputType::Bitcode => "llvm-bc",
            OutputType::ThinLinkBitcode => "thin-link-bitcode",
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
            "thin-link-bitcode" => OutputType::ThinLinkBitcode,
            "obj" => OutputType::Object,
            "metadata" => OutputType::Metadata,
            "link" => OutputType::Exe,
            "dep-info" => OutputType::DepInfo,
            _ => return None,
        })
    }

    fn shorthands_display() -> String {
        format!(
            "`{}`, `{}`, `{}`, `{}`, `{}`, `{}`, `{}`, `{}`, `{}`",
            OutputType::Bitcode.shorthand(),
            OutputType::ThinLinkBitcode.shorthand(),
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
            OutputType::ThinLinkBitcode => "indexing.o",
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
            OutputType::Bitcode
            | OutputType::ThinLinkBitcode
            | OutputType::Object
            | OutputType::Metadata
            | OutputType::Exe => false,
        }
    }
}

/// The type of diagnostics output to generate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ErrorOutputType {
    /// Output meant for the consumption of humans.
    #[default]
    HumanReadable {
        kind: HumanReadableErrorType = HumanReadableErrorType::Default,
        color_config: ColorConfig = ColorConfig::Auto,
    },
    /// Output that's consumed by other tools such as `rustfix` or the `RLS`.
    Json {
        /// Render the JSON in a human readable way (with indents and newlines).
        pretty: bool,
        /// The JSON output includes a `rendered` field that includes the rendered
        /// human output.
        json_rendered: HumanReadableErrorType,
        color_config: ColorConfig,
    },
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

    pub(crate) fn get(&self, key: &OutputType) -> Option<&Option<OutFileName>> {
        self.0.get(key)
    }

    pub fn contains_key(&self, key: &OutputType) -> bool {
        self.0.contains_key(key)
    }

    /// Returns `true` if user specified a name and not just produced type
    pub fn contains_explicit_name(&self, key: &OutputType) -> bool {
        matches!(self.0.get(key), Some(Some(..)))
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
            | OutputType::ThinLinkBitcode
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
            | OutputType::ThinLinkBitcode
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
    // tidy-alphabetical-start
    AllTargetSpecsJson,
    CallingConventions,
    Cfg,
    CheckCfg,
    CodeModels,
    CrateName,
    CrateRootLintLevels,
    DeploymentTarget,
    FileNames,
    HostTuple,
    LinkArgs,
    NativeStaticLibs,
    RelocationModels,
    SplitDebuginfo,
    StackProtectorStrategies,
    SupportedCrateTypes,
    Sysroot,
    TargetCPUs,
    TargetFeatures,
    TargetLibdir,
    TargetList,
    TargetSpecJson,
    TlsModels,
    // tidy-alphabetical-end
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Default)]
pub struct NextSolverConfig {
    /// Whether the new trait solver should be enabled in coherence.
    pub coherence: bool = true,
    /// Whether the new trait solver should be enabled everywhere.
    /// This is only `true` if `coherence` is also enabled.
    pub globally: bool = false,
}

#[derive(Clone)]
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
        if let Input::File(ifile) = self {
            // If for some reason getting the file stem as a UTF-8 string fails,
            // then fallback to a fixed name.
            if let Some(name) = ifile.file_stem().and_then(OsStr::to_str) {
                return name;
            }
        }
        "rust_out"
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
                FileName::CfgSpec(_) => None,
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
        codegen_unit_name: &str,
        invocation_temp: Option<&str>,
    ) -> PathBuf {
        match *self {
            OutFileName::Real(ref path) => path.clone(),
            OutFileName::Stdout => {
                outputs.temp_path_for_cgu(flavor, codegen_unit_name, invocation_temp)
            }
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
    pub(crate) out_directory: PathBuf,
    /// Crate name. Never contains '-'.
    crate_stem: String,
    /// Typically based on `.rs` input file name. Any '-' is preserved.
    filestem: String,
    pub single_output_file: Option<OutFileName>,
    temps_directory: Option<PathBuf>,
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
    fn output_path(&self, flavor: OutputType) -> PathBuf {
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
    pub fn temp_path_for_cgu(
        &self,
        flavor: OutputType,
        codegen_unit_name: &str,
        invocation_temp: Option<&str>,
    ) -> PathBuf {
        let extension = flavor.extension();
        self.temp_path_ext_for_cgu(extension, codegen_unit_name, invocation_temp)
    }

    /// Like `temp_path`, but specifically for dwarf objects.
    pub fn temp_path_dwo_for_cgu(
        &self,
        codegen_unit_name: &str,
        invocation_temp: Option<&str>,
    ) -> PathBuf {
        self.temp_path_ext_for_cgu(DWARF_OBJECT_EXT, codegen_unit_name, invocation_temp)
    }

    /// Like `temp_path`, but also supports things where there is no corresponding
    /// OutputType, like noopt-bitcode or lto-bitcode.
    pub fn temp_path_ext_for_cgu(
        &self,
        ext: &str,
        codegen_unit_name: &str,
        invocation_temp: Option<&str>,
    ) -> PathBuf {
        let mut extension = codegen_unit_name.to_string();

        // Append `.{invocation_temp}` to ensure temporary files are unique.
        if let Some(rng) = invocation_temp {
            extension.push('.');
            extension.push_str(rng);
        }

        // FIXME: This is sketchy that we're not appending `.rcgu` when the ext is empty.
        // Append `.rcgu.{ext}`.
        if !ext.is_empty() {
            extension.push('.');
            extension.push_str(RUST_CGU_EXT);
            extension.push('.');
            extension.push_str(ext);
        }

        let temps_directory = self.temps_directory.as_ref().unwrap_or(&self.out_directory);
        self.with_directory_and_extension(temps_directory, &extension)
    }

    pub fn temp_path_for_diagnostic(&self, ext: &str) -> PathBuf {
        let temps_directory = self.temps_directory.as_ref().unwrap_or(&self.out_directory);
        self.with_directory_and_extension(temps_directory, &ext)
    }

    pub fn with_extension(&self, extension: &str) -> PathBuf {
        self.with_directory_and_extension(&self.out_directory, extension)
    }

    pub fn with_directory_and_extension(&self, directory: &Path, extension: &str) -> PathBuf {
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
        cgu_name: &str,
        invocation_temp: Option<&str>,
    ) -> Option<PathBuf> {
        let obj_out = self.temp_path_for_cgu(OutputType::Object, cgu_name, invocation_temp);
        let dwo_out = self.temp_path_dwo_for_cgu(cgu_name, invocation_temp);
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
        /// Apply remappings to debug information
        const DEBUGINFO = 1 << 3;

        /// An alias for `macro` and `debuginfo`. This ensures all paths in compiled
        /// executables or libraries are remapped but not elsewhere.
        const OBJECT = Self::MACRO.bits() | Self::DEBUGINFO.bits();
    }
}

pub fn host_tuple() -> &'static str {
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
            sysroot: filesearch::materialize_sysroot(None),
            target_triple: TargetTuple::from_tuple(host_tuple()),
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
            target_modifiers: BTreeMap::default(),
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
        !self.unstable_opts.parse_crate_root_only && // The file is just being parsed
            self.unstable_opts.ls.is_empty() // The file is just being queried
    }

    #[inline]
    pub fn share_generics(&self) -> bool {
        match self.unstable_opts.share_generics {
            Some(setting) => setting,
            None => match self.optimize {
                OptLevel::No | OptLevel::Less | OptLevel::Size | OptLevel::SizeMin => true,
                OptLevel::More | OptLevel::Aggressive => false,
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

    pub fn src_hash_algorithm(&self, target: &Target) -> SourceFileHashAlgorithm {
        self.src_hash_algorithm.unwrap_or_else(|| {
            if target.is_like_msvc {
                SourceFileHashAlgorithm::Sha256
            } else {
                SourceFileHashAlgorithm::Md5
            }
        })
    }

    pub fn checksum_hash_algorithm(&self) -> Option<SourceFileHashAlgorithm> {
        self.checksum_hash_algorithm
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
    fn is_empty(&self) -> bool {
        match *self {
            Passes::Some(ref v) => v.is_empty(),
            Passes::All => false,
        }
    }

    pub(crate) fn extend(&mut self, passes: impl IntoIterator<Item = String>) {
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
    pub pc: bool,
    pub key: PAuthKey,
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Default)]
pub struct BranchProtection {
    pub bti: bool,
    pub pac_ret: Option<PacRet>,
}

pub(crate) const fn default_lib_output() -> CrateType {
    CrateType::Rlib
}

pub fn build_configuration(sess: &Session, mut user_cfg: Cfg) -> Cfg {
    // First disallow some configuration given on the command line
    cfg::disallow_cfgs(sess, &user_cfg);

    // Then combine the configuration requested by the session (command line) with
    // some default and generated configuration items.
    user_cfg.extend(cfg::default_configuration(sess));
    user_cfg
}

pub fn build_target_config(
    early_dcx: &EarlyDiagCtxt,
    target: &TargetTuple,
    sysroot: &Path,
) -> Target {
    match Target::search(target, sysroot) {
        Ok((target, warnings)) => {
            for warning in warnings.warning_messages() {
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
        Err(e) => {
            let mut err =
                early_dcx.early_struct_fatal(format!("error loading target specification: {e}"));
            err.help("run `rustc --print target-list` for a list of built-in targets");
            err.emit();
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OptionStability {
    Stable,
    Unstable,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OptionKind {
    /// An option that takes a value, and cannot appear more than once (e.g. `--out-dir`).
    ///
    /// Corresponds to [`getopts::Options::optopt`].
    Opt,

    /// An option that takes a value, and can appear multiple times (e.g. `--emit`).
    ///
    /// Corresponds to [`getopts::Options::optmulti`].
    Multi,

    /// An option that does not take a value, and cannot appear more than once (e.g. `--help`).
    ///
    /// Corresponds to [`getopts::Options::optflag`].
    /// The `hint` string must be empty.
    Flag,

    /// An option that does not take a value, and can appear multiple times (e.g. `-O`).
    ///
    /// Corresponds to [`getopts::Options::optflagmulti`].
    /// The `hint` string must be empty.
    FlagMulti,
}

pub struct RustcOptGroup {
    /// The "primary" name for this option. Normally equal to `long_name`,
    /// except for options that don't have a long name, in which case
    /// `short_name` is used.
    ///
    /// This is needed when interacting with `getopts` in some situations,
    /// because if an option has both forms, that library treats the long name
    /// as primary and the short name as an alias.
    pub name: &'static str,
    stability: OptionStability,
    kind: OptionKind,

    short_name: &'static str,
    long_name: &'static str,
    desc: &'static str,
    value_hint: &'static str,

    /// If true, this option should not be printed by `rustc --help`, but
    /// should still be printed by `rustc --help -v`.
    pub is_verbose_help_only: bool,
}

impl RustcOptGroup {
    pub fn is_stable(&self) -> bool {
        self.stability == OptionStability::Stable
    }

    pub fn apply(&self, options: &mut getopts::Options) {
        let &Self { short_name, long_name, desc, value_hint, .. } = self;
        match self.kind {
            OptionKind::Opt => options.optopt(short_name, long_name, desc, value_hint),
            OptionKind::Multi => options.optmulti(short_name, long_name, desc, value_hint),
            OptionKind::Flag => options.optflag(short_name, long_name, desc),
            OptionKind::FlagMulti => options.optflagmulti(short_name, long_name, desc),
        };
    }
}

pub fn make_opt(
    stability: OptionStability,
    kind: OptionKind,
    short_name: &'static str,
    long_name: &'static str,
    desc: &'static str,
    value_hint: &'static str,
) -> RustcOptGroup {
    // "Flag" options don't have a value, and therefore don't have a value hint.
    match kind {
        OptionKind::Opt | OptionKind::Multi => {}
        OptionKind::Flag | OptionKind::FlagMulti => assert_eq!(value_hint, ""),
    }
    RustcOptGroup {
        name: cmp::max_by_key(short_name, long_name, |s| s.len()),
        stability,
        kind,
        short_name,
        long_name,
        desc,
        value_hint,
        is_verbose_help_only: false,
    }
}

static EDITION_STRING: LazyLock<String> = LazyLock::new(|| {
    format!(
        "Specify which edition of the compiler to use when compiling code. \
The default is {DEFAULT_EDITION} and the latest stable edition is {LATEST_STABLE_EDITION}."
    )
});

static PRINT_HELP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "Compiler information to print on stdout (or to a file)\n\
        INFO may be one of ({}).",
        PRINT_KINDS.iter().map(|(name, _)| format!("{name}")).collect::<Vec<_>>().join("|")
    )
});

/// Returns all rustc command line options, including metadata for
/// each option, such as whether the option is stable.
pub fn rustc_optgroups() -> Vec<RustcOptGroup> {
    use OptionKind::{Flag, FlagMulti, Multi, Opt};
    use OptionStability::{Stable, Unstable};

    use self::make_opt as opt;

    let mut options = vec![
        opt(Stable, Flag, "h", "help", "Display this message", ""),
        opt(
            Stable,
            Multi,
            "",
            "cfg",
            "Configure the compilation environment.\n\
                SPEC supports the syntax `NAME[=\"VALUE\"]`.",
            "SPEC",
        ),
        opt(Stable, Multi, "", "check-cfg", "Provide list of expected cfgs for checking", "SPEC"),
        opt(
            Stable,
            Multi,
            "L",
            "",
            "Add a directory to the library search path. \
                The optional KIND can be one of dependency, crate, native, framework, or all (the default).",
            "[KIND=]PATH",
        ),
        opt(
            Stable,
            Multi,
            "l",
            "",
            "Link the generated crate(s) to the specified native\n\
                library NAME. The optional KIND can be one of\n\
                static, framework, or dylib (the default).\n\
                Optional comma separated MODIFIERS\n\
                (bundle|verbatim|whole-archive|as-needed)\n\
                may be specified each with a prefix of either '+' to\n\
                enable or '-' to disable.",
            "[KIND[:MODIFIERS]=]NAME[:RENAME]",
        ),
        make_crate_type_option(),
        opt(Stable, Opt, "", "crate-name", "Specify the name of the crate being built", "NAME"),
        opt(Stable, Opt, "", "edition", &EDITION_STRING, EDITION_NAME_LIST),
        opt(
            Stable,
            Multi,
            "",
            "emit",
            "Comma separated list of types of output for the compiler to emit",
            "[asm|llvm-bc|llvm-ir|obj|metadata|link|dep-info|mir]",
        ),
        opt(Stable, Multi, "", "print", &PRINT_HELP, "INFO[=FILE]"),
        opt(Stable, FlagMulti, "g", "", "Equivalent to -C debuginfo=2", ""),
        opt(Stable, FlagMulti, "O", "", "Equivalent to -C opt-level=3", ""),
        opt(Stable, Opt, "o", "", "Write output to <filename>", "FILENAME"),
        opt(Stable, Opt, "", "out-dir", "Write output to compiler-chosen filename in <dir>", "DIR"),
        opt(
            Stable,
            Opt,
            "",
            "explain",
            "Provide a detailed explanation of an error message",
            "OPT",
        ),
        opt(Stable, Flag, "", "test", "Build a test harness", ""),
        opt(Stable, Opt, "", "target", "Target triple for which the code is compiled", "TARGET"),
        opt(Stable, Multi, "A", "allow", "Set lint allowed", "LINT"),
        opt(Stable, Multi, "W", "warn", "Set lint warnings", "LINT"),
        opt(Stable, Multi, "", "force-warn", "Set lint force-warn", "LINT"),
        opt(Stable, Multi, "D", "deny", "Set lint denied", "LINT"),
        opt(Stable, Multi, "F", "forbid", "Set lint forbidden", "LINT"),
        opt(
            Stable,
            Multi,
            "",
            "cap-lints",
            "Set the most restrictive lint level. More restrictive lints are capped at this level",
            "LEVEL",
        ),
        opt(Stable, Multi, "C", "codegen", "Set a codegen option", "OPT[=VALUE]"),
        opt(Stable, Flag, "V", "version", "Print version info and exit", ""),
        opt(Stable, Flag, "v", "verbose", "Use verbose output", ""),
    ];

    // Options in this list are hidden from `rustc --help` by default, but are
    // shown by `rustc --help -v`.
    let verbose_only = [
        opt(
            Stable,
            Multi,
            "",
            "extern",
            "Specify where an external rust library is located",
            "NAME[=PATH]",
        ),
        opt(Stable, Opt, "", "sysroot", "Override the system root", "PATH"),
        opt(Unstable, Multi, "Z", "", "Set unstable / perma-unstable options", "FLAG"),
        opt(
            Stable,
            Opt,
            "",
            "error-format",
            "How errors and other messages are produced",
            "human|json|short",
        ),
        opt(Stable, Multi, "", "json", "Configure the JSON output of the compiler", "CONFIG"),
        opt(
            Stable,
            Opt,
            "",
            "color",
            "Configure coloring of output:
                auto   = colorize, if output goes to a tty (default);
                always = always colorize output;
                never  = never colorize output",
            "auto|always|never",
        ),
        opt(
            Stable,
            Opt,
            "",
            "diagnostic-width",
            "Inform rustc of the width of the output so that diagnostics can be truncated to fit",
            "WIDTH",
        ),
        opt(
            Stable,
            Multi,
            "",
            "remap-path-prefix",
            "Remap source names in all output (compiler messages and output files)",
            "FROM=TO",
        ),
        opt(Unstable, Multi, "", "env-set", "Inject an environment variable", "VAR=VALUE"),
    ];
    options.extend(verbose_only.into_iter().map(|mut opt| {
        opt.is_verbose_help_only = true;
        opt
    }));

    options
}

pub fn get_cmd_lint_options(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
) -> (Vec<(String, lint::Level)>, bool, Option<lint::Level>) {
    let mut lint_opts_with_position = vec![];
    let mut describe_lints = false;

    for level in [lint::Allow, lint::Warn, lint::ForceWarn, lint::Deny, lint::Forbid] {
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
    pub json_color: ColorConfig,
    json_artifact_notifications: bool,
    pub json_unused_externs: JsonUnusedExterns,
    json_future_incompat: bool,
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
    let mut json_rendered = HumanReadableErrorType::Default;
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
                "diagnostic-unicode" => {
                    json_rendered = HumanReadableErrorType::Unicode;
                }
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
        json_rendered,
        json_color,
        json_artifact_notifications,
        json_unused_externs,
        json_future_incompat,
    }
}

/// Parses the `--error-format` flag.
pub fn parse_error_format(
    early_dcx: &mut EarlyDiagCtxt,
    matches: &getopts::Matches,
    color_config: ColorConfig,
    json_color: ColorConfig,
    json_rendered: HumanReadableErrorType,
) -> ErrorOutputType {
    // We need the `opts_present` check because the driver will send us Matches
    // with only stable options if no unstable options are used. Since error-format
    // is unstable, it will not be present. We have to use `opts_present` not
    // `opt_present` because the latter will panic.
    let error_format = if matches.opts_present(&["error-format".to_owned()]) {
        match matches.opt_str("error-format").as_deref() {
            None | Some("human") => ErrorOutputType::HumanReadable { color_config, .. },
            Some("human-annotate-rs") => ErrorOutputType::HumanReadable {
                kind: HumanReadableErrorType::AnnotateSnippet,
                color_config,
            },
            Some("json") => {
                ErrorOutputType::Json { pretty: false, json_rendered, color_config: json_color }
            }
            Some("pretty-json") => {
                ErrorOutputType::Json { pretty: true, json_rendered, color_config: json_color }
            }
            Some("short") => {
                ErrorOutputType::HumanReadable { kind: HumanReadableErrorType::Short, color_config }
            }
            Some("human-unicode") => ErrorOutputType::HumanReadable {
                kind: HumanReadableErrorType::Unicode,
                color_config,
            },
            Some(arg) => {
                early_dcx.set_error_format(ErrorOutputType::HumanReadable { color_config, .. });
                early_dcx.early_fatal(format!(
                    "argument for `--error-format` must be `human`, `human-annotate-rs`, \
                    `human-unicode`, `json`, `pretty-json` or `short` (instead was `{arg}`)"
                ))
            }
        }
    } else {
        ErrorOutputType::HumanReadable { color_config, .. }
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
    early_dcx: &EarlyDiagCtxt,
    unstable_opts: &UnstableOptions,
    format: ErrorOutputType,
) {
    if unstable_opts.unstable_options {
        return;
    }
    let format = match format {
        ErrorOutputType::Json { pretty: true, .. } => "pretty-json",
        ErrorOutputType::HumanReadable { kind, .. } => match kind {
            HumanReadableErrorType::AnnotateSnippet => "human-annotate-rs",
            HumanReadableErrorType::Unicode => "human-unicode",
            _ => return,
        },
        _ => return,
    };
    early_dcx.early_fatal(format!("`--error-format={format}` is unstable"))
}

fn parse_output_types(
    early_dcx: &EarlyDiagCtxt,
    unstable_opts: &UnstableOptions,
    matches: &getopts::Matches,
) -> OutputTypes {
    let mut output_types = BTreeMap::new();
    if !unstable_opts.parse_crate_root_only {
        for list in matches.opt_strs("emit") {
            for output_type in list.split(',') {
                let (shorthand, path) = split_out_file_name(output_type);
                let output_type = OutputType::from_shorthand(shorthand).unwrap_or_else(|| {
                    early_dcx.early_fatal(format!(
                        "unknown emission type: `{shorthand}` - expected one of: {display}",
                        display = OutputType::shorthands_display(),
                    ))
                });
                if output_type == OutputType::ThinLinkBitcode && !unstable_opts.unstable_options {
                    early_dcx.early_fatal(format!(
                        "{} requested but -Zunstable-options not specified",
                        OutputType::ThinLinkBitcode.shorthand()
                    ));
                }
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
    unstable_opts: &UnstableOptions,
    matches: &getopts::Matches,
) -> Vec<PrintRequest> {
    let mut prints = Vec::<PrintRequest>::new();
    if cg.target_cpu.as_deref() == Some("help") {
        prints.push(PrintRequest { kind: PrintKind::TargetCPUs, out: OutFileName::Stdout });
        cg.target_cpu = None;
    };
    if cg.target_feature == "help" {
        prints.push(PrintRequest { kind: PrintKind::TargetFeatures, out: OutFileName::Stdout });
        cg.target_feature = String::new();
    }

    // We disallow reusing the same path in multiple prints, such as `--print
    // cfg=output.txt --print link-args=output.txt`, because outputs are printed
    // by disparate pieces of the compiler, and keeping track of which files
    // need to be overwritten vs appended to is annoying.
    let mut printed_paths = FxHashSet::default();

    prints.extend(matches.opt_strs("print").into_iter().map(|req| {
        let (req, out) = split_out_file_name(&req);

        let kind = if let Some((print_name, print_kind)) =
            PRINT_KINDS.iter().find(|&&(name, _)| name == req)
        {
            check_print_request_stability(early_dcx, unstable_opts, (print_name, *print_kind));
            *print_kind
        } else {
            emit_unknown_print_request_help(early_dcx, req)
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

fn check_print_request_stability(
    early_dcx: &EarlyDiagCtxt,
    unstable_opts: &UnstableOptions,
    (print_name, print_kind): (&str, PrintKind),
) {
    match print_kind {
        PrintKind::AllTargetSpecsJson
        | PrintKind::CheckCfg
        | PrintKind::CrateRootLintLevels
        | PrintKind::SupportedCrateTypes
        | PrintKind::TargetSpecJson
            if !unstable_opts.unstable_options =>
        {
            early_dcx.early_fatal(format!(
                "the `-Z unstable-options` flag must also be passed to enable the `{print_name}` \
                print option"
            ));
        }
        _ => {}
    }
}

fn emit_unknown_print_request_help(early_dcx: &EarlyDiagCtxt, req: &str) -> ! {
    let prints = PRINT_KINDS.iter().map(|(name, _)| format!("`{name}`")).collect::<Vec<_>>();
    let prints = prints.join(", ");

    let mut diag = early_dcx.early_struct_fatal(format!("unknown print request: `{req}`"));
    #[allow(rustc::diagnostic_outside_of_impl)]
    diag.help(format!("valid print requests are: {prints}"));

    if req == "lints" {
        diag.help(format!("use `-Whelp` to print a list of lints"));
    }

    diag.help(format!("for more information, see the rustc book: https://doc.rust-lang.org/rustc/command-line-arguments.html#--print-print-compiler-information"));
    diag.emit()
}

pub fn parse_target_triple(early_dcx: &EarlyDiagCtxt, matches: &getopts::Matches) -> TargetTuple {
    match matches.opt_str("target") {
        Some(target) if target.ends_with(".json") => {
            let path = Path::new(&target);
            TargetTuple::from_path(path).unwrap_or_else(|_| {
                early_dcx.early_fatal(format!("target file {path:?} does not exist"))
            })
        }
        Some(target) => TargetTuple::TargetTuple(target),
        _ => TargetTuple::from_tuple(host_tuple()),
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
        OptLevel::Aggressive
    } else {
        match cg.opt_level.as_ref() {
            "0" => OptLevel::No,
            "1" => OptLevel::Less,
            "2" => OptLevel::More,
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
                #[allow(rustc::diagnostic_outside_of_impl)] // FIXME
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
    early_dcx: &EarlyDiagCtxt,
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
        json_color,
        json_artifact_notifications,
        json_unused_externs,
        json_future_incompat,
    } = parse_json(early_dcx, matches);

    let error_format = parse_error_format(early_dcx, matches, color, json_color, json_rendered);

    early_dcx.set_error_format(error_format);

    let diagnostic_width = matches.opt_get("diagnostic-width").unwrap_or_else(|_| {
        early_dcx.early_fatal("`--diagnostic-width` must be an positive integer");
    });

    let unparsed_crate_types = matches.opt_strs("crate-type");
    let crate_types = parse_crate_types_from_list(unparsed_crate_types)
        .unwrap_or_else(|e| early_dcx.early_fatal(e));

    let mut target_modifiers = BTreeMap::<OptionsTargetModifiers, String>::new();

    let mut unstable_opts = UnstableOptions::build(early_dcx, matches, &mut target_modifiers);
    let (lint_opts, describe_lints, lint_cap) = get_cmd_lint_options(early_dcx, matches);

    check_error_format_stability(early_dcx, &unstable_opts, error_format);

    let output_types = parse_output_types(early_dcx, &unstable_opts, matches);

    let mut cg = CodegenOptions::build(early_dcx, matches, &mut target_modifiers);
    let (disable_local_thinlto, codegen_units) = should_override_cgus_and_disable_thinlto(
        early_dcx,
        &output_types,
        matches,
        cg.codegen_units,
    );

    if unstable_opts.threads == 0 {
        early_dcx.early_fatal("value for threads must be a positive non-zero integer");
    }

    if unstable_opts.threads == parse::MAX_THREADS_CAP {
        early_dcx.early_warn(format!("number of threads was capped at {}", parse::MAX_THREADS_CAP));
    }

    let incremental = cg.incremental.as_ref().map(PathBuf::from);

    let assert_incr_state = parse_assert_incr_state(early_dcx, &unstable_opts.assert_incr_state);

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

    if cg.instrument_coverage != InstrumentCoverage::No {
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
        // FIXME: this is only mutation of UnstableOptions here, move into
        // UnstableOptions::build?
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

    if !nightly_options::is_unstable_enabled(matches)
        && cg.force_frame_pointers == FramePointer::NonLeaf
    {
        early_dcx.early_fatal(
            "`-Cforce-frame-pointers=non-leaf` or `always` also requires `-Zunstable-options` \
                and a nightly compiler",
        )
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

    let prints = collect_print_requests(early_dcx, &mut cg, &unstable_opts, matches);

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

    let crate_name = matches.opt_str("crate-name");
    let unstable_features = UnstableFeatures::from_environment(crate_name.as_deref());
    // Parse any `-l` flags, which link to native libraries.
    let libs = parse_native_libs(early_dcx, &unstable_opts, unstable_features, matches);

    let test = matches.opt_present("test");

    if !cg.remark.is_empty() && debuginfo == DebugInfo::None {
        early_dcx.early_warn("-C remark requires \"-C debuginfo=n\" to show source locations");
    }

    if cg.remark.is_empty() && unstable_opts.remark_dir.is_some() {
        early_dcx
            .early_warn("using -Z remark-dir without enabling remarks using e.g. -C remark=all");
    }

    let externs = parse_externs(early_dcx, matches, &unstable_opts);

    let remap_path_prefix = parse_remap_path_prefix(early_dcx, matches, &unstable_opts);

    let pretty = parse_pretty(early_dcx, &unstable_opts);

    // query-dep-graph is required if dump-dep-graph is given #106736
    if unstable_opts.dump_dep_graph && !unstable_opts.query_dep_graph {
        early_dcx.early_fatal("can't dump dependency graph without `-Z query-dep-graph`");
    }

    let logical_env = parse_logical_env(early_dcx, matches);

    let sysroot = filesearch::materialize_sysroot(sysroot_opt);

    let real_rust_source_base_dir = {
        // This is the location used by the `rust-src` `rustup` component.
        let mut candidate = sysroot.join("lib/rustlib/src/rust");
        if let Ok(metadata) = candidate.symlink_metadata() {
            // Replace the symlink bootstrap creates, with its destination.
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

    let mut search_paths = vec![];
    for s in &matches.opt_strs("L") {
        search_paths.push(SearchPath::from_cli_opt(
            &sysroot,
            &target_triple,
            early_dcx,
            s,
            unstable_opts.unstable_options,
        ));
    }

    let working_dir = std::env::current_dir().unwrap_or_else(|e| {
        early_dcx.early_fatal(format!("Current directory is invalid: {e}"));
    });

    let file_mapping = file_path_mapping(remap_path_prefix.clone(), &unstable_opts);
    let working_dir = file_mapping.to_real_filename(&working_dir);

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
        sysroot,
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
        unstable_features,
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
        target_modifiers,
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
    make_opt(
        OptionStability::Stable,
        OptionKind::Multi,
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
                _ => {
                    return Err(format!(
                        "unknown crate type: `{part}`, expected one of: \
                        `lib`, `rlib`, `staticlib`, `dylib`, `cdylib`, `bin`, `proc-macro`",
                    ));
                }
            };
            if !crate_types.contains(&new_part) {
                crate_types.push(new_part)
            }
        }
    }

    Ok(crate_types)
}

pub mod nightly_options {
    use rustc_feature::UnstableFeatures;

    use super::{OptionStability, RustcOptGroup};
    use crate::EarlyDiagCtxt;

    pub fn is_unstable_enabled(matches: &getopts::Matches) -> bool {
        match_is_nightly_build(matches)
            && matches.opt_strs("Z").iter().any(|x| *x == "unstable-options")
    }

    pub fn match_is_nightly_build(matches: &getopts::Matches) -> bool {
        is_nightly_build(matches.opt_str("crate-name").as_deref())
    }

    fn is_nightly_build(krate: Option<&str>) -> bool {
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
                    // The non-zero nightly_options_on_stable will force an early_fatal eventually.
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

impl IntoDiagArg for CrateType {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_string().into_diag_arg(&mut None)
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
/// Pretty print mode
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
    use std::collections::BTreeMap;
    use std::hash::Hash;
    use std::num::NonZero;
    use std::path::PathBuf;

    use rustc_abi::Align;
    use rustc_data_structures::fx::FxIndexMap;
    use rustc_data_structures::stable_hasher::StableHasher;
    use rustc_errors::LanguageIdentifier;
    use rustc_feature::UnstableFeatures;
    use rustc_hashes::Hash64;
    use rustc_span::RealFileName;
    use rustc_span::edition::Edition;
    use rustc_target::spec::{
        CodeModel, FramePointer, MergeFunctions, OnBrokenPipe, PanicStrategy, RelocModel,
        RelroLevel, SanitizerSet, SplitDebuginfo, StackProtector, SymbolVisibility, TargetTuple,
        TlsModel, WasmCAbi,
    };

    use super::{
        AutoDiff, BranchProtection, CFGuard, CFProtection, CollapseMacroDebuginfo, CoverageOptions,
        CrateType, DebugInfo, DebugInfoCompression, ErrorOutputType, FmtDebug, FunctionReturn,
        InliningThreshold, InstrumentCoverage, InstrumentXRay, LinkerPluginLto, LocationDetail,
        LtoCli, MirStripDebugInfo, NextSolverConfig, OomStrategy, OptLevel, OutFileName,
        OutputType, OutputTypes, PatchableFunctionEntry, Polonius, RemapPathScopeComponents,
        ResolveDocLinks, SourceFileHashAlgorithm, SplitDwarfKind, SwitchWithOptPath,
        SymbolManglingVersion, WasiExecModel,
    };
    use crate::lint;
    use crate::utils::NativeLib;

    pub(crate) trait DepTrackingHash {
        fn hash(
            &self,
            hasher: &mut StableHasher,
            error_format: ErrorOutputType,
            for_crate_hash: bool,
        );
    }

    macro_rules! impl_dep_tracking_hash_via_hash {
        ($($t:ty),+ $(,)?) => {$(
            impl DepTrackingHash for $t {
                fn hash(&self, hasher: &mut StableHasher, _: ErrorOutputType, _for_crate_hash: bool) {
                    Hash::hash(self, hasher);
                }
            }
        )+};
    }

    impl<T: DepTrackingHash> DepTrackingHash for Option<T> {
        fn hash(
            &self,
            hasher: &mut StableHasher,
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
        AutoDiff,
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
        FramePointer,
        RelocModel,
        CodeModel,
        TlsModel,
        InstrumentCoverage,
        CoverageOptions,
        InstrumentXRay,
        CrateType,
        MergeFunctions,
        OnBrokenPipe,
        PanicStrategy,
        RelroLevel,
        OptLevel,
        LtoCli,
        DebugInfo,
        DebugInfoCompression,
        MirStripDebugInfo,
        CollapseMacroDebuginfo,
        UnstableFeatures,
        NativeLib,
        SanitizerSet,
        CFGuard,
        CFProtection,
        TargetTuple,
        Edition,
        LinkerPluginLto,
        ResolveDocLinks,
        SplitDebuginfo,
        SplitDwarfKind,
        StackProtector,
        SwitchWithOptPath,
        SymbolManglingVersion,
        SymbolVisibility,
        RemapPathScopeComponents,
        SourceFileHashAlgorithm,
        OutFileName,
        OutputType,
        RealFileName,
        LocationDetail,
        FmtDebug,
        BranchProtection,
        OomStrategy,
        LanguageIdentifier,
        NextSolverConfig,
        PatchableFunctionEntry,
        Polonius,
        InliningThreshold,
        FunctionReturn,
        WasmCAbi,
        Align,
    );

    impl<T1, T2> DepTrackingHash for (T1, T2)
    where
        T1: DepTrackingHash,
        T2: DepTrackingHash,
    {
        fn hash(
            &self,
            hasher: &mut StableHasher,
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
            hasher: &mut StableHasher,
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
            hasher: &mut StableHasher,
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
            hasher: &mut StableHasher,
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
            hasher: &mut StableHasher,
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
        hasher: &mut StableHasher,
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

/// `-Z patchable-function-entry` representation - how many nops to put before and after function
/// entry.
#[derive(Clone, Copy, PartialEq, Hash, Debug, Default)]
pub struct PatchableFunctionEntry {
    /// Nops before the entry
    prefix: u8,
    /// Nops after the entry
    entry: u8,
}

impl PatchableFunctionEntry {
    pub fn from_total_and_prefix_nops(
        total_nops: u8,
        prefix_nops: u8,
    ) -> Option<PatchableFunctionEntry> {
        if total_nops < prefix_nops {
            None
        } else {
            Some(Self { prefix: prefix_nops, entry: total_nops - prefix_nops })
        }
    }
    pub fn prefix(&self) -> u8 {
        self.prefix
    }
    pub fn entry(&self) -> u8 {
        self.entry
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

/// Whether extra span comments are included when dumping MIR, via the `-Z mir-include-spans` flag.
/// By default, only enabled in the NLL MIR dumps, and disabled in all other passes.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub enum MirIncludeSpans {
    Off,
    On,
    /// Default: include extra comments in NLL MIR dumps only. Can be ignored and considered as
    /// `Off` in all other cases.
    #[default]
    Nll,
}

impl MirIncludeSpans {
    /// Unless opting into extra comments for all passes, they can be considered disabled.
    /// The cases where a distinction between on/off and a per-pass value can exist will be handled
    /// in the passes themselves: i.e. the `Nll` value is considered off for all intents and
    /// purposes, except for the NLL MIR dump pass.
    pub fn is_enabled(self) -> bool {
        self == MirIncludeSpans::On
    }
}
