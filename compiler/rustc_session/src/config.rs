//! Contains infrastructure for configuring the compiler, including parsing
//! command-line options.

pub use crate::options::*;

use crate::lint;
use crate::search_paths::SearchPath;
use crate::utils::NativeLibKind;
use crate::{early_error, early_warn, Session};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::impl_stable_hash_via_hash;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

use rustc_target::abi::{Align, TargetDataLayout};
use rustc_target::spec::{Target, TargetTriple};

use crate::parse::CrateConfig;
use rustc_feature::UnstableFeatures;
use rustc_span::edition::{Edition, DEFAULT_EDITION, EDITION_NAME_LIST};
use rustc_span::source_map::{FileName, FilePathMapping};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::SourceFileHashAlgorithm;

use rustc_errors::emitter::HumanReadableErrorType;
use rustc_errors::{ColorConfig, HandlerFlags};

use std::collections::btree_map::{
    Iter as BTreeMapIter, Keys as BTreeMapKeysIter, Values as BTreeMapValuesIter,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::iter::{self, FromIterator};
use std::path::{Path, PathBuf};
use std::str::{self, FromStr};

bitflags! {
    #[derive(Default, Encodable, Decodable)]
    pub struct SanitizerSet: u8 {
        const ADDRESS = 1 << 0;
        const LEAK    = 1 << 1;
        const MEMORY  = 1 << 2;
        const THREAD  = 1 << 3;
    }
}

/// Formats a sanitizer set as a comma separated list of sanitizers' names.
impl fmt::Display for SanitizerSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for s in *self {
            let name = match s {
                SanitizerSet::ADDRESS => "address",
                SanitizerSet::LEAK => "leak",
                SanitizerSet::MEMORY => "memory",
                SanitizerSet::THREAD => "thread",
                _ => panic!("unrecognized sanitizer {:?}", s),
            };
            if !first {
                f.write_str(",")?;
            }
            f.write_str(name)?;
            first = false;
        }
        Ok(())
    }
}

impl IntoIterator for SanitizerSet {
    type Item = SanitizerSet;
    type IntoIter = std::vec::IntoIter<SanitizerSet>;

    fn into_iter(self) -> Self::IntoIter {
        [SanitizerSet::ADDRESS, SanitizerSet::LEAK, SanitizerSet::MEMORY, SanitizerSet::THREAD]
            .iter()
            .copied()
            .filter(|&s| self.contains(s))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<CTX> HashStable<CTX> for SanitizerSet {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.bits().hash_stable(ctx, hasher);
    }
}

/// The different settings that the `-Z strip` flag can have.
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

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum OptLevel {
    No,         // -O0
    Less,       // -O1
    Default,    // -O2
    Aggressive, // -O3
    Size,       // -Os
    SizeMin,    // -Oz
}

impl_stable_hash_via_hash!(OptLevel);

/// This is what the `LtoCli` values get mapped to after resolving defaults and
/// and taking other command line options into account.
#[derive(Clone, PartialEq)]
pub enum Lto {
    /// Don't do any LTO whatsoever
    No,

    /// Do a full crate graph LTO with ThinLTO
    Thin,

    /// Do a local graph LTO with ThinLTO (only relevant for multiple codegen
    /// units).
    ThinLocal,

    /// Do a full crate graph LTO with "fat" LTO
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

/// The different settings that the `-Z dump_mir_spanview` flag can have. `Statement` generates a
/// document highlighting each span of every statement (including terminators). `Terminator` and
/// `Block` highlight a single span per `BasicBlock`: the span of the block's `Terminator`, or a
/// computed span for the block, representing the entire range, covering the block's terminator and
/// all of its statements.
#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum MirSpanview {
    /// Default `-Z dump_mir_spanview` or `-Z dump_mir_spanview=statement`
    Statement,
    /// `-Z dump_mir_spanview=terminator`
    Terminator,
    /// `-Z dump_mir_spanview=block`
    Block,
}

#[derive(Clone, PartialEq, Hash)]
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

#[derive(Clone, PartialEq, Hash)]
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Encodable, Decodable)]
pub enum SymbolManglingVersion {
    Legacy,
    V0,
}

impl_stable_hash_via_hash!(SymbolManglingVersion);

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum DebugInfo {
    None,
    Limited,
    Full,
}

/// Some debuginfo requires link-time relocation and some does not. LLVM can partition the debuginfo
/// into sections depending on whether or not it requires link-time relocation. Split DWARF
/// provides a mechanism which allows the linker to skip the sections which don't require link-time
/// relocation - either by putting those sections into DWARF object files, or keeping them in the
/// object file in such a way that the linker will skip them.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum SplitDwarfKind {
    /// Disabled.
    None,
    /// Sections which do not require relocation are written into the object file but ignored
    /// by the linker.
    Single,
    /// Sections which do not require relocation are written into a DWARF object (`.dwo`) file,
    /// which is skipped by the linker by virtue of being a different file.
    Split,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
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

impl_stable_hash_via_hash!(OutputType);

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

    fn shorthand(&self) -> &'static str {
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

/// Parameter to control path trimming.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrimmedDefPaths {
    /// `try_print_trimmed_def_path` never prints a trimmed path and never calls the expensive query
    Never,
    /// `try_print_trimmed_def_path` calls the expensive query, the query doesn't call `delay_good_path_bug`
    Always,
    /// `try_print_trimmed_def_path` calls the expensive query, the query calls `delay_good_path_bug`
    GoodPath,
}

impl Default for TrimmedDefPaths {
    fn default() -> Self {
        Self::Never
    }
}

/// Use tree-based collections to cheaply get a deterministic `Hash` implementation.
/// *Do not* switch `BTreeMap` out for an unsorted container type! That would break
/// dependency tracking for command-line arguments.
#[derive(Clone, Hash)]
pub struct OutputTypes(BTreeMap<OutputType, Option<PathBuf>>);

impl_stable_hash_via_hash!(OutputTypes);

impl OutputTypes {
    pub fn new(entries: &[(OutputType, Option<PathBuf>)]) -> OutputTypes {
        OutputTypes(BTreeMap::from_iter(entries.iter().map(|&(k, ref v)| (k, v.clone()))))
    }

    pub fn get(&self, key: &OutputType) -> Option<&Option<PathBuf>> {
        self.0.get(key)
    }

    pub fn contains_key(&self, key: &OutputType) -> bool {
        self.0.contains_key(key)
    }

    pub fn keys(&self) -> BTreeMapKeysIter<'_, OutputType, Option<PathBuf>> {
        self.0.keys()
    }

    pub fn values(&self) -> BTreeMapValuesIter<'_, OutputType, Option<PathBuf>> {
        self.0.values()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    // Returns `true` if any of the output types require codegen or linking.
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
    ExactPaths(BTreeSet<String>),
}

impl Externs {
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
        ExternEntry { location, is_private_dep: false, add_prelude: false }
    }

    pub fn files(&self) -> Option<impl Iterator<Item = &String>> {
        match &self.location {
            ExternLocation::ExactPaths(set) => Some(set.iter()),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PrintRequest {
    FileNames,
    Sysroot,
    TargetLibdir,
    CrateName,
    Cfg,
    TargetList,
    TargetCPUs,
    TargetFeatures,
    RelocationModels,
    CodeModels,
    TlsModels,
    TargetSpec,
    NativeStaticLibs,
}

#[derive(Copy, Clone)]
pub enum BorrowckMode {
    Mir,
    Migrate,
}

impl BorrowckMode {
    /// Returns whether we should run the MIR-based borrow check, but also fall back
    /// on the AST borrow check if the MIR-based one errors.
    pub fn migrate(self) -> bool {
        match self {
            BorrowckMode::Mir => false,
            BorrowckMode::Migrate => true,
        }
    }
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

    pub fn get_input(&mut self) -> Option<&mut String> {
        match *self {
            Input::File(_) => None,
            Input::Str { ref mut input, .. } => Some(input),
        }
    }

    pub fn source_name(&self) -> FileName {
        match *self {
            Input::File(ref ifile) => ifile.clone().into(),
            Input::Str { ref name, .. } => name.clone(),
        }
    }
}

#[derive(Clone, Hash)]
pub struct OutputFilenames {
    pub out_directory: PathBuf,
    filestem: String,
    pub single_output_file: Option<PathBuf>,
    pub outputs: OutputTypes,
}

impl_stable_hash_via_hash!(OutputFilenames);

pub const RLINK_EXT: &str = "rlink";
pub const RUST_CGU_EXT: &str = "rcgu";
pub const DWARF_OBJECT_EXT: &str = "dwo";

impl OutputFilenames {
    pub fn new(
        out_directory: PathBuf,
        out_filestem: String,
        single_output_file: Option<PathBuf>,
        extra: String,
        outputs: OutputTypes,
    ) -> Self {
        OutputFilenames {
            out_directory,
            single_output_file,
            outputs,
            filestem: format!("{}{}", out_filestem, extra),
        }
    }

    pub fn path(&self, flavor: OutputType) -> PathBuf {
        self.outputs
            .get(&flavor)
            .and_then(|p| p.to_owned())
            .or_else(|| self.single_output_file.clone())
            .unwrap_or_else(|| self.temp_path(flavor, None))
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

        self.with_extension(&extension)
    }

    pub fn with_extension(&self, extension: &str) -> PathBuf {
        let mut path = self.out_directory.join(&self.filestem);
        path.set_extension(extension);
        path
    }

    /// Returns the name of the Split DWARF file - this can differ depending on which Split DWARF
    /// mode is being used, which is the logic that this function is intended to encapsulate.
    pub fn split_dwarf_filename(
        &self,
        split_dwarf_kind: SplitDwarfKind,
        cgu_name: Option<&str>,
    ) -> Option<PathBuf> {
        self.split_dwarf_path(split_dwarf_kind, cgu_name)
            .map(|path| path.strip_prefix(&self.out_directory).unwrap_or(&path).to_path_buf())
    }

    /// Returns the path for the Split DWARF file - this can differ depending on which Split DWARF
    /// mode is being used, which is the logic that this function is intended to encapsulate.
    pub fn split_dwarf_path(
        &self,
        split_dwarf_kind: SplitDwarfKind,
        cgu_name: Option<&str>,
    ) -> Option<PathBuf> {
        let obj_out = self.temp_path(OutputType::Object, cgu_name);
        let dwo_out = self.temp_path_dwo(cgu_name);
        match split_dwarf_kind {
            SplitDwarfKind::None => None,
            // Single mode doesn't change how DWARF is emitted, but does add Split DWARF attributes
            // (pointing at the path which is being determined here). Use the path to the current
            // object file.
            SplitDwarfKind::Single => Some(obj_out),
            // Split mode emits the DWARF into a different file, use that path.
            SplitDwarfKind::Split => Some(dwo_out),
        }
    }
}

pub fn host_triple() -> &'static str {
    // Get the host triple out of the build environment. This ensures that our
    // idea of the host triple is the same as for the set of libraries we've
    // actually built.  We can't just take LLVM's host triple because they
    // normalize all ix86 architectures to i386.
    //
    // Instead of grabbing the host triple (for the current host), we grab (at
    // compile time) the target triple that this rustc is built with and
    // calling that (at runtime) the host triple.
    (option_env!("CFG_COMPILER_HOST_TRIPLE")).expect("CFG_COMPILER_HOST_TRIPLE")
}

impl Default for Options {
    fn default() -> Options {
        Options {
            crate_types: Vec::new(),
            optimize: OptLevel::No,
            debuginfo: DebugInfo::None,
            lint_opts: Vec::new(),
            lint_cap: None,
            describe_lints: false,
            output_types: OutputTypes(BTreeMap::new()),
            search_paths: vec![],
            maybe_sysroot: None,
            target_triple: TargetTriple::from_triple(host_triple()),
            test: false,
            incremental: None,
            debugging_opts: basic_debugging_options(),
            prints: Vec::new(),
            borrowck_mode: BorrowckMode::Migrate,
            cg: basic_codegen_options(),
            error_format: ErrorOutputType::default(),
            externs: Externs(BTreeMap::new()),
            crate_name: None,
            alt_std_name: None,
            libs: Vec::new(),
            unstable_features: UnstableFeatures::Disallow,
            debug_assertions: true,
            actually_rustdoc: false,
            trimmed_def_paths: TrimmedDefPaths::default(),
            cli_forced_codegen_units: None,
            cli_forced_thinlto_off: false,
            remap_path_prefix: Vec::new(),
            edition: DEFAULT_EDITION,
            json_artifact_notifications: false,
            pretty: None,
        }
    }
}

impl Options {
    /// Returns `true` if there is a reason to build the dep graph.
    pub fn build_dep_graph(&self) -> bool {
        self.incremental.is_some()
            || self.debugging_opts.dump_dep_graph
            || self.debugging_opts.query_dep_graph
    }

    #[inline(always)]
    pub fn enable_dep_node_debug_strs(&self) -> bool {
        cfg!(debug_assertions)
            && (self.debugging_opts.query_dep_graph || self.debugging_opts.incremental_info)
    }

    pub fn file_path_mapping(&self) -> FilePathMapping {
        FilePathMapping::new(self.remap_path_prefix.clone())
    }

    /// Returns `true` if there will be an output file generated.
    pub fn will_create_output_file(&self) -> bool {
        !self.debugging_opts.parse_only && // The file is just being parsed
            !self.debugging_opts.ls // The file is just being queried
    }

    #[inline]
    pub fn share_generics(&self) -> bool {
        match self.debugging_opts.share_generics {
            Some(setting) => setting,
            None => match self.optimize {
                OptLevel::No | OptLevel::Less | OptLevel::Size | OptLevel::SizeMin => true,
                OptLevel::Default | OptLevel::Aggressive => false,
            },
        }
    }
}

impl DebuggingOptions {
    pub fn diagnostic_handler_flags(&self, can_emit_warnings: bool) -> HandlerFlags {
        HandlerFlags {
            can_emit_warnings,
            treat_err_as_bug: self.treat_err_as_bug,
            dont_buffer_diagnostics: self.dont_buffer_diagnostics,
            report_delayed_bugs: self.report_delayed_bugs,
            macro_backtrace: self.macro_backtrace,
            deduplicate_diagnostics: self.deduplicate_diagnostics,
        }
    }

    pub fn get_symbol_mangling_version(&self) -> SymbolManglingVersion {
        self.symbol_mangling_version.unwrap_or(SymbolManglingVersion::Legacy)
    }
}

// The type of entry function, so users can have their own entry functions
#[derive(Copy, Clone, PartialEq, Hash, Debug)]
pub enum EntryFnType {
    Main,
    Start,
}

impl_stable_hash_via_hash!(EntryFnType);

#[derive(Copy, PartialEq, PartialOrd, Clone, Ord, Eq, Hash, Debug, Encodable, Decodable)]
pub enum CrateType {
    Executable,
    Dylib,
    Rlib,
    Staticlib,
    Cdylib,
    ProcMacro,
}

impl_stable_hash_via_hash!(CrateType);

#[derive(Clone, Hash)]
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
}

pub const fn default_lib_output() -> CrateType {
    CrateType::Rlib
}

pub fn default_configuration(sess: &Session) -> CrateConfig {
    let end = &sess.target.endian;
    let arch = &sess.target.arch;
    let wordsz = sess.target.pointer_width.to_string();
    let os = &sess.target.os;
    let env = &sess.target.env;
    let vendor = &sess.target.vendor;
    let min_atomic_width = sess.target.min_atomic_width();
    let max_atomic_width = sess.target.max_atomic_width();
    let atomic_cas = sess.target.atomic_cas;
    let layout = TargetDataLayout::parse(&sess.target).unwrap_or_else(|err| {
        sess.fatal(&err);
    });

    let mut ret = FxHashSet::default();
    ret.reserve(6); // the minimum number of insertions
    // Target bindings.
    ret.insert((sym::target_os, Some(Symbol::intern(os))));
    if let Some(ref fam) = sess.target.os_family {
        ret.insert((sym::target_family, Some(Symbol::intern(fam))));
        if fam == "windows" {
            ret.insert((sym::windows, None));
        } else if fam == "unix" {
            ret.insert((sym::unix, None));
        }
    }
    ret.insert((sym::target_arch, Some(Symbol::intern(arch))));
    ret.insert((sym::target_endian, Some(Symbol::intern(end))));
    ret.insert((sym::target_pointer_width, Some(Symbol::intern(&wordsz))));
    ret.insert((sym::target_env, Some(Symbol::intern(env))));
    ret.insert((sym::target_vendor, Some(Symbol::intern(vendor))));
    if sess.target.has_elf_tls {
        ret.insert((sym::target_thread_local, None));
    }
    for &(i, align) in &[
        (8, layout.i8_align.abi),
        (16, layout.i16_align.abi),
        (32, layout.i32_align.abi),
        (64, layout.i64_align.abi),
        (128, layout.i128_align.abi),
    ] {
        if i >= min_atomic_width && i <= max_atomic_width {
            let mut insert_atomic = |s, align: Align| {
                ret.insert((sym::target_has_atomic_load_store, Some(Symbol::intern(s))));
                if atomic_cas {
                    ret.insert((sym::target_has_atomic, Some(Symbol::intern(s))));
                }
                if align.bits() == i {
                    ret.insert((sym::target_has_atomic_equal_alignment, Some(Symbol::intern(s))));
                }
            };
            let s = i.to_string();
            insert_atomic(&s, align);
            if s == wordsz {
                insert_atomic("ptr", layout.pointer_align.abi);
            }
        }
    }

    let panic_strategy = sess.panic_strategy();
    ret.insert((sym::panic, Some(panic_strategy.desc_symbol())));

    for s in sess.opts.debugging_opts.sanitizer {
        let symbol = Symbol::intern(&s.to_string());
        ret.insert((sym::sanitize, Some(symbol)));
    }

    if sess.opts.debug_assertions {
        ret.insert((sym::debug_assertions, None));
    }
    if sess.opts.crate_types.contains(&CrateType::ProcMacro) {
        ret.insert((sym::proc_macro, None));
    }
    ret
}

/// Converts the crate `cfg!` configuration from `String` to `Symbol`.
/// `rustc_interface::interface::Config` accepts this in the compiler configuration,
/// but the symbol interner is not yet set up then, so we must convert it later.
pub fn to_crate_config(cfg: FxHashSet<(String, Option<String>)>) -> CrateConfig {
    cfg.into_iter().map(|(a, b)| (Symbol::intern(&a), b.map(|b| Symbol::intern(&b)))).collect()
}

pub fn build_configuration(sess: &Session, mut user_cfg: CrateConfig) -> CrateConfig {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items.
    let default_cfg = default_configuration(sess);
    // If the user wants a test runner, then add the test cfg.
    if sess.opts.test {
        user_cfg.insert((sym::test, None));
    }
    user_cfg.extend(default_cfg.iter().cloned());
    user_cfg
}

pub fn build_target_config(opts: &Options, target_override: Option<Target>) -> Target {
    let target_result = target_override.map_or_else(|| Target::search(&opts.target_triple), Ok);
    let target = target_result.unwrap_or_else(|e| {
        early_error(
            opts.error_format,
            &format!(
                "Error loading target specification: {}. \
            Use `--print target-list` for a list of built-in targets",
                e
            ),
        )
    });

    if !matches!(target.pointer_width, 16 | 32 | 64) {
        early_error(
            opts.error_format,
            &format!(
                "target specification was invalid: \
             unrecognized target-pointer-width {}",
                target.pointer_width
            ),
        )
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
// is exposed by .  The public
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
    pub fn flagopt_s(a: S, b: S, c: S, d: S) -> R {
        stable(longer(a, b), move |opts| opts.optflagopt(a, b, c, d))
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
    pub fn flag(a: S, b: S, c: S) -> R {
        unstable(longer(a, b), move |opts| opts.optflag(a, b, c))
    }
    pub fn flagopt(a: S, b: S, c: S, d: S) -> R {
        unstable(longer(a, b), move |opts| opts.optflagopt(a, b, c, d))
    }
    pub fn flagmulti(a: S, b: S, c: S) -> R {
        unstable(longer(a, b), move |opts| opts.optflagmulti(a, b, c))
    }
}

/// Returns the "short" subset of the rustc command line options,
/// including metadata for each option, such as whether the option is
/// part of the stable long-term interface for rustc.
pub fn rustc_short_optgroups() -> Vec<RustcOptGroup> {
    vec![
        opt::flag_s("h", "help", "Display this message"),
        opt::multi_s("", "cfg", "Configure the compilation environment", "SPEC"),
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
                             static, framework, or dylib (the default).",
            "[KIND=]NAME",
        ),
        make_crate_type_option(),
        opt::opt_s("", "crate-name", "Specify the name of the crate being built", "NAME"),
        opt::opt_s(
            "",
            "edition",
            "Specify which edition of the compiler to use when compiling code.",
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
            "[crate-name|file-names|sysroot|target-libdir|cfg|target-list|\
             target-cpus|target-features|relocation-models|\
             code-models|tls-models|target-spec-json|native-static-libs]",
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
        opt::multi_s("W", "warn", "Set lint warnings", "OPT"),
        opt::multi_s("A", "allow", "Set lint allowed", "OPT"),
        opt::multi_s("D", "deny", "Set lint denied", "OPT"),
        opt::multi_s("F", "forbid", "Set lint forbidden", "OPT"),
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
    opts.extend(vec![
        opt::multi_s(
            "",
            "extern",
            "Specify where an external rust library is located",
            "NAME[=PATH]",
        ),
        opt::opt_s("", "sysroot", "Override the system root", "PATH"),
        opt::multi("Z", "", "Set internal debugging options", "FLAG"),
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
        opt::opt(
            "",
            "pretty",
            "Pretty-print the input instead of compiling;
                  valid types are: `normal` (un-annotated source),
                  `expanded` (crates expanded), or
                  `expanded,identified` (fully parenthesized, AST nodes with IDs).",
            "TYPE",
        ),
        opt::multi_s(
            "",
            "remap-path-prefix",
            "Remap source names in all output (compiler messages and output files)",
            "FROM=TO",
        ),
    ]);
    opts
}

pub fn get_cmd_lint_options(
    matches: &getopts::Matches,
    error_format: ErrorOutputType,
) -> (Vec<(String, lint::Level)>, bool, Option<lint::Level>) {
    let mut lint_opts_with_position = vec![];
    let mut describe_lints = false;

    for &level in &[lint::Allow, lint::Warn, lint::Deny, lint::Forbid] {
        for (passed_arg_pos, lint_name) in matches.opt_strs_pos(level.as_str()) {
            let arg_pos = if let lint::Forbid = level {
                // HACK: forbid is always specified last, so it can't be overridden.
                // FIXME: remove this once <https://github.com/rust-lang/rust/issues/70819> is
                // fixed and `forbid` works as expected.
                usize::MAX
            } else {
                passed_arg_pos
            };
            if lint_name == "help" {
                describe_lints = true;
            } else {
                lint_opts_with_position.push((arg_pos, lint_name.replace("-", "_"), level));
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
            .unwrap_or_else(|| early_error(error_format, &format!("unknown lint level: `{}`", cap)))
    });
    (lint_opts, describe_lints, lint_cap)
}

/// Parses the `--color` flag.
pub fn parse_color(matches: &getopts::Matches) -> ColorConfig {
    match matches.opt_str("color").as_ref().map(|s| &s[..]) {
        Some("auto") => ColorConfig::Auto,
        Some("always") => ColorConfig::Always,
        Some("never") => ColorConfig::Never,

        None => ColorConfig::Auto,

        Some(arg) => early_error(
            ErrorOutputType::default(),
            &format!(
                "argument for `--color` must be auto, \
                 always or never (instead was `{}`)",
                arg
            ),
        ),
    }
}

/// Parse the `--json` flag.
///
/// The first value returned is how to render JSON diagnostics, and the second
/// is whether or not artifact notifications are enabled.
pub fn parse_json(matches: &getopts::Matches) -> (HumanReadableErrorType, bool) {
    let mut json_rendered: fn(ColorConfig) -> HumanReadableErrorType =
        HumanReadableErrorType::Default;
    let mut json_color = ColorConfig::Never;
    let mut json_artifact_notifications = false;
    for option in matches.opt_strs("json") {
        // For now conservatively forbid `--color` with `--json` since `--json`
        // won't actually be emitting any colors and anything colorized is
        // embedded in a diagnostic message anyway.
        if matches.opt_str("color").is_some() {
            early_error(
                ErrorOutputType::default(),
                "cannot specify the `--color` option with `--json`",
            );
        }

        for sub_option in option.split(',') {
            match sub_option {
                "diagnostic-short" => json_rendered = HumanReadableErrorType::Short,
                "diagnostic-rendered-ansi" => json_color = ColorConfig::Always,
                "artifacts" => json_artifact_notifications = true,
                s => early_error(
                    ErrorOutputType::default(),
                    &format!("unknown `--json` option `{}`", s),
                ),
            }
        }
    }
    (json_rendered(json_color), json_artifact_notifications)
}

/// Parses the `--error-format` flag.
pub fn parse_error_format(
    matches: &getopts::Matches,
    color: ColorConfig,
    json_rendered: HumanReadableErrorType,
) -> ErrorOutputType {
    // We need the `opts_present` check because the driver will send us Matches
    // with only stable options if no unstable options are used. Since error-format
    // is unstable, it will not be present. We have to use `opts_present` not
    // `opt_present` because the latter will panic.
    let error_format = if matches.opts_present(&["error-format".to_owned()]) {
        match matches.opt_str("error-format").as_ref().map(|s| &s[..]) {
            None | Some("human") => {
                ErrorOutputType::HumanReadable(HumanReadableErrorType::Default(color))
            }
            Some("human-annotate-rs") => {
                ErrorOutputType::HumanReadable(HumanReadableErrorType::AnnotateSnippet(color))
            }
            Some("json") => ErrorOutputType::Json { pretty: false, json_rendered },
            Some("pretty-json") => ErrorOutputType::Json { pretty: true, json_rendered },
            Some("short") => ErrorOutputType::HumanReadable(HumanReadableErrorType::Short(color)),

            Some(arg) => early_error(
                ErrorOutputType::HumanReadable(HumanReadableErrorType::Default(color)),
                &format!(
                    "argument for `--error-format` must be `human`, `json` or \
                     `short` (instead was `{}`)",
                    arg
                ),
            ),
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
            early_error(
                ErrorOutputType::default(),
                "using `--json` requires also using `--error-format=json`",
            );
        }

        _ => {}
    }

    error_format
}

fn parse_crate_edition(matches: &getopts::Matches) -> Edition {
    let edition = match matches.opt_str("edition") {
        Some(arg) => Edition::from_str(&arg).unwrap_or_else(|_| {
            early_error(
                ErrorOutputType::default(),
                &format!(
                    "argument for `--edition` must be one of: \
                     {}. (instead was `{}`)",
                    EDITION_NAME_LIST, arg
                ),
            )
        }),
        None => DEFAULT_EDITION,
    };

    if !edition.is_stable() && !nightly_options::match_is_nightly_build(matches) {
        early_error(
            ErrorOutputType::default(),
            &format!(
                "edition {} is unstable and only \
                     available for nightly builds of rustc.",
                edition,
            ),
        )
    }

    edition
}

fn check_debug_option_stability(
    debugging_opts: &DebuggingOptions,
    error_format: ErrorOutputType,
    json_rendered: HumanReadableErrorType,
) {
    if !debugging_opts.unstable_options {
        if let ErrorOutputType::Json { pretty: true, json_rendered } = error_format {
            early_error(
                ErrorOutputType::Json { pretty: false, json_rendered },
                "`--error-format=pretty-json` is unstable",
            );
        }
        if let ErrorOutputType::HumanReadable(HumanReadableErrorType::AnnotateSnippet(_)) =
            error_format
        {
            early_error(
                ErrorOutputType::Json { pretty: false, json_rendered },
                "`--error-format=human-annotate-rs` is unstable",
            );
        }
    }
}

fn parse_output_types(
    debugging_opts: &DebuggingOptions,
    matches: &getopts::Matches,
    error_format: ErrorOutputType,
) -> OutputTypes {
    let mut output_types = BTreeMap::new();
    if !debugging_opts.parse_only {
        for list in matches.opt_strs("emit") {
            for output_type in list.split(',') {
                let (shorthand, path) = match output_type.split_once('=') {
                    None => (output_type, None),
                    Some((shorthand, path)) => (shorthand, Some(PathBuf::from(path))),
                };
                let output_type = OutputType::from_shorthand(shorthand).unwrap_or_else(|| {
                    early_error(
                        error_format,
                        &format!(
                            "unknown emission type: `{}` - expected one of: {}",
                            shorthand,
                            OutputType::shorthands_display(),
                        ),
                    )
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

fn should_override_cgus_and_disable_thinlto(
    output_types: &OutputTypes,
    matches: &getopts::Matches,
    error_format: ErrorOutputType,
    mut codegen_units: Option<usize>,
) -> (bool, Option<usize>) {
    let mut disable_thinlto = false;
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
                        early_warn(
                            error_format,
                            &format!(
                                "`--emit={}` with `-o` incompatible with \
                                 `-C codegen-units=N` for N > 1",
                                ot
                            ),
                        );
                    }
                    early_warn(error_format, "resetting to default -C codegen-units=1");
                    codegen_units = Some(1);
                    disable_thinlto = true;
                }
            }
            _ => {
                codegen_units = Some(1);
                disable_thinlto = true;
            }
        }
    }

    if codegen_units == Some(0) {
        early_error(error_format, "value for codegen units must be a positive non-zero integer");
    }

    (disable_thinlto, codegen_units)
}

fn check_thread_count(debugging_opts: &DebuggingOptions, error_format: ErrorOutputType) {
    if debugging_opts.threads == 0 {
        early_error(error_format, "value for threads must be a positive non-zero integer");
    }

    if debugging_opts.threads > 1 && debugging_opts.fuel.is_some() {
        early_error(error_format, "optimization fuel is incompatible with multiple threads");
    }
}

fn collect_print_requests(
    cg: &mut CodegenOptions,
    dopts: &mut DebuggingOptions,
    matches: &getopts::Matches,
    error_format: ErrorOutputType,
) -> Vec<PrintRequest> {
    let mut prints = Vec::<PrintRequest>::new();
    if cg.target_cpu.as_ref().map_or(false, |s| s == "help") {
        prints.push(PrintRequest::TargetCPUs);
        cg.target_cpu = None;
    };
    if cg.target_feature == "help" {
        prints.push(PrintRequest::TargetFeatures);
        cg.target_feature = String::new();
    }

    prints.extend(matches.opt_strs("print").into_iter().map(|s| match &*s {
        "crate-name" => PrintRequest::CrateName,
        "file-names" => PrintRequest::FileNames,
        "sysroot" => PrintRequest::Sysroot,
        "target-libdir" => PrintRequest::TargetLibdir,
        "cfg" => PrintRequest::Cfg,
        "target-list" => PrintRequest::TargetList,
        "target-cpus" => PrintRequest::TargetCPUs,
        "target-features" => PrintRequest::TargetFeatures,
        "relocation-models" => PrintRequest::RelocationModels,
        "code-models" => PrintRequest::CodeModels,
        "tls-models" => PrintRequest::TlsModels,
        "native-static-libs" => PrintRequest::NativeStaticLibs,
        "target-spec-json" => {
            if dopts.unstable_options {
                PrintRequest::TargetSpec
            } else {
                early_error(
                    error_format,
                    "the `-Z unstable-options` flag must also be passed to \
                     enable the target-spec-json print option",
                );
            }
        }
        req => early_error(error_format, &format!("unknown print request `{}`", req)),
    }));

    prints
}

fn parse_target_triple(matches: &getopts::Matches, error_format: ErrorOutputType) -> TargetTriple {
    match matches.opt_str("target") {
        Some(target) if target.ends_with(".json") => {
            let path = Path::new(&target);
            TargetTriple::from_path(&path).unwrap_or_else(|_| {
                early_error(error_format, &format!("target file {:?} does not exist", path))
            })
        }
        Some(target) => TargetTriple::TargetTriple(target),
        _ => TargetTriple::from_triple(host_triple()),
    }
}

fn parse_opt_level(
    matches: &getopts::Matches,
    cg: &CodegenOptions,
    error_format: ErrorOutputType,
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
            if let Some("opt-level") = s.splitn(2, '=').next() { Some(i) } else { None }
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
                early_error(
                    error_format,
                    &format!(
                        "optimization level needs to be \
                            between 0-3, s or z (instead was `{}`)",
                        arg
                    ),
                );
            }
        }
    }
}

fn select_debuginfo(
    matches: &getopts::Matches,
    cg: &CodegenOptions,
    error_format: ErrorOutputType,
) -> DebugInfo {
    let max_g = matches.opt_positions("g").into_iter().max();
    let max_c = matches
        .opt_strs_pos("C")
        .into_iter()
        .flat_map(|(i, s)| {
            // NB: This can match a string without `=`.
            if let Some("debuginfo") = s.splitn(2, '=').next() { Some(i) } else { None }
        })
        .max();
    if max_g > max_c {
        DebugInfo::Full
    } else {
        match cg.debuginfo {
            0 => DebugInfo::None,
            1 => DebugInfo::Limited,
            2 => DebugInfo::Full,
            arg => {
                early_error(
                    error_format,
                    &format!(
                        "debug info level needs to be between \
                         0-2 (instead was `{}`)",
                        arg
                    ),
                );
            }
        }
    }
}

fn parse_libs(
    matches: &getopts::Matches,
    error_format: ErrorOutputType,
) -> Vec<(String, Option<String>, NativeLibKind)> {
    matches
        .opt_strs("l")
        .into_iter()
        .map(|s| {
            // Parse string of the form "[KIND=]lib[:new_name]",
            // where KIND is one of "dylib", "framework", "static".
            let (name, kind) = match s.split_once('=') {
                None => (s, NativeLibKind::Unspecified),
                Some((kind, name)) => {
                    let kind = match kind {
                        "dylib" => NativeLibKind::Dylib,
                        "framework" => NativeLibKind::Framework,
                        "static" => NativeLibKind::StaticBundle,
                        "static-nobundle" => NativeLibKind::StaticNoBundle,
                        s => {
                            early_error(
                                error_format,
                                &format!(
                                    "unknown library kind `{}`, expected \
                                     one of dylib, framework, or static",
                                    s
                                ),
                            );
                        }
                    };
                    (name.to_string(), kind)
                }
            };
            if kind == NativeLibKind::StaticNoBundle
                && !nightly_options::match_is_nightly_build(matches)
            {
                early_error(
                    error_format,
                    "the library kind 'static-nobundle' is only \
                     accepted on the nightly compiler",
                );
            }
            let (name, new_name) = match name.split_once(':') {
                None => (name, None),
                Some((name, new_name)) => (name.to_string(), Some(new_name.to_owned())),
            };
            (name, new_name, kind)
        })
        .collect()
}

fn parse_borrowck_mode(dopts: &DebuggingOptions, error_format: ErrorOutputType) -> BorrowckMode {
    match dopts.borrowck.as_ref() {
        "migrate" => BorrowckMode::Migrate,
        "mir" => BorrowckMode::Mir,
        m => early_error(error_format, &format!("unknown borrowck mode `{}`", m)),
    }
}

pub fn parse_externs(
    matches: &getopts::Matches,
    debugging_opts: &DebuggingOptions,
    error_format: ErrorOutputType,
) -> Externs {
    let is_unstable_enabled = debugging_opts.unstable_options;
    let mut externs: BTreeMap<String, ExternEntry> = BTreeMap::new();
    for arg in matches.opt_strs("extern") {
        let (name, path) = match arg.split_once('=') {
            None => (arg, None),
            Some((name, path)) => (name.to_string(), Some(path.to_string())),
        };
        let (options, name) = match name.split_once(':') {
            None => (None, name),
            Some((opts, name)) => (Some(opts), name.to_string()),
        };

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
        if let Some(opts) = options {
            if !is_unstable_enabled {
                early_error(
                    error_format,
                    "the `-Z unstable-options` flag must also be passed to \
                     enable `--extern options",
                );
            }
            for opt in opts.split(',') {
                match opt {
                    "priv" => is_private_dep = true,
                    "noprelude" => {
                        if let ExternLocation::ExactPaths(_) = &entry.location {
                            add_prelude = false;
                        } else {
                            early_error(
                                error_format,
                                "the `noprelude` --extern option requires a file path",
                            );
                        }
                    }
                    _ => early_error(error_format, &format!("unknown --extern option `{}`", opt)),
                }
            }
        }

        // Crates start out being not private, and go to being private `priv`
        // is specified.
        entry.is_private_dep |= is_private_dep;
        // If any flag is missing `noprelude`, then add to the prelude.
        entry.add_prelude |= add_prelude;
    }
    Externs(externs)
}

fn parse_remap_path_prefix(
    matches: &getopts::Matches,
    error_format: ErrorOutputType,
) -> Vec<(PathBuf, PathBuf)> {
    matches
        .opt_strs("remap-path-prefix")
        .into_iter()
        .map(|remap| match remap.rsplit_once('=') {
            None => early_error(
                error_format,
                "--remap-path-prefix must contain '=' between FROM and TO",
            ),
            Some((from, to)) => (PathBuf::from(from), PathBuf::from(to)),
        })
        .collect()
}

pub fn build_session_options(matches: &getopts::Matches) -> Options {
    let color = parse_color(matches);

    let edition = parse_crate_edition(matches);

    let (json_rendered, json_artifact_notifications) = parse_json(matches);

    let error_format = parse_error_format(matches, color, json_rendered);

    let unparsed_crate_types = matches.opt_strs("crate-type");
    let crate_types = parse_crate_types_from_list(unparsed_crate_types)
        .unwrap_or_else(|e| early_error(error_format, &e[..]));

    let (lint_opts, describe_lints, lint_cap) = get_cmd_lint_options(matches, error_format);

    let mut debugging_opts = build_debugging_options(matches, error_format);
    check_debug_option_stability(&debugging_opts, error_format, json_rendered);

    let output_types = parse_output_types(&debugging_opts, matches, error_format);

    let mut cg = build_codegen_options(matches, error_format);
    let (disable_thinlto, mut codegen_units) = should_override_cgus_and_disable_thinlto(
        &output_types,
        matches,
        error_format,
        cg.codegen_units,
    );

    check_thread_count(&debugging_opts, error_format);

    let incremental = cg.incremental.as_ref().map(PathBuf::from);

    if debugging_opts.profile && incremental.is_some() {
        early_error(
            error_format,
            "can't instrument with gcov profiling when compiling incrementally",
        );
    }
    if debugging_opts.profile {
        match codegen_units {
            Some(1) => {}
            None => codegen_units = Some(1),
            Some(_) => early_error(
                error_format,
                "can't instrument with gcov profiling with multiple codegen units",
            ),
        }
    }

    if cg.profile_generate.enabled() && cg.profile_use.is_some() {
        early_error(
            error_format,
            "options `-C profile-generate` and `-C profile-use` are exclusive",
        );
    }

    if debugging_opts.instrument_coverage {
        if cg.profile_generate.enabled() || cg.profile_use.is_some() {
            early_error(
                error_format,
                "option `-Z instrument-coverage` is not compatible with either `-C profile-use` \
                or `-C profile-generate`",
            );
        }

        // `-Z instrument-coverage` implies `-Z symbol-mangling-version=v0` - to ensure consistent
        // and reversible name mangling. Note, LLVM coverage tools can analyze coverage over
        // multiple runs, including some changes to source code; so mangled names must be consistent
        // across compilations.
        match debugging_opts.symbol_mangling_version {
            None => {
                debugging_opts.symbol_mangling_version = Some(SymbolManglingVersion::V0);
            }
            Some(SymbolManglingVersion::Legacy) => {
                early_warn(
                    error_format,
                    "-Z instrument-coverage requires symbol mangling version `v0`, \
                    but `-Z symbol-mangling-version=legacy` was specified",
                );
            }
            Some(SymbolManglingVersion::V0) => {}
        }

        if debugging_opts.mir_opt_level > 1 {
            early_warn(
                error_format,
                &format!(
                    "`-Z mir-opt-level={}` (any level > 1) enables function inlining, which \
                    limits the effectiveness of `-Z instrument-coverage`.",
                    debugging_opts.mir_opt_level,
                ),
            );
        }
    }

    if let Ok(graphviz_font) = std::env::var("RUSTC_GRAPHVIZ_FONT") {
        debugging_opts.graphviz_font = graphviz_font;
    }

    if !cg.embed_bitcode {
        match cg.lto {
            LtoCli::No | LtoCli::Unspecified => {}
            LtoCli::Yes | LtoCli::NoParam | LtoCli::Thin | LtoCli::Fat => early_error(
                error_format,
                "options `-C embed-bitcode=no` and `-C lto` are incompatible",
            ),
        }
    }

    let prints = collect_print_requests(&mut cg, &mut debugging_opts, matches, error_format);

    let cg = cg;

    let sysroot_opt = matches.opt_str("sysroot").map(|m| PathBuf::from(&m));
    let target_triple = parse_target_triple(matches, error_format);
    let opt_level = parse_opt_level(matches, &cg, error_format);
    // The `-g` and `-C debuginfo` flags specify the same setting, so we want to be able
    // to use them interchangeably. See the note above (regarding `-O` and `-C opt-level`)
    // for more details.
    let debug_assertions = cg.debug_assertions.unwrap_or(opt_level == OptLevel::No);
    let debuginfo = select_debuginfo(matches, &cg, error_format);

    let mut search_paths = vec![];
    for s in &matches.opt_strs("L") {
        search_paths.push(SearchPath::from_cli_opt(&s[..], error_format));
    }

    let libs = parse_libs(matches, error_format);

    let test = matches.opt_present("test");

    let borrowck_mode = parse_borrowck_mode(&debugging_opts, error_format);

    if !cg.remark.is_empty() && debuginfo == DebugInfo::None {
        early_warn(error_format, "-C remark requires \"-C debuginfo=n\" to show source locations");
    }

    let externs = parse_externs(matches, &debugging_opts, error_format);

    let crate_name = matches.opt_str("crate-name");

    let remap_path_prefix = parse_remap_path_prefix(matches, error_format);

    let pretty = parse_pretty(matches, &debugging_opts, error_format);

    Options {
        crate_types,
        optimize: opt_level,
        debuginfo,
        lint_opts,
        lint_cap,
        describe_lints,
        output_types,
        search_paths,
        maybe_sysroot: sysroot_opt,
        target_triple,
        test,
        incremental,
        debugging_opts,
        prints,
        borrowck_mode,
        cg,
        error_format,
        externs,
        unstable_features: UnstableFeatures::from_environment(crate_name.as_deref()),
        crate_name,
        alt_std_name: None,
        libs,
        debug_assertions,
        actually_rustdoc: false,
        trimmed_def_paths: TrimmedDefPaths::default(),
        cli_forced_codegen_units: codegen_units,
        cli_forced_thinlto_off: disable_thinlto,
        remap_path_prefix,
        edition,
        json_artifact_notifications,
        pretty,
    }
}

fn parse_pretty(
    matches: &getopts::Matches,
    debugging_opts: &DebuggingOptions,
    efmt: ErrorOutputType,
) -> Option<PpMode> {
    let pretty = if debugging_opts.unstable_options {
        matches.opt_default("pretty", "normal").map(|a| {
            // stable pretty-print variants only
            parse_pretty_inner(efmt, &a, false)
        })
    } else {
        None
    };

    return if pretty.is_none() {
        debugging_opts.unpretty.as_ref().map(|a| {
            // extended with unstable pretty-print variants
            parse_pretty_inner(efmt, &a, true)
        })
    } else {
        pretty
    };

    fn parse_pretty_inner(efmt: ErrorOutputType, name: &str, extended: bool) -> PpMode {
        use PpMode::*;
        use PpSourceMode::*;
        let first = match (name, extended) {
            ("normal", _) => PpmSource(PpmNormal),
            ("identified", _) => PpmSource(PpmIdentified),
            ("everybody_loops", true) => PpmSource(PpmEveryBodyLoops),
            ("expanded", _) => PpmSource(PpmExpanded),
            ("expanded,identified", _) => PpmSource(PpmExpandedIdentified),
            ("expanded,hygiene", _) => PpmSource(PpmExpandedHygiene),
            ("hir", true) => PpmHir(PpmNormal),
            ("hir,identified", true) => PpmHir(PpmIdentified),
            ("hir,typed", true) => PpmHir(PpmTyped),
            ("hir-tree", true) => PpmHirTree(PpmNormal),
            ("mir", true) => PpmMir,
            ("mir-cfg", true) => PpmMirCFG,
            _ => {
                if extended {
                    early_error(
                        efmt,
                        &format!(
                            "argument to `unpretty` must be one of `normal`, \
                                        `expanded`, `identified`, `expanded,identified`, \
                                        `expanded,hygiene`, `everybody_loops`, \
                                        `hir`, `hir,identified`, `hir,typed`, `hir-tree`, \
                                        `mir` or `mir-cfg`; got {}",
                            name
                        ),
                    );
                } else {
                    early_error(
                        efmt,
                        &format!(
                            "argument to `pretty` must be one of `normal`, \
                                        `expanded`, `identified`, or `expanded,identified`; got {}",
                            name
                        ),
                    );
                }
            }
        };
        tracing::debug!("got unpretty option: {:?}", first);
        first
    }
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
                _ => return Err(format!("unknown crate type: `{}`", part)),
            };
            if !crate_types.contains(&new_part) {
                crate_types.push(new_part)
            }
        }
    }

    Ok(crate_types)
}

pub mod nightly_options {
    use super::{ErrorOutputType, OptionStability, RustcOptGroup};
    use crate::early_error;
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

    pub fn check_nightly_options(matches: &getopts::Matches, flags: &[RustcOptGroup]) {
        let has_z_unstable_option = matches.opt_strs("Z").iter().any(|x| *x == "unstable-options");
        let really_allows_unstable_options = match_is_nightly_build(matches);

        for opt in flags.iter() {
            if opt.stability == OptionStability::Stable {
                continue;
            }
            if !matches.opt_present(opt.name) {
                continue;
            }
            if opt.name != "Z" && !has_z_unstable_option {
                early_error(
                    ErrorOutputType::default(),
                    &format!(
                        "the `-Z unstable-options` flag must also be passed to enable \
                         the flag `{}`",
                        opt.name
                    ),
                );
            }
            if really_allows_unstable_options {
                continue;
            }
            match opt.stability {
                OptionStability::Unstable => {
                    let msg = format!(
                        "the option `{}` is only accepted on the \
                         nightly compiler",
                        opt.name
                    );
                    early_error(ErrorOutputType::default(), &msg);
                }
                OptionStability::Stable => {}
            }
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

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpSourceMode {
    PpmNormal,
    PpmEveryBodyLoops,
    PpmExpanded,
    PpmIdentified,
    PpmExpandedIdentified,
    PpmExpandedHygiene,
    PpmTyped,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpMode {
    PpmSource(PpSourceMode),
    PpmHir(PpSourceMode),
    PpmHirTree(PpSourceMode),
    PpmMir,
    PpmMirCFG,
}

impl PpMode {
    pub fn needs_ast_map(&self) -> bool {
        use PpMode::*;
        use PpSourceMode::*;
        match *self {
            PpmSource(PpmNormal | PpmIdentified) => false,

            PpmSource(
                PpmExpanded | PpmEveryBodyLoops | PpmExpandedIdentified | PpmExpandedHygiene,
            )
            | PpmHir(_)
            | PpmHirTree(_)
            | PpmMir
            | PpmMirCFG => true,
            PpmSource(PpmTyped) => panic!("invalid state"),
        }
    }

    pub fn needs_analysis(&self) -> bool {
        use PpMode::*;
        matches!(*self, PpmMir | PpmMirCFG)
    }
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
crate mod dep_tracking {
    use super::{
        CFGuard, CrateType, DebugInfo, ErrorOutputType, LinkerPluginLto, LtoCli, OptLevel,
        OutputTypes, Passes, SanitizerSet, SourceFileHashAlgorithm, SwitchWithOptPath,
        SymbolManglingVersion, TrimmedDefPaths,
    };
    use crate::lint;
    use crate::utils::NativeLibKind;
    use rustc_feature::UnstableFeatures;
    use rustc_span::edition::Edition;
    use rustc_target::spec::{CodeModel, MergeFunctions, PanicStrategy, RelocModel};
    use rustc_target::spec::{RelroLevel, TargetTriple, TlsModel};
    use std::collections::hash_map::DefaultHasher;
    use std::collections::BTreeMap;
    use std::hash::Hash;
    use std::path::PathBuf;

    pub trait DepTrackingHash {
        fn hash(&self, hasher: &mut DefaultHasher, error_format: ErrorOutputType);
    }

    macro_rules! impl_dep_tracking_hash_via_hash {
        ($t:ty) => {
            impl DepTrackingHash for $t {
                fn hash(&self, hasher: &mut DefaultHasher, _: ErrorOutputType) {
                    Hash::hash(self, hasher);
                }
            }
        };
    }

    macro_rules! impl_dep_tracking_hash_for_sortable_vec_of {
        ($t:ty) => {
            impl DepTrackingHash for Vec<$t> {
                fn hash(&self, hasher: &mut DefaultHasher, error_format: ErrorOutputType) {
                    let mut elems: Vec<&$t> = self.iter().collect();
                    elems.sort();
                    Hash::hash(&elems.len(), hasher);
                    for (index, elem) in elems.iter().enumerate() {
                        Hash::hash(&index, hasher);
                        DepTrackingHash::hash(*elem, hasher, error_format);
                    }
                }
            }
        };
    }

    impl_dep_tracking_hash_via_hash!(bool);
    impl_dep_tracking_hash_via_hash!(usize);
    impl_dep_tracking_hash_via_hash!(u64);
    impl_dep_tracking_hash_via_hash!(String);
    impl_dep_tracking_hash_via_hash!(PathBuf);
    impl_dep_tracking_hash_via_hash!(lint::Level);
    impl_dep_tracking_hash_via_hash!(Option<bool>);
    impl_dep_tracking_hash_via_hash!(Option<usize>);
    impl_dep_tracking_hash_via_hash!(Option<String>);
    impl_dep_tracking_hash_via_hash!(Option<(String, u64)>);
    impl_dep_tracking_hash_via_hash!(Option<Vec<String>>);
    impl_dep_tracking_hash_via_hash!(Option<MergeFunctions>);
    impl_dep_tracking_hash_via_hash!(Option<RelocModel>);
    impl_dep_tracking_hash_via_hash!(Option<CodeModel>);
    impl_dep_tracking_hash_via_hash!(Option<TlsModel>);
    impl_dep_tracking_hash_via_hash!(Option<PanicStrategy>);
    impl_dep_tracking_hash_via_hash!(Option<RelroLevel>);
    impl_dep_tracking_hash_via_hash!(Option<lint::Level>);
    impl_dep_tracking_hash_via_hash!(Option<PathBuf>);
    impl_dep_tracking_hash_via_hash!(CrateType);
    impl_dep_tracking_hash_via_hash!(MergeFunctions);
    impl_dep_tracking_hash_via_hash!(PanicStrategy);
    impl_dep_tracking_hash_via_hash!(RelroLevel);
    impl_dep_tracking_hash_via_hash!(Passes);
    impl_dep_tracking_hash_via_hash!(OptLevel);
    impl_dep_tracking_hash_via_hash!(LtoCli);
    impl_dep_tracking_hash_via_hash!(DebugInfo);
    impl_dep_tracking_hash_via_hash!(UnstableFeatures);
    impl_dep_tracking_hash_via_hash!(OutputTypes);
    impl_dep_tracking_hash_via_hash!(NativeLibKind);
    impl_dep_tracking_hash_via_hash!(SanitizerSet);
    impl_dep_tracking_hash_via_hash!(CFGuard);
    impl_dep_tracking_hash_via_hash!(TargetTriple);
    impl_dep_tracking_hash_via_hash!(Edition);
    impl_dep_tracking_hash_via_hash!(LinkerPluginLto);
    impl_dep_tracking_hash_via_hash!(SwitchWithOptPath);
    impl_dep_tracking_hash_via_hash!(Option<SymbolManglingVersion>);
    impl_dep_tracking_hash_via_hash!(Option<SourceFileHashAlgorithm>);
    impl_dep_tracking_hash_via_hash!(TrimmedDefPaths);

    impl_dep_tracking_hash_for_sortable_vec_of!(String);
    impl_dep_tracking_hash_for_sortable_vec_of!(PathBuf);
    impl_dep_tracking_hash_for_sortable_vec_of!(CrateType);
    impl_dep_tracking_hash_for_sortable_vec_of!((String, lint::Level));
    impl_dep_tracking_hash_for_sortable_vec_of!((String, Option<String>, NativeLibKind));
    impl_dep_tracking_hash_for_sortable_vec_of!((String, u64));

    impl<T1, T2> DepTrackingHash for (T1, T2)
    where
        T1: DepTrackingHash,
        T2: DepTrackingHash,
    {
        fn hash(&self, hasher: &mut DefaultHasher, error_format: ErrorOutputType) {
            Hash::hash(&0, hasher);
            DepTrackingHash::hash(&self.0, hasher, error_format);
            Hash::hash(&1, hasher);
            DepTrackingHash::hash(&self.1, hasher, error_format);
        }
    }

    impl<T1, T2, T3> DepTrackingHash for (T1, T2, T3)
    where
        T1: DepTrackingHash,
        T2: DepTrackingHash,
        T3: DepTrackingHash,
    {
        fn hash(&self, hasher: &mut DefaultHasher, error_format: ErrorOutputType) {
            Hash::hash(&0, hasher);
            DepTrackingHash::hash(&self.0, hasher, error_format);
            Hash::hash(&1, hasher);
            DepTrackingHash::hash(&self.1, hasher, error_format);
            Hash::hash(&2, hasher);
            DepTrackingHash::hash(&self.2, hasher, error_format);
        }
    }

    // This is a stable hash because BTreeMap is a sorted container
    pub fn stable_hash(
        sub_hashes: BTreeMap<&'static str, &dyn DepTrackingHash>,
        hasher: &mut DefaultHasher,
        error_format: ErrorOutputType,
    ) {
        for (key, sub_hash) in sub_hashes {
            // Using Hash::hash() instead of DepTrackingHash::hash() is fine for
            // the keys, as they are just plain strings
            Hash::hash(&key.len(), hasher);
            Hash::hash(key, hasher);
            sub_hash.hash(hasher, error_format);
        }
    }
}
