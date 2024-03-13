use crate::config::*;

use crate::search_paths::SearchPath;
use crate::utils::NativeLib;
use crate::{lint, EarlyDiagCtxt};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::profiling::TimePassesFormat;
use rustc_data_structures::stable_hasher::Hash64;
use rustc_errors::ColorConfig;
use rustc_errors::{LanguageIdentifier, TerminalUrl};
use rustc_target::spec::{CodeModel, LinkerFlavorCli, MergeFunctions, PanicStrategy, SanitizerSet};
use rustc_target::spec::{
    RelocModel, RelroLevel, SplitDebuginfo, StackProtector, TargetTriple, TlsModel,
};

use rustc_feature::UnstableFeatures;
use rustc_span::edition::Edition;
use rustc_span::RealFileName;
use rustc_span::SourceFileHashAlgorithm;

use std::collections::BTreeMap;

use std::hash::{DefaultHasher, Hasher};
use std::num::{IntErrorKind, NonZero};
use std::path::PathBuf;
use std::str;

macro_rules! insert {
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr) => {
        if $sub_hashes
            .insert(stringify!($opt_name), $opt_expr as &dyn dep_tracking::DepTrackingHash)
            .is_some()
        {
            panic!("duplicate key in CLI DepTrackingHash: {}", stringify!($opt_name))
        }
    };
}

macro_rules! hash_opt {
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, $_for_crate_hash: ident, [UNTRACKED]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, $_for_crate_hash: ident, [TRACKED]) => {{ insert!($opt_name, $opt_expr, $sub_hashes) }};
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, $for_crate_hash: ident, [TRACKED_NO_CRATE_HASH]) => {{
        if !$for_crate_hash {
            insert!($opt_name, $opt_expr, $sub_hashes)
        }
    }};
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, $_for_crate_hash: ident, [SUBSTRUCT]) => {{}};
}

macro_rules! hash_substruct {
    ($opt_name:ident, $opt_expr:expr, $error_format:expr, $for_crate_hash:expr, $hasher:expr, [UNTRACKED]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $error_format:expr, $for_crate_hash:expr, $hasher:expr, [TRACKED]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $error_format:expr, $for_crate_hash:expr, $hasher:expr, [TRACKED_NO_CRATE_HASH]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $error_format:expr, $for_crate_hash:expr, $hasher:expr, [SUBSTRUCT]) => {
        use crate::config::dep_tracking::DepTrackingHash;
        $opt_expr.dep_tracking_hash($for_crate_hash, $error_format).hash(
            $hasher,
            $error_format,
            $for_crate_hash,
        );
    };
}

macro_rules! top_level_options {
    ( $( #[$top_level_attr:meta] )* pub struct Options { $(
        $( #[$attr:meta] )*
        $opt:ident : $t:ty [$dep_tracking_marker:ident],
    )* } ) => (
        #[derive(Clone)]
        $( #[$top_level_attr] )*
        pub struct Options {
            $(
                $( #[$attr] )*
                pub $opt: $t
            ),*
        }

        impl Options {
            pub fn dep_tracking_hash(&self, for_crate_hash: bool) -> u64 {
                let mut sub_hashes = BTreeMap::new();
                $({
                    hash_opt!($opt,
                                &self.$opt,
                                &mut sub_hashes,
                                for_crate_hash,
                                [$dep_tracking_marker]);
                })*
                let mut hasher = DefaultHasher::new();
                dep_tracking::stable_hash(sub_hashes,
                                          &mut hasher,
                                          self.error_format,
                                          for_crate_hash);
                $({
                    hash_substruct!($opt,
                        &self.$opt,
                        self.error_format,
                        for_crate_hash,
                        &mut hasher,
                        [$dep_tracking_marker]);
                })*
                hasher.finish()
            }
        }
    );
}

top_level_options!(
    /// The top-level command-line options struct.
    ///
    /// For each option, one has to specify how it behaves with regard to the
    /// dependency tracking system of incremental compilation. This is done via the
    /// square-bracketed directive after the field type. The options are:
    ///
    /// - `[TRACKED]`
    /// A change in the given field will cause the compiler to completely clear the
    /// incremental compilation cache before proceeding.
    ///
    /// - `[TRACKED_NO_CRATE_HASH]`
    /// Same as `[TRACKED]`, but will not affect the crate hash. This is useful for options that only
    /// affect the incremental cache.
    ///
    /// - `[UNTRACKED]`
    /// Incremental compilation is not influenced by this option.
    ///
    /// - `[SUBSTRUCT]`
    /// Second-level sub-structs containing more options.
    ///
    /// If you add a new option to this struct or one of the sub-structs like
    /// `CodegenOptions`, think about how it influences incremental compilation. If in
    /// doubt, specify `[TRACKED]`, which is always "correct" but might lead to
    /// unnecessary re-compilation.
    #[rustc_lint_opt_ty]
    pub struct Options {
        /// The crate config requested for the session, which may be combined
        /// with additional crate configurations during the compile process.
        #[rustc_lint_opt_deny_field_access("use `Session::crate_types` instead of this field")]
        crate_types: Vec<CrateType> [TRACKED],
        optimize: OptLevel [TRACKED],
        /// Include the `debug_assertions` flag in dependency tracking, since it
        /// can influence whether overflow checks are done or not.
        debug_assertions: bool [TRACKED],
        debuginfo: DebugInfo [TRACKED],
        debuginfo_compression: DebugInfoCompression [TRACKED],
        lint_opts: Vec<(String, lint::Level)> [TRACKED_NO_CRATE_HASH],
        lint_cap: Option<lint::Level> [TRACKED_NO_CRATE_HASH],
        describe_lints: bool [UNTRACKED],
        output_types: OutputTypes [TRACKED],
        search_paths: Vec<SearchPath> [UNTRACKED],
        libs: Vec<NativeLib> [TRACKED],
        maybe_sysroot: Option<PathBuf> [UNTRACKED],

        target_triple: TargetTriple [TRACKED],

        /// Effective logical environment used by `env!`/`option_env!` macros
        logical_env: FxIndexMap<String, String> [TRACKED],

        test: bool [TRACKED],
        error_format: ErrorOutputType [UNTRACKED],
        diagnostic_width: Option<usize> [UNTRACKED],

        /// If `Some`, enable incremental compilation, using the given
        /// directory to store intermediate results.
        incremental: Option<PathBuf> [UNTRACKED],
        assert_incr_state: Option<IncrementalStateAssertion> [UNTRACKED],
        /// Set by the `Config::hash_untracked_state` callback for custom
        /// drivers to invalidate the incremental cache
        #[rustc_lint_opt_deny_field_access("should only be used via `Config::hash_untracked_state`")]
        untracked_state_hash: Hash64 [TRACKED_NO_CRATE_HASH],

        unstable_opts: UnstableOptions [SUBSTRUCT],
        prints: Vec<PrintRequest> [UNTRACKED],
        cg: CodegenOptions [SUBSTRUCT],
        externs: Externs [UNTRACKED],
        crate_name: Option<String> [TRACKED],
        /// Indicates how the compiler should treat unstable features.
        unstable_features: UnstableFeatures [TRACKED],

        /// Indicates whether this run of the compiler is actually rustdoc. This
        /// is currently just a hack and will be removed eventually, so please
        /// try to not rely on this too much.
        actually_rustdoc: bool [TRACKED],
        /// Whether name resolver should resolve documentation links.
        resolve_doc_links: ResolveDocLinks [TRACKED],

        /// Control path trimming.
        trimmed_def_paths: bool [TRACKED],

        /// Specifications of codegen units / ThinLTO which are forced as a
        /// result of parsing command line options. These are not necessarily
        /// what rustc was invoked with, but massaged a bit to agree with
        /// commands like `--emit llvm-ir` which they're often incompatible with
        /// if we otherwise use the defaults of rustc.
        #[rustc_lint_opt_deny_field_access("use `Session::codegen_units` instead of this field")]
        cli_forced_codegen_units: Option<usize> [UNTRACKED],
        #[rustc_lint_opt_deny_field_access("use `Session::lto` instead of this field")]
        cli_forced_local_thinlto_off: bool [UNTRACKED],

        /// Remap source path prefixes in all output (messages, object files, debug, etc.).
        remap_path_prefix: Vec<(PathBuf, PathBuf)> [TRACKED_NO_CRATE_HASH],
        /// Base directory containing the `src/` for the Rust standard library, and
        /// potentially `rustc` as well, if we can find it. Right now it's always
        /// `$sysroot/lib/rustlib/src/rust` (i.e. the `rustup` `rust-src` component).
        ///
        /// This directory is what the virtual `/rustc/$hash` is translated back to,
        /// if Rust was built with path remapping to `/rustc/$hash` enabled
        /// (the `rust.remap-debuginfo` option in `config.toml`).
        real_rust_source_base_dir: Option<PathBuf> [TRACKED_NO_CRATE_HASH],

        edition: Edition [TRACKED],

        /// `true` if we're emitting JSON blobs about each artifact produced
        /// by the compiler.
        json_artifact_notifications: bool [TRACKED],

        /// `true` if we're emitting a JSON blob containing the unused externs
        json_unused_externs: JsonUnusedExterns [UNTRACKED],

        /// `true` if we're emitting a JSON job containing a future-incompat report for lints
        json_future_incompat: bool [TRACKED],

        pretty: Option<PpMode> [UNTRACKED],

        /// The (potentially remapped) working directory
        working_dir: RealFileName [TRACKED],
        color: ColorConfig [UNTRACKED],

        verbose: bool [TRACKED_NO_CRATE_HASH],
    }
);

/// Defines all `CodegenOptions`/`DebuggingOptions` fields and parsers all at once. The goal of this
/// macro is to define an interface that can be programmatically used by the option parser
/// to initialize the struct without hardcoding field names all over the place.
///
/// The goal is to invoke this macro once with the correct fields, and then this macro generates all
/// necessary code. The main gotcha of this macro is the `cgsetters` module which is a bunch of
/// generated code to parse an option into its respective field in the struct. There are a few
/// hand-written parsers for parsing specific types of values in this module.
macro_rules! options {
    ($struct_name:ident, $stat:ident, $optmod:ident, $prefix:expr, $outputname:expr,
     $($( #[$attr:meta] )* $opt:ident : $t:ty = (
        $init:expr,
        $parse:ident,
        [$dep_tracking_marker:ident],
        $desc:expr)
     ),* ,) =>
(
    #[derive(Clone)]
    #[rustc_lint_opt_ty]
    pub struct $struct_name { $( $( #[$attr] )* pub $opt: $t),* }

    impl Default for $struct_name {
        fn default() -> $struct_name {
            $struct_name { $($opt: $init),* }
        }
    }

    impl $struct_name {
        pub fn build(
            early_dcx: &EarlyDiagCtxt,
            matches: &getopts::Matches,
        ) -> $struct_name {
            build_options(early_dcx, matches, $stat, $prefix, $outputname)
        }

        fn dep_tracking_hash(&self, for_crate_hash: bool, error_format: ErrorOutputType) -> u64 {
            let mut sub_hashes = BTreeMap::new();
            $({
                hash_opt!($opt,
                            &self.$opt,
                            &mut sub_hashes,
                            for_crate_hash,
                            [$dep_tracking_marker]);
            })*
            let mut hasher = DefaultHasher::new();
            dep_tracking::stable_hash(sub_hashes,
                                        &mut hasher,
                                        error_format,
                                        for_crate_hash
                                        );
            hasher.finish()
        }
    }

    pub const $stat: OptionDescrs<$struct_name> =
        &[ $( (stringify!($opt), $optmod::$opt, desc::$parse, $desc) ),* ];

    mod $optmod {
    $(
        pub(super) fn $opt(cg: &mut super::$struct_name, v: Option<&str>) -> bool {
            super::parse::$parse(&mut redirect_field!(cg.$opt), v)
        }
    )*
    }

) }

impl CodegenOptions {
    // JUSTIFICATION: defn of the suggested wrapper fn
    #[allow(rustc::bad_opt_access)]
    pub fn instrument_coverage(&self) -> InstrumentCoverage {
        self.instrument_coverage
    }
}

// Sometimes different options need to build a common structure.
// That structure can be kept in one of the options' fields, the others become dummy.
macro_rules! redirect_field {
    ($cg:ident.link_arg) => {
        $cg.link_args
    };
    ($cg:ident.pre_link_arg) => {
        $cg.pre_link_args
    };
    ($cg:ident.$field:ident) => {
        $cg.$field
    };
}

type OptionSetter<O> = fn(&mut O, v: Option<&str>) -> bool;
type OptionDescrs<O> = &'static [(&'static str, OptionSetter<O>, &'static str, &'static str)];

#[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
fn build_options<O: Default>(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
    descrs: OptionDescrs<O>,
    prefix: &str,
    outputname: &str,
) -> O {
    let mut op = O::default();
    for option in matches.opt_strs(prefix) {
        let (key, value) = match option.split_once('=') {
            None => (option, None),
            Some((k, v)) => (k.to_string(), Some(v)),
        };

        let option_to_lookup = key.replace('-', "_");
        match descrs.iter().find(|(name, ..)| *name == option_to_lookup) {
            Some((_, setter, type_desc, _)) => {
                if !setter(&mut op, value) {
                    match value {
                        None => early_dcx.early_fatal(
                            format!(
                                "{outputname} option `{key}` requires {type_desc} ({prefix} {key}=<value>)"
                            ),
                        ),
                        Some(value) => early_dcx.early_fatal(
                            format!(
                                "incorrect value `{value}` for {outputname} option `{key}` - {type_desc} was expected"
                            ),
                        ),
                    }
                }
            }
            None => early_dcx.early_fatal(format!("unknown {outputname} option: `{key}`")),
        }
    }
    return op;
}

#[allow(non_upper_case_globals)]
mod desc {
    pub const parse_no_flag: &str = "no value";
    pub const parse_bool: &str = "one of: `y`, `yes`, `on`, `true`, `n`, `no`, `off` or `false`";
    pub const parse_opt_bool: &str = parse_bool;
    pub const parse_string: &str = "a string";
    pub const parse_opt_string: &str = parse_string;
    pub const parse_string_push: &str = parse_string;
    pub const parse_opt_langid: &str = "a language identifier";
    pub const parse_opt_pathbuf: &str = "a path";
    pub const parse_list: &str = "a space-separated list of strings";
    pub const parse_list_with_polarity: &str =
        "a comma-separated list of strings, with elements beginning with + or -";
    pub const parse_comma_list: &str = "a comma-separated list of strings";
    pub const parse_opt_comma_list: &str = parse_comma_list;
    pub const parse_number: &str = "a number";
    pub const parse_opt_number: &str = parse_number;
    pub const parse_threads: &str = parse_number;
    pub const parse_time_passes_format: &str = "`text` (default) or `json`";
    pub const parse_passes: &str = "a space-separated list of passes, or `all`";
    pub const parse_panic_strategy: &str = "either `unwind` or `abort`";
    pub const parse_opt_panic_strategy: &str = parse_panic_strategy;
    pub const parse_oom_strategy: &str = "either `panic` or `abort`";
    pub const parse_relro_level: &str = "one of: `full`, `partial`, or `off`";
    pub const parse_sanitizers: &str = "comma separated list of sanitizers: `address`, `cfi`, `dataflow`, `hwaddress`, `kcfi`, `kernel-address`, `leak`, `memory`, `memtag`, `safestack`, `shadow-call-stack`, or `thread`";
    pub const parse_sanitizer_memory_track_origins: &str = "0, 1, or 2";
    pub const parse_cfguard: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), `checks`, or `nochecks`";
    pub const parse_cfprotection: &str = "`none`|`no`|`n` (default), `branch`, `return`, or `full`|`yes`|`y` (equivalent to `branch` and `return`)";
    pub const parse_debuginfo: &str = "either an integer (0, 1, 2), `none`, `line-directives-only`, `line-tables-only`, `limited`, or `full`";
    pub const parse_debuginfo_compression: &str = "one of `none`, `zlib`, or `zstd`";
    pub const parse_collapse_macro_debuginfo: &str = "one of `no`, `external`, or `yes`";
    pub const parse_strip: &str = "either `none`, `debuginfo`, or `symbols`";
    pub const parse_linker_flavor: &str = ::rustc_target::spec::LinkerFlavorCli::one_of();
    pub const parse_optimization_fuel: &str = "crate=integer";
    pub const parse_dump_mono_stats: &str = "`markdown` (default) or `json`";
    pub const parse_instrument_coverage: &str = parse_bool;
    pub const parse_coverage_options: &str = "`branch` or `no-branch`";
    pub const parse_instrument_xray: &str = "either a boolean (`yes`, `no`, `on`, `off`, etc), or a comma separated list of settings: `always` or `never` (mutually exclusive), `ignore-loops`, `instruction-threshold=N`, `skip-entry`, `skip-exit`";
    pub const parse_unpretty: &str = "`string` or `string=string`";
    pub const parse_treat_err_as_bug: &str = "either no value or a non-negative number";
    pub const parse_next_solver_config: &str = "a comma separated list of solver configurations: `globally` (default), `coherence`, `dump-tree`, `dump-tree-on-error";
    pub const parse_lto: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), `thin`, `fat`, or omitted";
    pub const parse_linker_plugin_lto: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), or the path to the linker plugin";
    pub const parse_location_detail: &str = "either `none`, or a comma separated list of location details to track: `file`, `line`, or `column`";
    pub const parse_switch_with_opt_path: &str =
        "an optional path to the profiling data output directory";
    pub const parse_merge_functions: &str = "one of: `disabled`, `trampolines`, or `aliases`";
    pub const parse_symbol_mangling_version: &str =
        "one of: `legacy`, `v0` (RFC 2603), or `hashed`";
    pub const parse_src_file_hash: &str = "either `md5` or `sha1`";
    pub const parse_relocation_model: &str =
        "one of supported relocation models (`rustc --print relocation-models`)";
    pub const parse_code_model: &str = "one of supported code models (`rustc --print code-models`)";
    pub const parse_tls_model: &str = "one of supported TLS models (`rustc --print tls-models`)";
    pub const parse_target_feature: &str = parse_string;
    pub const parse_terminal_url: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), or `auto`";
    pub const parse_wasi_exec_model: &str = "either `command` or `reactor`";
    pub const parse_split_debuginfo: &str =
        "one of supported split-debuginfo modes (`off`, `packed`, or `unpacked`)";
    pub const parse_split_dwarf_kind: &str =
        "one of supported split dwarf modes (`split` or `single`)";
    pub const parse_link_self_contained: &str = "one of: `y`, `yes`, `on`, `n`, `no`, `off`, or a list of enabled (`+` prefix) and disabled (`-` prefix) \
        components: `crto`, `libc`, `unwind`, `linker`, `sanitizers`, `mingw`";
    pub const parse_polonius: &str = "either no value or `legacy` (the default), or `next`";
    pub const parse_stack_protector: &str =
        "one of (`none` (default), `basic`, `strong`, or `all`)";
    pub const parse_branch_protection: &str =
        "a `,` separated combination of `bti`, `b-key`, `pac-ret`, or `leaf`";
    pub const parse_proc_macro_execution_strategy: &str =
        "one of supported execution strategies (`same-thread`, or `cross-thread`)";
    pub const parse_remap_path_scope: &str = "comma separated list of scopes: `macro`, `diagnostics`, `unsplit-debuginfo`, `split-debuginfo`, `split-debuginfo-path`, `object`, `all`";
    pub const parse_inlining_threshold: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), or a non-negative number";
    pub const parse_llvm_module_flag: &str = "<key>:<type>:<value>:<behavior>. Type must currently be `u32`. Behavior should be one of (`error`, `warning`, `require`, `override`, `append`, `appendunique`, `max`, `min`)";
    pub const parse_function_return: &str = "`keep` or `thunk-extern`";
}

mod parse {
    pub(crate) use super::*;
    use std::str::FromStr;

    /// This is for boolean options that don't take a value and start with
    /// `no-`. This style of option is deprecated.
    pub(crate) fn parse_no_flag(slot: &mut bool, v: Option<&str>) -> bool {
        match v {
            None => {
                *slot = true;
                true
            }
            Some(_) => false,
        }
    }

    /// Use this for any boolean option that has a static default.
    pub(crate) fn parse_bool(slot: &mut bool, v: Option<&str>) -> bool {
        match v {
            Some("y") | Some("yes") | Some("on") | Some("true") | None => {
                *slot = true;
                true
            }
            Some("n") | Some("no") | Some("off") | Some("false") => {
                *slot = false;
                true
            }
            _ => false,
        }
    }

    /// Use this for any boolean option that lacks a static default. (The
    /// actions taken when such an option is not specified will depend on
    /// other factors, such as other options, or target options.)
    pub(crate) fn parse_opt_bool(slot: &mut Option<bool>, v: Option<&str>) -> bool {
        match v {
            Some("y") | Some("yes") | Some("on") | Some("true") | None => {
                *slot = Some(true);
                true
            }
            Some("n") | Some("no") | Some("off") | Some("false") => {
                *slot = Some(false);
                true
            }
            _ => false,
        }
    }

    /// Parses whether polonius is enabled, and if so, which version.
    pub(crate) fn parse_polonius(slot: &mut Polonius, v: Option<&str>) -> bool {
        match v {
            Some("legacy") | None => {
                *slot = Polonius::Legacy;
                true
            }
            Some("next") => {
                *slot = Polonius::Next;
                true
            }
            _ => false,
        }
    }

    /// Use this for any string option that has a static default.
    pub(crate) fn parse_string(slot: &mut String, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                *slot = s.to_string();
                true
            }
            None => false,
        }
    }

    /// Use this for any string option that lacks a static default.
    pub(crate) fn parse_opt_string(slot: &mut Option<String>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                *slot = Some(s.to_string());
                true
            }
            None => false,
        }
    }

    /// Parse an optional language identifier, e.g. `en-US` or `zh-CN`.
    pub(crate) fn parse_opt_langid(slot: &mut Option<LanguageIdentifier>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                *slot = rustc_errors::LanguageIdentifier::from_str(s).ok();
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_opt_pathbuf(slot: &mut Option<PathBuf>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                *slot = Some(PathBuf::from(s));
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_string_push(slot: &mut Vec<String>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                slot.push(s.to_string());
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_list(slot: &mut Vec<String>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                slot.extend(s.split_whitespace().map(|s| s.to_string()));
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_list_with_polarity(
        slot: &mut Vec<(String, bool)>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some(s) => {
                for s in s.split(',') {
                    let Some(pass_name) = s.strip_prefix(&['+', '-'][..]) else { return false };
                    slot.push((pass_name.to_string(), &s[..1] == "+"));
                }
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_location_detail(ld: &mut LocationDetail, v: Option<&str>) -> bool {
        if let Some(v) = v {
            ld.line = false;
            ld.file = false;
            ld.column = false;
            if v == "none" {
                return true;
            }
            for s in v.split(',') {
                match s {
                    "file" => ld.file = true,
                    "line" => ld.line = true,
                    "column" => ld.column = true,
                    _ => return false,
                }
            }
            true
        } else {
            false
        }
    }

    pub(crate) fn parse_comma_list(slot: &mut Vec<String>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                let mut v: Vec<_> = s.split(',').map(|s| s.to_string()).collect();
                v.sort_unstable();
                *slot = v;
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_opt_comma_list(slot: &mut Option<Vec<String>>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                let mut v: Vec<_> = s.split(',').map(|s| s.to_string()).collect();
                v.sort_unstable();
                *slot = Some(v);
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_threads(slot: &mut usize, v: Option<&str>) -> bool {
        match v.and_then(|s| s.parse().ok()) {
            Some(0) => {
                *slot = std::thread::available_parallelism().map_or(1, NonZero::<usize>::get);
                true
            }
            Some(i) => {
                *slot = i;
                true
            }
            None => false,
        }
    }

    /// Use this for any numeric option that has a static default.
    pub(crate) fn parse_number<T: Copy + FromStr>(slot: &mut T, v: Option<&str>) -> bool {
        match v.and_then(|s| s.parse().ok()) {
            Some(i) => {
                *slot = i;
                true
            }
            None => false,
        }
    }

    /// Use this for any numeric option that lacks a static default.
    pub(crate) fn parse_opt_number<T: Copy + FromStr>(
        slot: &mut Option<T>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some(s) => {
                *slot = s.parse().ok();
                slot.is_some()
            }
            None => false,
        }
    }

    pub(crate) fn parse_passes(slot: &mut Passes, v: Option<&str>) -> bool {
        match v {
            Some("all") => {
                *slot = Passes::All;
                true
            }
            v => {
                let mut passes = vec![];
                if parse_list(&mut passes, v) {
                    slot.extend(passes);
                    true
                } else {
                    false
                }
            }
        }
    }

    pub(crate) fn parse_opt_panic_strategy(
        slot: &mut Option<PanicStrategy>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some("unwind") => *slot = Some(PanicStrategy::Unwind),
            Some("abort") => *slot = Some(PanicStrategy::Abort),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_panic_strategy(slot: &mut PanicStrategy, v: Option<&str>) -> bool {
        match v {
            Some("unwind") => *slot = PanicStrategy::Unwind,
            Some("abort") => *slot = PanicStrategy::Abort,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_oom_strategy(slot: &mut OomStrategy, v: Option<&str>) -> bool {
        match v {
            Some("panic") => *slot = OomStrategy::Panic,
            Some("abort") => *slot = OomStrategy::Abort,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_relro_level(slot: &mut Option<RelroLevel>, v: Option<&str>) -> bool {
        match v {
            Some(s) => match s.parse::<RelroLevel>() {
                Ok(level) => *slot = Some(level),
                _ => return false,
            },
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_sanitizers(slot: &mut SanitizerSet, v: Option<&str>) -> bool {
        if let Some(v) = v {
            for s in v.split(',') {
                *slot |= match s {
                    "address" => SanitizerSet::ADDRESS,
                    "cfi" => SanitizerSet::CFI,
                    "dataflow" => SanitizerSet::DATAFLOW,
                    "kcfi" => SanitizerSet::KCFI,
                    "kernel-address" => SanitizerSet::KERNELADDRESS,
                    "leak" => SanitizerSet::LEAK,
                    "memory" => SanitizerSet::MEMORY,
                    "memtag" => SanitizerSet::MEMTAG,
                    "shadow-call-stack" => SanitizerSet::SHADOWCALLSTACK,
                    "thread" => SanitizerSet::THREAD,
                    "hwaddress" => SanitizerSet::HWADDRESS,
                    "safestack" => SanitizerSet::SAFESTACK,
                    _ => return false,
                }
            }
            true
        } else {
            false
        }
    }

    pub(crate) fn parse_sanitizer_memory_track_origins(slot: &mut usize, v: Option<&str>) -> bool {
        match v {
            Some("2") | None => {
                *slot = 2;
                true
            }
            Some("1") => {
                *slot = 1;
                true
            }
            Some("0") => {
                *slot = 0;
                true
            }
            Some(_) => false,
        }
    }

    pub(crate) fn parse_strip(slot: &mut Strip, v: Option<&str>) -> bool {
        match v {
            Some("none") => *slot = Strip::None,
            Some("debuginfo") => *slot = Strip::Debuginfo,
            Some("symbols") => *slot = Strip::Symbols,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_cfguard(slot: &mut CFGuard, v: Option<&str>) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() { CFGuard::Checks } else { CFGuard::Disabled };
                return true;
            }
        }

        *slot = match v {
            None => CFGuard::Checks,
            Some("checks") => CFGuard::Checks,
            Some("nochecks") => CFGuard::NoChecks,
            Some(_) => return false,
        };
        true
    }

    pub(crate) fn parse_cfprotection(slot: &mut CFProtection, v: Option<&str>) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() { CFProtection::Full } else { CFProtection::None };
                return true;
            }
        }

        *slot = match v {
            None | Some("none") => CFProtection::None,
            Some("branch") => CFProtection::Branch,
            Some("return") => CFProtection::Return,
            Some("full") => CFProtection::Full,
            Some(_) => return false,
        };
        true
    }

    pub(crate) fn parse_debuginfo(slot: &mut DebugInfo, v: Option<&str>) -> bool {
        match v {
            Some("0") | Some("none") => *slot = DebugInfo::None,
            Some("line-directives-only") => *slot = DebugInfo::LineDirectivesOnly,
            Some("line-tables-only") => *slot = DebugInfo::LineTablesOnly,
            Some("1") | Some("limited") => *slot = DebugInfo::Limited,
            Some("2") | Some("full") => *slot = DebugInfo::Full,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_debuginfo_compression(
        slot: &mut DebugInfoCompression,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some("none") => *slot = DebugInfoCompression::None,
            Some("zlib") => *slot = DebugInfoCompression::Zlib,
            Some("zstd") => *slot = DebugInfoCompression::Zstd,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_linker_flavor(slot: &mut Option<LinkerFlavorCli>, v: Option<&str>) -> bool {
        match v.and_then(LinkerFlavorCli::from_str) {
            Some(lf) => *slot = Some(lf),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_optimization_fuel(
        slot: &mut Option<(String, u64)>,
        v: Option<&str>,
    ) -> bool {
        match v {
            None => false,
            Some(s) => {
                let parts = s.split('=').collect::<Vec<_>>();
                if parts.len() != 2 {
                    return false;
                }
                let crate_name = parts[0].to_string();
                let fuel = parts[1].parse::<u64>();
                if fuel.is_err() {
                    return false;
                }
                *slot = Some((crate_name, fuel.unwrap()));
                true
            }
        }
    }

    pub(crate) fn parse_unpretty(slot: &mut Option<String>, v: Option<&str>) -> bool {
        match v {
            None => false,
            Some(s) if s.split('=').count() <= 2 => {
                *slot = Some(s.to_string());
                true
            }
            _ => false,
        }
    }

    pub(crate) fn parse_time_passes_format(slot: &mut TimePassesFormat, v: Option<&str>) -> bool {
        match v {
            None => true,
            Some("json") => {
                *slot = TimePassesFormat::Json;
                true
            }
            Some("text") => {
                *slot = TimePassesFormat::Text;
                true
            }
            Some(_) => false,
        }
    }

    pub(crate) fn parse_dump_mono_stats(slot: &mut DumpMonoStatsFormat, v: Option<&str>) -> bool {
        match v {
            None => true,
            Some("json") => {
                *slot = DumpMonoStatsFormat::Json;
                true
            }
            Some("markdown") => {
                *slot = DumpMonoStatsFormat::Markdown;
                true
            }
            Some(_) => false,
        }
    }

    pub(crate) fn parse_instrument_coverage(
        slot: &mut InstrumentCoverage,
        v: Option<&str>,
    ) -> bool {
        if v.is_some() {
            let mut bool_arg = false;
            if parse_bool(&mut bool_arg, v) {
                *slot = if bool_arg { InstrumentCoverage::Yes } else { InstrumentCoverage::No };
                return true;
            }
        }

        let Some(v) = v else {
            *slot = InstrumentCoverage::Yes;
            return true;
        };

        // Parse values that have historically been accepted by stable compilers,
        // even though they're currently just aliases for boolean values.
        *slot = match v {
            "all" => InstrumentCoverage::Yes,
            "0" => InstrumentCoverage::No,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_coverage_options(slot: &mut CoverageOptions, v: Option<&str>) -> bool {
        let Some(v) = v else { return true };

        for option in v.split(',') {
            let (option, enabled) = match option.strip_prefix("no-") {
                Some(without_no) => (without_no, false),
                None => (option, true),
            };
            let slot = match option {
                "branch" => &mut slot.branch,
                _ => return false,
            };
            *slot = enabled;
        }

        true
    }

    pub(crate) fn parse_instrument_xray(
        slot: &mut Option<InstrumentXRay>,
        v: Option<&str>,
    ) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() { Some(InstrumentXRay::default()) } else { None };
                return true;
            }
        }

        let options = slot.get_or_insert_default();
        let mut seen_always = false;
        let mut seen_never = false;
        let mut seen_ignore_loops = false;
        let mut seen_instruction_threshold = false;
        let mut seen_skip_entry = false;
        let mut seen_skip_exit = false;
        for option in v.into_iter().flat_map(|v| v.split(',')) {
            match option {
                "always" if !seen_always && !seen_never => {
                    options.always = true;
                    options.never = false;
                    seen_always = true;
                }
                "never" if !seen_never && !seen_always => {
                    options.never = true;
                    options.always = false;
                    seen_never = true;
                }
                "ignore-loops" if !seen_ignore_loops => {
                    options.ignore_loops = true;
                    seen_ignore_loops = true;
                }
                option
                    if option.starts_with("instruction-threshold")
                        && !seen_instruction_threshold =>
                {
                    let Some(("instruction-threshold", n)) = option.split_once('=') else {
                        return false;
                    };
                    match n.parse() {
                        Ok(n) => options.instruction_threshold = Some(n),
                        Err(_) => return false,
                    }
                    seen_instruction_threshold = true;
                }
                "skip-entry" if !seen_skip_entry => {
                    options.skip_entry = true;
                    seen_skip_entry = true;
                }
                "skip-exit" if !seen_skip_exit => {
                    options.skip_exit = true;
                    seen_skip_exit = true;
                }
                _ => return false,
            }
        }
        true
    }

    pub(crate) fn parse_treat_err_as_bug(
        slot: &mut Option<NonZero<usize>>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some(s) => match s.parse() {
                Ok(val) => {
                    *slot = Some(val);
                    true
                }
                Err(e) => {
                    *slot = None;
                    e.kind() == &IntErrorKind::Zero
                }
            },
            None => {
                *slot = NonZero::new(1);
                true
            }
        }
    }

    pub(crate) fn parse_next_solver_config(
        slot: &mut Option<NextSolverConfig>,
        v: Option<&str>,
    ) -> bool {
        if let Some(config) = v {
            let mut coherence = false;
            let mut globally = true;
            let mut dump_tree = None;
            for c in config.split(',') {
                match c {
                    "globally" => globally = true,
                    "coherence" => {
                        globally = false;
                        coherence = true;
                    }
                    "dump-tree" => {
                        if dump_tree.replace(DumpSolverProofTree::Always).is_some() {
                            return false;
                        }
                    }
                    "dump-tree-on-error" => {
                        if dump_tree.replace(DumpSolverProofTree::OnError).is_some() {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }

            *slot = Some(NextSolverConfig {
                coherence: coherence || globally,
                globally,
                dump_tree: dump_tree.unwrap_or_default(),
            });
        } else {
            *slot = Some(NextSolverConfig {
                coherence: true,
                globally: true,
                dump_tree: Default::default(),
            });
        }

        true
    }

    pub(crate) fn parse_lto(slot: &mut LtoCli, v: Option<&str>) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() { LtoCli::Yes } else { LtoCli::No };
                return true;
            }
        }

        *slot = match v {
            None => LtoCli::NoParam,
            Some("thin") => LtoCli::Thin,
            Some("fat") => LtoCli::Fat,
            Some(_) => return false,
        };
        true
    }

    pub(crate) fn parse_linker_plugin_lto(slot: &mut LinkerPluginLto, v: Option<&str>) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() {
                    LinkerPluginLto::LinkerPluginAuto
                } else {
                    LinkerPluginLto::Disabled
                };
                return true;
            }
        }

        *slot = match v {
            None => LinkerPluginLto::LinkerPluginAuto,
            Some(path) => LinkerPluginLto::LinkerPlugin(PathBuf::from(path)),
        };
        true
    }

    pub(crate) fn parse_switch_with_opt_path(
        slot: &mut SwitchWithOptPath,
        v: Option<&str>,
    ) -> bool {
        *slot = match v {
            None => SwitchWithOptPath::Enabled(None),
            Some(path) => SwitchWithOptPath::Enabled(Some(PathBuf::from(path))),
        };
        true
    }

    pub(crate) fn parse_merge_functions(
        slot: &mut Option<MergeFunctions>,
        v: Option<&str>,
    ) -> bool {
        match v.and_then(|s| MergeFunctions::from_str(s).ok()) {
            Some(mergefunc) => *slot = Some(mergefunc),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_remap_path_scope(
        slot: &mut RemapPathScopeComponents,
        v: Option<&str>,
    ) -> bool {
        if let Some(v) = v {
            *slot = RemapPathScopeComponents::empty();
            for s in v.split(',') {
                *slot |= match s {
                    "macro" => RemapPathScopeComponents::MACRO,
                    "diagnostics" => RemapPathScopeComponents::DIAGNOSTICS,
                    "unsplit-debuginfo" => RemapPathScopeComponents::UNSPLIT_DEBUGINFO,
                    "split-debuginfo" => RemapPathScopeComponents::SPLIT_DEBUGINFO,
                    "split-debuginfo-path" => RemapPathScopeComponents::SPLIT_DEBUGINFO_PATH,
                    "object" => RemapPathScopeComponents::OBJECT,
                    "all" => RemapPathScopeComponents::all(),
                    _ => return false,
                }
            }
            true
        } else {
            false
        }
    }

    pub(crate) fn parse_relocation_model(slot: &mut Option<RelocModel>, v: Option<&str>) -> bool {
        match v.and_then(|s| RelocModel::from_str(s).ok()) {
            Some(relocation_model) => *slot = Some(relocation_model),
            None if v == Some("default") => *slot = None,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_code_model(slot: &mut Option<CodeModel>, v: Option<&str>) -> bool {
        match v.and_then(|s| CodeModel::from_str(s).ok()) {
            Some(code_model) => *slot = Some(code_model),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_tls_model(slot: &mut Option<TlsModel>, v: Option<&str>) -> bool {
        match v.and_then(|s| TlsModel::from_str(s).ok()) {
            Some(tls_model) => *slot = Some(tls_model),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_terminal_url(slot: &mut TerminalUrl, v: Option<&str>) -> bool {
        *slot = match v {
            Some("on" | "" | "yes" | "y") | None => TerminalUrl::Yes,
            Some("off" | "no" | "n") => TerminalUrl::No,
            Some("auto") => TerminalUrl::Auto,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_symbol_mangling_version(
        slot: &mut Option<SymbolManglingVersion>,
        v: Option<&str>,
    ) -> bool {
        *slot = match v {
            Some("legacy") => Some(SymbolManglingVersion::Legacy),
            Some("v0") => Some(SymbolManglingVersion::V0),
            Some("hashed") => Some(SymbolManglingVersion::Hashed),
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_src_file_hash(
        slot: &mut Option<SourceFileHashAlgorithm>,
        v: Option<&str>,
    ) -> bool {
        match v.and_then(|s| SourceFileHashAlgorithm::from_str(s).ok()) {
            Some(hash_kind) => *slot = Some(hash_kind),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_target_feature(slot: &mut String, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                if !slot.is_empty() {
                    slot.push(',');
                }
                slot.push_str(s);
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_link_self_contained(slot: &mut LinkSelfContained, v: Option<&str>) -> bool {
        // Whenever `-C link-self-contained` is passed without a value, it's an opt-in
        // just like `parse_opt_bool`, the historical value of this flag.
        //
        // 1. Parse historical single bool values
        let s = v.unwrap_or("y");
        match s {
            "y" | "yes" | "on" => {
                slot.set_all_explicitly(true);
                return true;
            }
            "n" | "no" | "off" => {
                slot.set_all_explicitly(false);
                return true;
            }
            _ => {}
        }

        // 2. Parse a list of enabled and disabled components.
        for comp in s.split(',') {
            if slot.handle_cli_component(comp).is_none() {
                return false;
            }
        }

        true
    }

    pub(crate) fn parse_wasi_exec_model(slot: &mut Option<WasiExecModel>, v: Option<&str>) -> bool {
        match v {
            Some("command") => *slot = Some(WasiExecModel::Command),
            Some("reactor") => *slot = Some(WasiExecModel::Reactor),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_split_debuginfo(
        slot: &mut Option<SplitDebuginfo>,
        v: Option<&str>,
    ) -> bool {
        match v.and_then(|s| SplitDebuginfo::from_str(s).ok()) {
            Some(e) => *slot = Some(e),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_split_dwarf_kind(slot: &mut SplitDwarfKind, v: Option<&str>) -> bool {
        match v.and_then(|s| SplitDwarfKind::from_str(s).ok()) {
            Some(e) => *slot = e,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_stack_protector(slot: &mut StackProtector, v: Option<&str>) -> bool {
        match v.and_then(|s| StackProtector::from_str(s).ok()) {
            Some(ssp) => *slot = ssp,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_branch_protection(
        slot: &mut Option<BranchProtection>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some(s) => {
                let slot = slot.get_or_insert_default();
                for opt in s.split(',') {
                    match opt {
                        "bti" => slot.bti = true,
                        "pac-ret" if slot.pac_ret.is_none() => {
                            slot.pac_ret = Some(PacRet { leaf: false, key: PAuthKey::A })
                        }
                        "leaf" => match slot.pac_ret.as_mut() {
                            Some(pac) => pac.leaf = true,
                            _ => return false,
                        },
                        "b-key" => match slot.pac_ret.as_mut() {
                            Some(pac) => pac.key = PAuthKey::B,
                            _ => return false,
                        },
                        _ => return false,
                    };
                }
            }
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_collapse_macro_debuginfo(
        slot: &mut CollapseMacroDebuginfo,
        v: Option<&str>,
    ) -> bool {
        *slot = match v {
            Some("no") => CollapseMacroDebuginfo::No,
            Some("external") => CollapseMacroDebuginfo::External,
            Some("yes") => CollapseMacroDebuginfo::Yes,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_proc_macro_execution_strategy(
        slot: &mut ProcMacroExecutionStrategy,
        v: Option<&str>,
    ) -> bool {
        *slot = match v {
            Some("same-thread") => ProcMacroExecutionStrategy::SameThread,
            Some("cross-thread") => ProcMacroExecutionStrategy::CrossThread,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_inlining_threshold(slot: &mut InliningThreshold, v: Option<&str>) -> bool {
        match v {
            Some("always" | "yes") => {
                *slot = InliningThreshold::Always;
            }
            Some("never") => {
                *slot = InliningThreshold::Never;
            }
            Some(v) => {
                if let Ok(threshold) = v.parse() {
                    *slot = InliningThreshold::Sometimes(threshold);
                } else {
                    return false;
                }
            }
            None => return false,
        }
        true
    }

    pub(crate) fn parse_llvm_module_flag(
        slot: &mut Vec<(String, u32, String)>,
        v: Option<&str>,
    ) -> bool {
        let elements = v.unwrap_or_default().split(':').collect::<Vec<_>>();
        let [key, md_type, value, behavior] = elements.as_slice() else {
            return false;
        };
        if *md_type != "u32" {
            // Currently we only support u32 metadata flags, but require the
            // type for forward-compatibility.
            return false;
        }
        let Ok(value) = value.parse::<u32>() else {
            return false;
        };
        let behavior = behavior.to_lowercase();
        let all_behaviors =
            ["error", "warning", "require", "override", "append", "appendunique", "max", "min"];
        if !all_behaviors.contains(&behavior.as_str()) {
            return false;
        }

        slot.push((key.to_string(), value, behavior));
        true
    }

    pub(crate) fn parse_function_return(slot: &mut FunctionReturn, v: Option<&str>) -> bool {
        match v {
            Some("keep") => *slot = FunctionReturn::Keep,
            Some("thunk-extern") => *slot = FunctionReturn::ThunkExtern,
            _ => return false,
        }
        true
    }
}

options! {
    CodegenOptions, CG_OPTIONS, cgopts, "C", "codegen",

    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/rustc/src/codegen-options/index.md

    // tidy-alphabetical-start
    ar: String = (String::new(), parse_string, [UNTRACKED],
        "this option is deprecated and does nothing"),
    #[rustc_lint_opt_deny_field_access("use `Session::code_model` instead of this field")]
    code_model: Option<CodeModel> = (None, parse_code_model, [TRACKED],
        "choose the code model to use (`rustc --print code-models` for details)"),
    codegen_units: Option<usize> = (None, parse_opt_number, [UNTRACKED],
        "divide crate into N units to optimize in parallel"),
    control_flow_guard: CFGuard = (CFGuard::Disabled, parse_cfguard, [TRACKED],
        "use Windows Control Flow Guard (default: no)"),
    debug_assertions: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "explicitly enable the `cfg(debug_assertions)` directive"),
    debuginfo: DebugInfo = (DebugInfo::None, parse_debuginfo, [TRACKED],
        "debug info emission level (0-2, none, line-directives-only, \
        line-tables-only, limited, or full; default: 0)"),
    default_linker_libraries: bool = (false, parse_bool, [UNTRACKED],
        "allow the linker to link its default libraries (default: no)"),
    dlltool: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "import library generation tool (ignored except when targeting windows-gnu)"),
    embed_bitcode: bool = (true, parse_bool, [TRACKED],
        "emit bitcode in rlibs (default: yes)"),
    extra_filename: String = (String::new(), parse_string, [UNTRACKED],
        "extra data to put in each output filename"),
    force_frame_pointers: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "force use of the frame pointers"),
    #[rustc_lint_opt_deny_field_access("use `Session::must_emit_unwind_tables` instead of this field")]
    force_unwind_tables: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "force use of unwind tables"),
    incremental: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "enable incremental compilation"),
    inline_threshold: Option<u32> = (None, parse_opt_number, [TRACKED],
        "set the threshold for inlining a function"),
    #[rustc_lint_opt_deny_field_access("use `Session::instrument_coverage` instead of this field")]
    instrument_coverage: InstrumentCoverage = (InstrumentCoverage::No, parse_instrument_coverage, [TRACKED],
        "instrument the generated code to support LLVM source-based code coverage reports \
        (note, the compiler build config must include `profiler = true`); \
        implies `-C symbol-mangling-version=v0`"),
    link_arg: (/* redirected to link_args */) = ((), parse_string_push, [UNTRACKED],
        "a single extra argument to append to the linker invocation (can be used several times)"),
    link_args: Vec<String> = (Vec::new(), parse_list, [UNTRACKED],
        "extra arguments to append to the linker invocation (space separated)"),
    #[rustc_lint_opt_deny_field_access("use `Session::link_dead_code` instead of this field")]
    link_dead_code: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "keep dead code at link time (useful for code coverage) (default: no)"),
    link_self_contained: LinkSelfContained = (LinkSelfContained::default(), parse_link_self_contained, [UNTRACKED],
        "control whether to link Rust provided C objects/libraries or rely
        on a C toolchain or linker installed in the system"),
    linker: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "system linker to link outputs with"),
    linker_flavor: Option<LinkerFlavorCli> = (None, parse_linker_flavor, [UNTRACKED],
        "linker flavor"),
    linker_plugin_lto: LinkerPluginLto = (LinkerPluginLto::Disabled,
        parse_linker_plugin_lto, [TRACKED],
        "generate build artifacts that are compatible with linker-based LTO"),
    llvm_args: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of arguments to pass to LLVM (space separated)"),
    #[rustc_lint_opt_deny_field_access("use `Session::lto` instead of this field")]
    lto: LtoCli = (LtoCli::Unspecified, parse_lto, [TRACKED],
        "perform LLVM link-time optimizations"),
    metadata: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "metadata to mangle symbol names with"),
    no_prepopulate_passes: bool = (false, parse_no_flag, [TRACKED],
        "give an empty list of passes to the pass manager"),
    no_redzone: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "disable the use of the redzone"),
    no_stack_check: bool = (false, parse_no_flag, [UNTRACKED],
        "this option is deprecated and does nothing"),
    no_vectorize_loops: bool = (false, parse_no_flag, [TRACKED],
        "disable loop vectorization optimization passes"),
    no_vectorize_slp: bool = (false, parse_no_flag, [TRACKED],
        "disable LLVM's SLP vectorization pass"),
    opt_level: String = ("0".to_string(), parse_string, [TRACKED],
        "optimization level (0-3, s, or z; default: 0)"),
    #[rustc_lint_opt_deny_field_access("use `Session::overflow_checks` instead of this field")]
    overflow_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "use overflow checks for integer arithmetic"),
    #[rustc_lint_opt_deny_field_access("use `Session::panic_strategy` instead of this field")]
    panic: Option<PanicStrategy> = (None, parse_opt_panic_strategy, [TRACKED],
        "panic strategy to compile crate with"),
    passes: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of extra LLVM passes to run (space separated)"),
    prefer_dynamic: bool = (false, parse_bool, [TRACKED],
        "prefer dynamic linking to static linking (default: no)"),
    profile_generate: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [TRACKED],
        "compile the program with profiling instrumentation"),
    profile_use: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "use the given `.profdata` file for profile-guided optimization"),
    #[rustc_lint_opt_deny_field_access("use `Session::relocation_model` instead of this field")]
    relocation_model: Option<RelocModel> = (None, parse_relocation_model, [TRACKED],
        "control generation of position-independent code (PIC) \
        (`rustc --print relocation-models` for details)"),
    remark: Passes = (Passes::Some(Vec::new()), parse_passes, [UNTRACKED],
        "output remarks for these optimization passes (space separated, or \"all\")"),
    rpath: bool = (false, parse_bool, [UNTRACKED],
        "set rpath values in libs/exes (default: no)"),
    save_temps: bool = (false, parse_bool, [UNTRACKED],
        "save all temporary output files during compilation (default: no)"),
    soft_float: bool = (false, parse_bool, [TRACKED],
        "use soft float ABI (*eabihf targets only) (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::split_debuginfo` instead of this field")]
    split_debuginfo: Option<SplitDebuginfo> = (None, parse_split_debuginfo, [TRACKED],
        "how to handle split-debuginfo, a platform-specific option"),
    strip: Strip = (Strip::None, parse_strip, [UNTRACKED],
        "tell the linker which information to strip (`none` (default), `debuginfo` or `symbols`)"),
    symbol_mangling_version: Option<SymbolManglingVersion> = (None,
        parse_symbol_mangling_version, [TRACKED],
        "which mangling version to use for symbol names ('legacy' (default), 'v0', or 'hashed')"),
    target_cpu: Option<String> = (None, parse_opt_string, [TRACKED],
        "select target processor (`rustc --print target-cpus` for details)"),
    target_feature: String = (String::new(), parse_target_feature, [TRACKED],
        "target specific attributes. (`rustc --print target-features` for details). \
        This feature is unsafe."),
    // tidy-alphabetical-end

    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/rustc/src/codegen-options/index.md
}

options! {
    UnstableOptions, Z_OPTIONS, dbopts, "Z", "unstable",

    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/unstable-book/src/compiler-flags

    // tidy-alphabetical-start
    allow_features: Option<Vec<String>> = (None, parse_opt_comma_list, [TRACKED],
        "only allow the listed language features to be enabled in code (comma separated)"),
    always_encode_mir: bool = (false, parse_bool, [TRACKED],
        "encode MIR of all functions into the crate metadata (default: no)"),
    asm_comments: bool = (false, parse_bool, [TRACKED],
        "generate comments into the assembly (may change behavior) (default: no)"),
    assert_incr_state: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "assert that the incremental cache is in given state: \
         either `loaded` or `not-loaded`."),
    assume_incomplete_release: bool = (false, parse_bool, [TRACKED],
        "make cfg(version) treat the current version as incomplete (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::binary_dep_depinfo` instead of this field")]
    binary_dep_depinfo: bool = (false, parse_bool, [TRACKED],
        "include artifacts (sysroot, crate dependencies) used during compilation in dep-info \
        (default: no)"),
    box_noalias: bool = (true, parse_bool, [TRACKED],
        "emit noalias metadata for box (default: yes)"),
    branch_protection: Option<BranchProtection> = (None, parse_branch_protection, [TRACKED],
        "set options for branch target identification and pointer authentication on AArch64"),
    cf_protection: CFProtection = (CFProtection::None, parse_cfprotection, [TRACKED],
        "instrument control-flow architecture protection"),
    check_cfg_all_expected: bool = (false, parse_bool, [UNTRACKED],
        "show all expected values in check-cfg diagnostics (default: no)"),
    codegen_backend: Option<String> = (None, parse_opt_string, [TRACKED],
        "the backend to use"),
    collapse_macro_debuginfo: CollapseMacroDebuginfo = (CollapseMacroDebuginfo::Unspecified,
        parse_collapse_macro_debuginfo, [TRACKED],
        "set option to collapse debuginfo for macros"),
    combine_cgu: bool = (false, parse_bool, [TRACKED],
        "combine CGUs into a single one"),
    coverage_options: CoverageOptions = (CoverageOptions::default(), parse_coverage_options, [TRACKED],
        "control details of coverage instrumentation"),
    crate_attr: Vec<String> = (Vec::new(), parse_string_push, [TRACKED],
        "inject the given attribute in the crate"),
    cross_crate_inline_threshold: InliningThreshold = (InliningThreshold::Sometimes(100), parse_inlining_threshold, [TRACKED],
        "threshold to allow cross crate inlining of functions"),
    debug_info_for_profiling: bool = (false, parse_bool, [TRACKED],
        "emit discriminators and other data necessary for AutoFDO"),
    debug_macros: bool = (false, parse_bool, [TRACKED],
        "emit line numbers debug info inside macros (default: no)"),
    debuginfo_compression: DebugInfoCompression = (DebugInfoCompression::None, parse_debuginfo_compression, [TRACKED],
        "compress debug info sections (none, zlib, zstd, default: none)"),
    deduplicate_diagnostics: bool = (true, parse_bool, [UNTRACKED],
        "deduplicate identical diagnostics (default: yes)"),
    default_hidden_visibility: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "overrides the `default_hidden_visibility` setting of the target"),
    dep_info_omit_d_target: bool = (false, parse_bool, [TRACKED],
        "in dep-info output, omit targets for tracking dependencies of the dep-info files \
        themselves (default: no)"),
    direct_access_external_data: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "Direct or use GOT indirect to reference external data symbols"),
    dual_proc_macros: bool = (false, parse_bool, [TRACKED],
        "load proc macros for both target and host, but only link to the target (default: no)"),
    dump_dep_graph: bool = (false, parse_bool, [UNTRACKED],
        "dump the dependency graph to $RUST_DEP_GRAPH (default: /tmp/dep_graph.gv) \
        (default: no)"),
    dump_mir: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "dump MIR state to file.
        `val` is used to select which passes and functions to dump. For example:
        `all` matches all passes and functions,
        `foo` matches all passes for functions whose name contains 'foo',
        `foo & ConstProp` only the 'ConstProp' pass for function names containing 'foo',
        `foo | bar` all passes for function names containing 'foo' or 'bar'."),
    dump_mir_dataflow: bool = (false, parse_bool, [UNTRACKED],
        "in addition to `.mir` files, create graphviz `.dot` files with dataflow results \
        (default: no)"),
    dump_mir_dir: String = ("mir_dump".to_string(), parse_string, [UNTRACKED],
        "the directory the MIR is dumped into (default: `mir_dump`)"),
    dump_mir_exclude_pass_number: bool = (false, parse_bool, [UNTRACKED],
        "exclude the pass number when dumping MIR (used in tests) (default: no)"),
    dump_mir_graphviz: bool = (false, parse_bool, [UNTRACKED],
        "in addition to `.mir` files, create graphviz `.dot` files (default: no)"),
    dump_mono_stats: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [UNTRACKED],
        "output statistics about monomorphization collection"),
    dump_mono_stats_format: DumpMonoStatsFormat = (DumpMonoStatsFormat::Markdown, parse_dump_mono_stats, [UNTRACKED],
        "the format to use for -Z dump-mono-stats (`markdown` (default) or `json`)"),
    dwarf_version: Option<u32> = (None, parse_opt_number, [TRACKED],
        "version of DWARF debug information to emit (default: 2 or 4, depending on platform)"),
    dylib_lto: bool = (false, parse_bool, [UNTRACKED],
        "enables LTO for dylib crate type"),
    eagerly_emit_delayed_bugs: bool = (false, parse_bool, [UNTRACKED],
        "emit delayed bugs eagerly as errors instead of stashing them and emitting \
        them only if an error has not been emitted"),
    ehcont_guard: bool = (false, parse_bool, [TRACKED],
        "generate Windows EHCont Guard tables"),
    emit_stack_sizes: bool = (false, parse_bool, [UNTRACKED],
        "emit a section containing stack size metadata (default: no)"),
    emit_thin_lto: bool = (true, parse_bool, [TRACKED],
        "emit the bc module with thin LTO info (default: yes)"),
    export_executable_symbols: bool = (false, parse_bool, [TRACKED],
        "export symbols from executables, as if they were dynamic libraries"),
    extra_const_ub_checks: bool = (false, parse_bool, [TRACKED],
        "turns on more checks to detect const UB, which can be slow (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::fewer_names` instead of this field")]
    fewer_names: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "reduce memory use by retaining fewer names within compilation artifacts (LLVM-IR) \
        (default: no)"),
    flatten_format_args: bool = (true, parse_bool, [TRACKED],
        "flatten nested format_args!() and literals into a simplified format_args!() call \
        (default: yes)"),
    force_unstable_if_unmarked: bool = (false, parse_bool, [TRACKED],
        "force all crates to be `rustc_private` unstable (default: no)"),
    fuel: Option<(String, u64)> = (None, parse_optimization_fuel, [TRACKED],
        "set the optimization fuel quota for a crate"),
    function_return: FunctionReturn = (FunctionReturn::default(), parse_function_return, [TRACKED],
        "replace returns with jumps to `__x86_return_thunk` (default: `keep`)"),
    function_sections: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether each function should go in its own section"),
    future_incompat_test: bool = (false, parse_bool, [UNTRACKED],
        "forces all lints to be future incompatible, used for internal testing (default: no)"),
    graphviz_dark_mode: bool = (false, parse_bool, [UNTRACKED],
        "use dark-themed colors in graphviz output (default: no)"),
    graphviz_font: String = ("Courier, monospace".to_string(), parse_string, [UNTRACKED],
        "use the given `fontname` in graphviz output; can be overridden by setting \
        environment variable `RUSTC_GRAPHVIZ_FONT` (default: `Courier, monospace`)"),
    has_thread_local: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "explicitly enable the `cfg(target_thread_local)` directive"),
    hir_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some statistics about AST and HIR (default: no)"),
    human_readable_cgu_names: bool = (false, parse_bool, [TRACKED],
        "generate human-readable, predictable names for codegen units (default: no)"),
    identify_regions: bool = (false, parse_bool, [UNTRACKED],
        "display unnamed regions as `'<id>`, using a non-ident unique id (default: no)"),
    ignore_directory_in_diagnostics_source_blocks: Vec<String> = (Vec::new(), parse_string_push, [UNTRACKED],
        "do not display the source code block in diagnostics for files in the directory"),
    incremental_ignore_spans: bool = (false, parse_bool, [TRACKED],
        "ignore spans during ICH computation -- used for testing (default: no)"),
    incremental_info: bool = (false, parse_bool, [UNTRACKED],
        "print high-level information about incremental reuse (or the lack thereof) \
        (default: no)"),
    incremental_verify_ich: bool = (false, parse_bool, [UNTRACKED],
        "verify extended properties for incr. comp. (default: no):
        - hashes of green query instances
        - hash collisions of query keys"),
    inline_in_all_cgus: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "control whether `#[inline]` functions are in all CGUs"),
    inline_llvm: bool = (true, parse_bool, [TRACKED],
        "enable LLVM inlining (default: yes)"),
    inline_mir: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable MIR inlining (default: no)"),
    inline_mir_hint_threshold: Option<usize> = (None, parse_opt_number, [TRACKED],
        "inlining threshold for functions with inline hint (default: 100)"),
    inline_mir_threshold: Option<usize> = (None, parse_opt_number, [TRACKED],
        "a default MIR inlining threshold (default: 50)"),
    input_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather statistics about the input (default: no)"),
    instrument_mcount: bool = (false, parse_bool, [TRACKED],
        "insert function instrument code for mcount-based tracing (default: no)"),
    instrument_xray: Option<InstrumentXRay> = (None, parse_instrument_xray, [TRACKED],
        "insert function instrument code for XRay-based tracing (default: no)
         Optional extra settings:
         `=always`
         `=never`
         `=ignore-loops`
         `=instruction-threshold=N`
         `=skip-entry`
         `=skip-exit`
         Multiple options can be combined with commas."),
    layout_seed: Option<u64> = (None, parse_opt_number, [TRACKED],
        "seed layout randomization"),
    link_directives: bool = (true, parse_bool, [TRACKED],
        "honor #[link] directives in the compiled crate (default: yes)"),
    link_native_libraries: bool = (true, parse_bool, [UNTRACKED],
        "link native libraries in the linker invocation (default: yes)"),
    link_only: bool = (false, parse_bool, [TRACKED],
        "link the `.rlink` file generated by `-Z no-link` (default: no)"),
    lint_mir: bool = (false, parse_bool, [UNTRACKED],
        "lint MIR before and after each transformation"),
    llvm_module_flag: Vec<(String, u32, String)> = (Vec::new(), parse_llvm_module_flag, [TRACKED],
        "a list of module flags to pass to LLVM (space separated)"),
    llvm_plugins: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list LLVM plugins to enable (space separated)"),
    llvm_time_trace: bool = (false, parse_bool, [UNTRACKED],
        "generate JSON tracing data file from LLVM data (default: no)"),
    location_detail: LocationDetail = (LocationDetail::all(), parse_location_detail, [TRACKED],
        "what location details should be tracked when using caller_location, either \
        `none`, or a comma separated list of location details, for which \
        valid options are `file`, `line`, and `column` (default: `file,line,column`)"),
    ls: Vec<String> = (Vec::new(), parse_list, [UNTRACKED],
        "decode and print various parts of the crate metadata for a library crate \
        (space separated)"),
    macro_backtrace: bool = (false, parse_bool, [UNTRACKED],
        "show macro backtraces (default: no)"),
    maximal_hir_to_mir_coverage: bool = (false, parse_bool, [TRACKED],
        "save as much information as possible about the correspondence between MIR and HIR \
        as source scopes (default: no)"),
    merge_functions: Option<MergeFunctions> = (None, parse_merge_functions, [TRACKED],
        "control the operation of the MergeFunctions LLVM pass, taking \
        the same values as the target option of the same name"),
    meta_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather metadata statistics (default: no)"),
    mir_emit_retag: bool = (false, parse_bool, [TRACKED],
        "emit Retagging MIR statements, interpreted e.g., by miri; implies -Zmir-opt-level=0 \
        (default: no)"),
    mir_enable_passes: Vec<(String, bool)> = (Vec::new(), parse_list_with_polarity, [TRACKED],
        "use like `-Zmir-enable-passes=+DestinationPropagation,-InstSimplify`. Forces the \
        specified passes to be enabled, overriding all other checks. In particular, this will \
        enable unsound (known-buggy and hence usually disabled) passes without further warning! \
        Passes that are not specified are enabled or disabled by other flags as usual."),
    mir_include_spans: bool = (false, parse_bool, [UNTRACKED],
        "use line numbers relative to the function in mir pretty printing"),
    mir_keep_place_mention: bool = (false, parse_bool, [TRACKED],
        "keep place mention MIR statements, interpreted e.g., by miri; implies -Zmir-opt-level=0 \
        (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::mir_opt_level` instead of this field")]
    mir_opt_level: Option<usize> = (None, parse_opt_number, [TRACKED],
        "MIR optimization level (0-4; default: 1 in non optimized builds and 2 in optimized builds)"),
    move_size_limit: Option<usize> = (None, parse_opt_number, [TRACKED],
        "the size at which the `large_assignments` lint starts to be emitted"),
    mutable_noalias: bool = (true, parse_bool, [TRACKED],
        "emit noalias metadata for mutable references (default: yes)"),
    next_solver: Option<NextSolverConfig> = (None, parse_next_solver_config, [TRACKED],
        "enable and configure the next generation trait solver used by rustc"),
    nll_facts: bool = (false, parse_bool, [UNTRACKED],
        "dump facts from NLL analysis into side files (default: no)"),
    nll_facts_dir: String = ("nll-facts".to_string(), parse_string, [UNTRACKED],
        "the directory the NLL facts are dumped into (default: `nll-facts`)"),
    no_analysis: bool = (false, parse_no_flag, [UNTRACKED],
        "parse and expand the source, but run no analysis"),
    no_codegen: bool = (false, parse_no_flag, [TRACKED_NO_CRATE_HASH],
        "run all passes except codegen; no output"),
    no_generate_arange_section: bool = (false, parse_no_flag, [TRACKED],
        "omit DWARF address ranges that give faster lookups"),
    no_implied_bounds_compat: bool = (false, parse_bool, [TRACKED],
        "disable the compatibility version of the `implied_bounds_ty` query"),
    no_jump_tables: bool = (false, parse_no_flag, [TRACKED],
        "disable the jump tables and lookup tables that can be generated from a switch case lowering"),
    no_leak_check: bool = (false, parse_no_flag, [UNTRACKED],
        "disable the 'leak check' for subtyping; unsound, but useful for tests"),
    no_link: bool = (false, parse_no_flag, [TRACKED],
        "compile without linking"),
    no_parallel_backend: bool = (false, parse_no_flag, [UNTRACKED],
        "run LLVM in non-parallel mode (while keeping codegen-units and ThinLTO)"),
    no_profiler_runtime: bool = (false, parse_no_flag, [TRACKED],
        "prevent automatic injection of the profiler_builtins crate"),
    no_trait_vptr: bool = (false, parse_no_flag, [TRACKED],
        "disable generation of trait vptr in vtable for upcasting"),
    no_unique_section_names: bool = (false, parse_bool, [TRACKED],
        "do not use unique names for text and data sections when -Z function-sections is used"),
    normalize_docs: bool = (false, parse_bool, [TRACKED],
        "normalize associated items in rustdoc when generating documentation"),
    oom: OomStrategy = (OomStrategy::Abort, parse_oom_strategy, [TRACKED],
        "panic strategy for out-of-memory handling"),
    osx_rpath_install_name: bool = (false, parse_bool, [TRACKED],
        "pass `-install_name @rpath/...` to the macOS linker (default: no)"),
    packed_bundled_libs: bool = (false, parse_bool, [TRACKED],
        "change rlib format to store native libraries as archives"),
    panic_abort_tests: bool = (false, parse_bool, [TRACKED],
        "support compiling tests with panic=abort (default: no)"),
    panic_in_drop: PanicStrategy = (PanicStrategy::Unwind, parse_panic_strategy, [TRACKED],
        "panic strategy for panics in drops"),
    parse_only: bool = (false, parse_bool, [UNTRACKED],
        "parse only; do not compile, assemble, or link (default: no)"),
    plt: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether to use the PLT when calling into shared libraries;
        only has effect for PIC code on systems with ELF binaries
        (default: PLT is disabled if full relro is enabled on x86_64)"),
    polonius: Polonius = (Polonius::default(), parse_polonius, [TRACKED],
        "enable polonius-based borrow-checker (default: no)"),
    polymorphize: bool = (false, parse_bool, [TRACKED],
          "perform polymorphization analysis"),
    pre_link_arg: (/* redirected to pre_link_args */) = ((), parse_string_push, [UNTRACKED],
        "a single extra argument to prepend the linker invocation (can be used several times)"),
    pre_link_args: Vec<String> = (Vec::new(), parse_list, [UNTRACKED],
        "extra arguments to prepend to the linker invocation (space separated)"),
    precise_enum_drop_elaboration: bool = (true, parse_bool, [TRACKED],
        "use a more precise version of drop elaboration for matches on enums (default: yes). \
        This results in better codegen, but has caused miscompilations on some tier 2 platforms. \
        See #77382 and #74551."),
    #[rustc_lint_opt_deny_field_access("use `Session::print_codegen_stats` instead of this field")]
    print_codegen_stats: bool = (false, parse_bool, [UNTRACKED],
        "print codegen statistics (default: no)"),
    print_fuel: Option<String> = (None, parse_opt_string, [TRACKED],
        "make rustc print the total optimization fuel used by a crate"),
    print_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "print the LLVM optimization passes being run (default: no)"),
    print_mono_items: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "print the result of the monomorphization collection pass"),
    print_type_sizes: bool = (false, parse_bool, [UNTRACKED],
        "print layout information for each type encountered (default: no)"),
    print_vtable_sizes: bool = (false, parse_bool, [UNTRACKED],
        "print size comparison between old and new vtable layouts (default: no)"),
    proc_macro_backtrace: bool = (false, parse_bool, [UNTRACKED],
         "show backtraces for panics during proc-macro execution (default: no)"),
    proc_macro_execution_strategy: ProcMacroExecutionStrategy = (ProcMacroExecutionStrategy::SameThread,
        parse_proc_macro_execution_strategy, [UNTRACKED],
        "how to run proc-macro code (default: same-thread)"),
    profile: bool = (false, parse_bool, [TRACKED],
        "insert profiling code (default: no)"),
    profile_closures: bool = (false, parse_no_flag, [UNTRACKED],
        "profile size of closures"),
    profile_emit: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "file path to emit profiling data at runtime when using 'profile' \
        (default based on relative source path)"),
    profile_sample_use: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "use the given `.prof` file for sampled profile-guided optimization (also known as AutoFDO)"),
    profiler_runtime: String = (String::from("profiler_builtins"), parse_string, [TRACKED],
        "name of the profiler runtime crate to automatically inject (default: `profiler_builtins`)"),
    query_dep_graph: bool = (false, parse_bool, [UNTRACKED],
        "enable queries of the dependency graph for regression testing (default: no)"),
    randomize_layout: bool = (false, parse_bool, [TRACKED],
        "randomize the layout of types (default: no)"),
    relax_elf_relocations: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether ELF relocations can be relaxed"),
    relro_level: Option<RelroLevel> = (None, parse_relro_level, [TRACKED],
        "choose which RELRO level to use"),
    remap_cwd_prefix: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "remap paths under the current working directory to this path prefix"),
    remap_path_scope: RemapPathScopeComponents = (RemapPathScopeComponents::all(), parse_remap_path_scope, [TRACKED],
        "remap path scope (default: all)"),
    remark_dir: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "directory into which to write optimization remarks (if not specified, they will be \
written to standard error output)"),
    sanitizer: SanitizerSet = (SanitizerSet::empty(), parse_sanitizers, [TRACKED],
        "use a sanitizer"),
    sanitizer_cfi_canonical_jump_tables: Option<bool> = (Some(true), parse_opt_bool, [TRACKED],
        "enable canonical jump tables (default: yes)"),
    sanitizer_cfi_generalize_pointers: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable generalizing pointer types (default: no)"),
    sanitizer_cfi_normalize_integers: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable normalizing integer types (default: no)"),
    sanitizer_dataflow_abilist: Vec<String> = (Vec::new(), parse_comma_list, [TRACKED],
        "additional ABI list files that control how shadow parameters are passed (comma separated)"),
    sanitizer_memory_track_origins: usize = (0, parse_sanitizer_memory_track_origins, [TRACKED],
        "enable origins tracking in MemorySanitizer"),
    sanitizer_recover: SanitizerSet = (SanitizerSet::empty(), parse_sanitizers, [TRACKED],
        "enable recovery for selected sanitizers"),
    saturating_float_casts: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "make float->int casts UB-free: numbers outside the integer type's range are clipped to \
        the max/min integer respectively, and NaN is mapped to 0 (default: yes)"),
    self_profile: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [UNTRACKED],
        "run the self profiler and output the raw event data"),
    self_profile_counter: String = ("wall-time".to_string(), parse_string, [UNTRACKED],
        "counter used by the self profiler (default: `wall-time`), one of:
        `wall-time` (monotonic clock, i.e. `std::time::Instant`)
        `instructions:u` (retired instructions, userspace-only)
        `instructions-minus-irqs:u` (subtracting hardware interrupt counts for extra accuracy)"
    ),
    /// keep this in sync with the event filter names in librustc_data_structures/profiling.rs
    self_profile_events: Option<Vec<String>> = (None, parse_opt_comma_list, [UNTRACKED],
        "specify the events recorded by the self profiler;
        for example: `-Z self-profile-events=default,query-keys`
        all options: none, all, default, generic-activity, query-provider, query-cache-hit
                     query-blocked, incr-cache-load, incr-result-hashing, query-keys, function-args, args, llvm, artifact-sizes"),
    share_generics: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "make the current crate share its generic instantiations"),
    shell_argfiles: bool = (false, parse_bool, [UNTRACKED],
        "allow argument files to be specified with POSIX \"shell-style\" argument quoting"),
    show_span: Option<String> = (None, parse_opt_string, [TRACKED],
        "show spans for compiler debugging (expr|pat|ty)"),
    simulate_remapped_rust_src_base: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "simulate the effect of remap-debuginfo = true at bootstrapping by remapping path \
        to rust's source base directory. only meant for testing purposes"),
    span_debug: bool = (false, parse_bool, [UNTRACKED],
        "forward proc_macro::Span's `Debug` impl to `Span`"),
    /// o/w tests have closure@path
    span_free_formats: bool = (false, parse_bool, [UNTRACKED],
        "exclude spans when debug-printing compiler state (default: no)"),
    split_dwarf_inlining: bool = (false, parse_bool, [TRACKED],
        "provide minimal debug info in the object/executable to facilitate online \
         symbolication/stack traces in the absence of .dwo/.dwp files when using Split DWARF"),
    split_dwarf_kind: SplitDwarfKind = (SplitDwarfKind::Split, parse_split_dwarf_kind, [TRACKED],
        "split dwarf variant (only if -Csplit-debuginfo is enabled and on relevant platform)
        (default: `split`)

        `split`: sections which do not require relocation are written into a DWARF object (`.dwo`)
                 file which is ignored by the linker
        `single`: sections which do not require relocation are written into object file but ignored
                  by the linker"),
    split_lto_unit: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable LTO unit splitting (default: no)"),
    src_hash_algorithm: Option<SourceFileHashAlgorithm> = (None, parse_src_file_hash, [TRACKED],
        "hash algorithm of source files in debug info (`md5`, `sha1`, or `sha256`)"),
    #[rustc_lint_opt_deny_field_access("use `Session::stack_protector` instead of this field")]
    stack_protector: StackProtector = (StackProtector::None, parse_stack_protector, [TRACKED],
        "control stack smash protection strategy (`rustc --print stack-protector-strategies` for details)"),
    staticlib_allow_rdylib_deps: bool = (false, parse_bool, [TRACKED],
        "allow staticlibs to have rust dylib dependencies"),
    staticlib_prefer_dynamic: bool = (false, parse_bool, [TRACKED],
        "prefer dynamic linking to static linking for staticlibs (default: no)"),
    strict_init_checks: bool = (false, parse_bool, [TRACKED],
        "control if mem::uninitialized and mem::zeroed panic on more UB"),
    #[rustc_lint_opt_deny_field_access("use `Session::teach` instead of this field")]
    teach: bool = (false, parse_bool, [TRACKED],
        "show extended diagnostic help (default: no)"),
    temps_dir: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "the directory the intermediate files are written to"),
    terminal_urls: TerminalUrl = (TerminalUrl::No, parse_terminal_url, [UNTRACKED],
        "use the OSC 8 hyperlink terminal specification to print hyperlinks in the compiler output"),
    #[rustc_lint_opt_deny_field_access("use `Session::lto` instead of this field")]
    thinlto: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable ThinLTO when possible"),
    thir_unsafeck: bool = (true, parse_bool, [TRACKED],
        "use the THIR unsafety checker (default: yes)"),
    /// We default to 1 here since we want to behave like
    /// a sequential compiler for now. This'll likely be adjusted
    /// in the future. Note that -Zthreads=0 is the way to get
    /// the num_cpus behavior.
    #[rustc_lint_opt_deny_field_access("use `Session::threads` instead of this field")]
    threads: usize = (1, parse_threads, [UNTRACKED],
        "use a thread pool with N threads"),
    time_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each LLVM pass (default: no)"),
    time_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each rustc pass (default: no)"),
    time_passes_format: TimePassesFormat = (TimePassesFormat::Text, parse_time_passes_format, [UNTRACKED],
        "the format to use for -Z time-passes (`text` (default) or `json`)"),
    tiny_const_eval_limit: bool = (false, parse_bool, [TRACKED],
        "sets a tiny, non-configurable limit for const eval; useful for compiler tests"),
    #[rustc_lint_opt_deny_field_access("use `Session::tls_model` instead of this field")]
    tls_model: Option<TlsModel> = (None, parse_tls_model, [TRACKED],
        "choose the TLS model to use (`rustc --print tls-models` for details)"),
    trace_macros: bool = (false, parse_bool, [UNTRACKED],
        "for every macro invocation, print its name and arguments (default: no)"),
    track_diagnostics: bool = (false, parse_bool, [UNTRACKED],
        "tracks where in rustc a diagnostic was emitted"),
    // Diagnostics are considered side-effects of a query (see `QuerySideEffects`) and are saved
    // alongside query results and changes to translation options can affect diagnostics - so
    // translation options should be tracked.
    translate_additional_ftl: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "additional fluent translation to preferentially use (for testing translation)"),
    translate_directionality_markers: bool = (false, parse_bool, [TRACKED],
        "emit directionality isolation markers in translated diagnostics"),
    translate_lang: Option<LanguageIdentifier> = (None, parse_opt_langid, [TRACKED],
        "language identifier for diagnostic output"),
    translate_remapped_path_to_local_path: bool = (true, parse_bool, [TRACKED],
        "translate remapped paths into local paths when possible (default: yes)"),
    trap_unreachable: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "generate trap instructions for unreachable intrinsics (default: use target setting, usually yes)"),
    treat_err_as_bug: Option<NonZero<usize>> = (None, parse_treat_err_as_bug, [TRACKED],
        "treat the `val`th error that occurs as bug (default if not specified: 0 - don't treat errors as bugs. \
        default if specified without a value: 1 - treat the first error as bug)"),
    trim_diagnostic_paths: bool = (true, parse_bool, [UNTRACKED],
        "in diagnostics, use heuristics to shorten paths referring to items"),
    tune_cpu: Option<String> = (None, parse_opt_string, [TRACKED],
        "select processor to schedule for (`rustc --print target-cpus` for details)"),
    ui_testing: bool = (false, parse_bool, [UNTRACKED],
        "emit compiler diagnostics in a form suitable for UI testing (default: no)"),
    uninit_const_chunk_threshold: usize = (16, parse_number, [TRACKED],
        "allow generating const initializers with mixed init/uninit chunks, \
        and set the maximum number of chunks for which this is allowed (default: 16)"),
    unleash_the_miri_inside_of_you: bool = (false, parse_bool, [TRACKED],
        "take the brakes off const evaluation. NOTE: this is unsound (default: no)"),
    unpretty: Option<String> = (None, parse_unpretty, [UNTRACKED],
        "present the input source, unstable (and less-pretty) variants;
        `normal`, `identified`,
        `expanded`, `expanded,identified`,
        `expanded,hygiene` (with internal representations),
        `ast-tree` (raw AST before expansion),
        `ast-tree,expanded` (raw AST after expansion),
        `hir` (the HIR), `hir,identified`,
        `hir,typed` (HIR with types for each node),
        `hir-tree` (dump the raw HIR),
        `thir-tree`, `thir-flat`,
        `mir` (the MIR), or `mir-cfg` (graphviz formatted MIR)"),
    unsound_mir_opts: bool = (false, parse_bool, [TRACKED],
        "enable unsound and buggy MIR optimizations (default: no)"),
    /// This name is kind of confusing: Most unstable options enable something themselves, while
    /// this just allows "normal" options to be feature-gated.
    #[rustc_lint_opt_deny_field_access("use `Session::unstable_options` instead of this field")]
    unstable_options: bool = (false, parse_bool, [UNTRACKED],
        "adds unstable command line options to rustc interface (default: no)"),
    use_ctors_section: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "use legacy .ctors section for initializers rather than .init_array"),
    use_sync_unwind: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "Generate sync unwind tables instead of async unwind tables (default: no)"),
    validate_mir: bool = (false, parse_bool, [UNTRACKED],
        "validate MIR after each transformation"),
    #[rustc_lint_opt_deny_field_access("use `Session::verbose_internals` instead of this field")]
    verbose_internals: bool = (false, parse_bool, [TRACKED_NO_CRATE_HASH],
        "in general, enable more debug printouts (default: no)"),
    #[rustc_lint_opt_deny_field_access("use `Session::verify_llvm_ir` instead of this field")]
    verify_llvm_ir: bool = (false, parse_bool, [TRACKED],
        "verify LLVM IR (default: no)"),
    virtual_function_elimination: bool = (false, parse_bool, [TRACKED],
        "enables dead virtual function elimination optimization. \
        Requires `-Clto[=[fat,yes]]`"),
    wasi_exec_model: Option<WasiExecModel> = (None, parse_wasi_exec_model, [TRACKED],
        "whether to build a wasi command or reactor"),
    write_long_types_to_disk: bool = (true, parse_bool, [UNTRACKED],
        "whether long type names should be written to files instead of being printed in errors"),
    // tidy-alphabetical-end

    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/unstable-book/src/compiler-flags
}
