use crate::config::*;

use crate::early_error;
use crate::lint;
use crate::search_paths::SearchPath;
use crate::utils::NativeLibKind;

use rustc_target::spec::{CodeModel, LinkerFlavor, MergeFunctions, PanicStrategy};
use rustc_target::spec::{RelocModel, RelroLevel, TargetTriple, TlsModel};

use rustc_feature::UnstableFeatures;
use rustc_span::edition::Edition;
use rustc_span::SourceFileHashAlgorithm;

use std::collections::BTreeMap;

use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::path::PathBuf;
use std::str;

macro_rules! hash_option {
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, [UNTRACKED]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, [TRACKED]) => {{
        if $sub_hashes
            .insert(stringify!($opt_name), $opt_expr as &dyn dep_tracking::DepTrackingHash)
            .is_some()
        {
            panic!("duplicate key in CLI DepTrackingHash: {}", stringify!($opt_name))
        }
    }};
}

macro_rules! top_level_options {
    (pub struct Options { $(
        $opt:ident : $t:ty [$dep_tracking_marker:ident $($warn_val:expr, $warn_text:expr)*],
    )* } ) => (
        #[derive(Clone)]
        pub struct Options {
            $(pub $opt: $t),*
        }

        impl Options {
            pub fn dep_tracking_hash(&self) -> u64 {
                let mut sub_hashes = BTreeMap::new();
                $({
                    hash_option!($opt,
                                 &self.$opt,
                                 &mut sub_hashes,
                                 [$dep_tracking_marker $($warn_val,
                                                         $warn_text,
                                                         self.error_format)*]);
                })*
                let mut hasher = DefaultHasher::new();
                dep_tracking::stable_hash(sub_hashes,
                                          &mut hasher,
                                          self.error_format);
                hasher.finish()
            }
        }
    );
}

// The top-level command-line options struct.
//
// For each option, one has to specify how it behaves with regard to the
// dependency tracking system of incremental compilation. This is done via the
// square-bracketed directive after the field type. The options are:
//
// [TRACKED]
// A change in the given field will cause the compiler to completely clear the
// incremental compilation cache before proceeding.
//
// [UNTRACKED]
// Incremental compilation is not influenced by this option.
//
// If you add a new option to this struct or one of the sub-structs like
// `CodegenOptions`, think about how it influences incremental compilation. If in
// doubt, specify [TRACKED], which is always "correct" but might lead to
// unnecessary re-compilation.
top_level_options!(
    pub struct Options {
        // The crate config requested for the session, which may be combined
        // with additional crate configurations during the compile process.
        crate_types: Vec<CrateType> [TRACKED],
        optimize: OptLevel [TRACKED],
        // Include the `debug_assertions` flag in dependency tracking, since it
        // can influence whether overflow checks are done or not.
        debug_assertions: bool [TRACKED],
        debuginfo: DebugInfo [TRACKED],
        lint_opts: Vec<(String, lint::Level)> [TRACKED],
        lint_cap: Option<lint::Level> [TRACKED],
        describe_lints: bool [UNTRACKED],
        output_types: OutputTypes [TRACKED],
        search_paths: Vec<SearchPath> [UNTRACKED],
        libs: Vec<(String, Option<String>, NativeLibKind)> [TRACKED],
        maybe_sysroot: Option<PathBuf> [UNTRACKED],

        target_triple: TargetTriple [TRACKED],

        test: bool [TRACKED],
        error_format: ErrorOutputType [UNTRACKED],

        // If `Some`, enable incremental compilation, using the given
        // directory to store intermediate results.
        incremental: Option<PathBuf> [UNTRACKED],

        debugging_opts: DebuggingOptions [TRACKED],
        prints: Vec<PrintRequest> [UNTRACKED],
        // Determines which borrow checker(s) to run. This is the parsed, sanitized
        // version of `debugging_opts.borrowck`, which is just a plain string.
        borrowck_mode: BorrowckMode [UNTRACKED],
        cg: CodegenOptions [TRACKED],
        externs: Externs [UNTRACKED],
        crate_name: Option<String> [TRACKED],
        // An optional name to use as the crate for std during std injection,
        // written `extern crate name as std`. Defaults to `std`. Used by
        // out-of-tree drivers.
        alt_std_name: Option<String> [TRACKED],
        // Indicates how the compiler should treat unstable features.
        unstable_features: UnstableFeatures [TRACKED],

        // Indicates whether this run of the compiler is actually rustdoc. This
        // is currently just a hack and will be removed eventually, so please
        // try to not rely on this too much.
        actually_rustdoc: bool [TRACKED],

        // Control path trimming.
        trimmed_def_paths: TrimmedDefPaths [TRACKED],

        // Specifications of codegen units / ThinLTO which are forced as a
        // result of parsing command line options. These are not necessarily
        // what rustc was invoked with, but massaged a bit to agree with
        // commands like `--emit llvm-ir` which they're often incompatible with
        // if we otherwise use the defaults of rustc.
        cli_forced_codegen_units: Option<usize> [UNTRACKED],
        cli_forced_thinlto_off: bool [UNTRACKED],

        // Remap source path prefixes in all output (messages, object files, debug, etc.).
        remap_path_prefix: Vec<(PathBuf, PathBuf)> [UNTRACKED],

        edition: Edition [TRACKED],

        // `true` if we're emitting JSON blobs about each artifact produced
        // by the compiler.
        json_artifact_notifications: bool [TRACKED],

        pretty: Option<PpMode> [UNTRACKED],
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
    ($struct_name:ident, $setter_name:ident, $defaultfn:ident,
     $buildfn:ident, $prefix:expr, $outputname:expr,
     $stat:ident, $mod_desc:ident, $mod_set:ident,
     $($opt:ident : $t:ty = (
        $init:expr,
        $parse:ident,
        [$dep_tracking_marker:ident $(($dep_warn_val:expr, $dep_warn_text:expr))*],
        $desc:expr)
     ),* ,) =>
(
    #[derive(Clone)]
    pub struct $struct_name { $(pub $opt: $t),* }

    pub fn $defaultfn() -> $struct_name {
        $struct_name { $($opt: $init),* }
    }

    pub fn $buildfn(matches: &getopts::Matches, error_format: ErrorOutputType) -> $struct_name
    {
        let mut op = $defaultfn();
        for option in matches.opt_strs($prefix) {
            let mut iter = option.splitn(2, '=');
            let key = iter.next().unwrap();
            let value = iter.next();
            let option_to_lookup = key.replace("-", "_");
            let mut found = false;
            for &(candidate, setter, type_desc, _) in $stat {
                if option_to_lookup != candidate { continue }
                if !setter(&mut op, value) {
                    match value {
                        None => {
                            early_error(error_format, &format!("{0} option `{1}` requires \
                                                                {2} ({3} {1}=<value>)",
                                                               $outputname, key,
                                                               type_desc, $prefix))
                        }
                        Some(value) => {
                            early_error(error_format, &format!("incorrect value `{}` for {} \
                                                                option `{}` - {} was expected",
                                                               value, $outputname,
                                                               key, type_desc))
                        }
                    }
                }
                found = true;
                break;
            }
            if !found {
                early_error(error_format, &format!("unknown {} option: `{}`",
                                                   $outputname, key));
            }
        }
        return op;
    }

    impl dep_tracking::DepTrackingHash for $struct_name {
        fn hash(&self, hasher: &mut DefaultHasher, error_format: ErrorOutputType) {
            let mut sub_hashes = BTreeMap::new();
            $({
                hash_option!($opt,
                             &self.$opt,
                             &mut sub_hashes,
                             [$dep_tracking_marker $($dep_warn_val,
                                                     $dep_warn_text,
                                                     error_format)*]);
            })*
            dep_tracking::stable_hash(sub_hashes, hasher, error_format);
        }
    }

    pub type $setter_name = fn(&mut $struct_name, v: Option<&str>) -> bool;
    pub const $stat: &[(&str, $setter_name, &str, &str)] =
        &[ $( (stringify!($opt), $mod_set::$opt, $mod_desc::$parse, $desc) ),* ];

    #[allow(non_upper_case_globals, dead_code)]
    mod $mod_desc {
        pub const parse_no_flag: &str = "no value";
        pub const parse_bool: &str = "one of: `y`, `yes`, `on`, `n`, `no`, or `off`";
        pub const parse_opt_bool: &str = parse_bool;
        pub const parse_string: &str = "a string";
        pub const parse_opt_string: &str = parse_string;
        pub const parse_string_push: &str = parse_string;
        pub const parse_opt_pathbuf: &str = "a path";
        pub const parse_pathbuf_push: &str = parse_opt_pathbuf;
        pub const parse_list: &str = "a space-separated list of strings";
        pub const parse_opt_list: &str = parse_list;
        pub const parse_opt_comma_list: &str = "a comma-separated list of strings";
        pub const parse_uint: &str = "a number";
        pub const parse_opt_uint: &str = parse_uint;
        pub const parse_threads: &str = parse_uint;
        pub const parse_passes: &str = "a space-separated list of passes, or `all`";
        pub const parse_panic_strategy: &str = "either `unwind` or `abort`";
        pub const parse_relro_level: &str = "one of: `full`, `partial`, or `off`";
        pub const parse_sanitizers: &str = "comma separated list of sanitizers: `address`, `leak`, `memory` or `thread`";
        pub const parse_sanitizer_memory_track_origins: &str = "0, 1, or 2";
        pub const parse_cfguard: &str =
            "either a boolean (`yes`, `no`, `on`, `off`, etc), `checks`, or `nochecks`";
        pub const parse_strip: &str = "either `none`, `debuginfo`, or `symbols`";
        pub const parse_linker_flavor: &str = ::rustc_target::spec::LinkerFlavor::one_of();
        pub const parse_optimization_fuel: &str = "crate=integer";
        pub const parse_mir_spanview: &str = "`statement` (default), `terminator`, or `block`";
        pub const parse_unpretty: &str = "`string` or `string=string`";
        pub const parse_treat_err_as_bug: &str = "either no value or a number bigger than 0";
        pub const parse_lto: &str =
            "either a boolean (`yes`, `no`, `on`, `off`, etc), `thin`, `fat`, or omitted";
        pub const parse_linker_plugin_lto: &str =
            "either a boolean (`yes`, `no`, `on`, `off`, etc), or the path to the linker plugin";
        pub const parse_switch_with_opt_path: &str =
            "an optional path to the profiling data output directory";
        pub const parse_merge_functions: &str = "one of: `disabled`, `trampolines`, or `aliases`";
        pub const parse_symbol_mangling_version: &str = "either `legacy` or `v0` (RFC 2603)";
        pub const parse_src_file_hash: &str = "either `md5` or `sha1`";
        pub const parse_relocation_model: &str =
            "one of supported relocation models (`rustc --print relocation-models`)";
        pub const parse_code_model: &str =
            "one of supported code models (`rustc --print code-models`)";
        pub const parse_tls_model: &str =
            "one of supported TLS models (`rustc --print tls-models`)";
        pub const parse_target_feature: &str = parse_string;
    }

    #[allow(dead_code)]
    mod $mod_set {
        use super::*;
        use std::str::FromStr;

        // Sometimes different options need to build a common structure.
        // That structure can kept in one of the options' fields, the others become dummy.
        macro_rules! redirect_field {
            ($cg:ident.link_arg) => { $cg.link_args };
            ($cg:ident.pre_link_arg) => { $cg.pre_link_args };
            ($cg:ident.$field:ident) => { $cg.$field };
        }

        $(
            pub fn $opt(cg: &mut $struct_name, v: Option<&str>) -> bool {
                $parse(&mut redirect_field!(cg.$opt), v)
            }
        )*

        /// This is for boolean options that don't take a value and start with
        /// `no-`. This style of option is deprecated.
        fn parse_no_flag(slot: &mut bool, v: Option<&str>) -> bool {
            match v {
                None => { *slot = true; true }
                Some(_) => false,
            }
        }

        /// Use this for any boolean option that has a static default.
        fn parse_bool(slot: &mut bool, v: Option<&str>) -> bool {
            match v {
                Some("y") | Some("yes") | Some("on") | None => { *slot = true; true }
                Some("n") | Some("no") | Some("off") => { *slot = false; true }
                _ => false,
            }
        }

        /// Use this for any boolean option that lacks a static default. (The
        /// actions taken when such an option is not specified will depend on
        /// other factors, such as other options, or target options.)
        fn parse_opt_bool(slot: &mut Option<bool>, v: Option<&str>) -> bool {
            match v {
                Some("y") | Some("yes") | Some("on") | None => { *slot = Some(true); true }
                Some("n") | Some("no") | Some("off") => { *slot = Some(false); true }
                _ => false,
            }
        }

        /// Use this for any string option that has a static default.
        fn parse_string(slot: &mut String, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = s.to_string(); true },
                None => false,
            }
        }

        /// Use this for any string option that lacks a static default.
        fn parse_opt_string(slot: &mut Option<String>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = Some(s.to_string()); true },
                None => false,
            }
        }

        fn parse_opt_pathbuf(slot: &mut Option<PathBuf>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = Some(PathBuf::from(s)); true },
                None => false,
            }
        }

        fn parse_string_push(slot: &mut Vec<String>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { slot.push(s.to_string()); true },
                None => false,
            }
        }

        fn parse_pathbuf_push(slot: &mut Vec<PathBuf>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { slot.push(PathBuf::from(s)); true },
                None => false,
            }
        }

        fn parse_list(slot: &mut Vec<String>, v: Option<&str>)
                      -> bool {
            match v {
                Some(s) => {
                    slot.extend(s.split_whitespace().map(|s| s.to_string()));
                    true
                },
                None => false,
            }
        }

        fn parse_opt_list(slot: &mut Option<Vec<String>>, v: Option<&str>)
                      -> bool {
            match v {
                Some(s) => {
                    let v = s.split_whitespace().map(|s| s.to_string()).collect();
                    *slot = Some(v);
                    true
                },
                None => false,
            }
        }

        fn parse_opt_comma_list(slot: &mut Option<Vec<String>>, v: Option<&str>)
                      -> bool {
            match v {
                Some(s) => {
                    let v = s.split(',').map(|s| s.to_string()).collect();
                    *slot = Some(v);
                    true
                },
                None => false,
            }
        }

        fn parse_threads(slot: &mut usize, v: Option<&str>) -> bool {
            match v.and_then(|s| s.parse().ok()) {
                Some(0) => { *slot = ::num_cpus::get(); true },
                Some(i) => { *slot = i; true },
                None => false
            }
        }

        /// Use this for any uint option that has a static default.
        fn parse_uint(slot: &mut usize, v: Option<&str>) -> bool {
            match v.and_then(|s| s.parse().ok()) {
                Some(i) => { *slot = i; true },
                None => false
            }
        }

        /// Use this for any uint option that lacks a static default.
        fn parse_opt_uint(slot: &mut Option<usize>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = s.parse().ok(); slot.is_some() }
                None => false
            }
        }

        fn parse_passes(slot: &mut Passes, v: Option<&str>) -> bool {
            match v {
                Some("all") => {
                    *slot = Passes::All;
                    true
                }
                v => {
                    let mut passes = vec![];
                    if parse_list(&mut passes, v) {
                        *slot = Passes::Some(passes);
                        true
                    } else {
                        false
                    }
                }
            }
        }

        fn parse_panic_strategy(slot: &mut Option<PanicStrategy>, v: Option<&str>) -> bool {
            match v {
                Some("unwind") => *slot = Some(PanicStrategy::Unwind),
                Some("abort") => *slot = Some(PanicStrategy::Abort),
                _ => return false
            }
            true
        }

        fn parse_relro_level(slot: &mut Option<RelroLevel>, v: Option<&str>) -> bool {
            match v {
                Some(s) => {
                    match s.parse::<RelroLevel>() {
                        Ok(level) => *slot = Some(level),
                        _ => return false
                    }
                },
                _ => return false
            }
            true
        }

        fn parse_sanitizers(slot: &mut SanitizerSet, v: Option<&str>) -> bool {
            if let Some(v) = v {
                for s in v.split(',') {
                    *slot |= match s {
                        "address" => SanitizerSet::ADDRESS,
                        "leak" => SanitizerSet::LEAK,
                        "memory" => SanitizerSet::MEMORY,
                        "thread" => SanitizerSet::THREAD,
                        _ => return false,
                    }
                }
                true
            } else {
                false
            }
        }

        fn parse_sanitizer_memory_track_origins(slot: &mut usize, v: Option<&str>) -> bool {
            match v {
                Some("2") | None => { *slot = 2; true }
                Some("1") => { *slot = 1; true }
                Some("0") => { *slot = 0; true }
                Some(_) => false,
            }
        }

        fn parse_strip(slot: &mut Strip, v: Option<&str>) -> bool {
            match v {
                Some("none") => *slot = Strip::None,
                Some("debuginfo") => *slot = Strip::Debuginfo,
                Some("symbols") => *slot = Strip::Symbols,
                _ => return false,
            }
            true
        }

        fn parse_cfguard(slot: &mut CFGuard, v: Option<&str>) -> bool {
            if v.is_some() {
                let mut bool_arg = None;
                if parse_opt_bool(&mut bool_arg, v) {
                    *slot = if bool_arg.unwrap() {
                        CFGuard::Checks
                    } else {
                        CFGuard::Disabled
                    };
                    return true
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

        fn parse_linker_flavor(slote: &mut Option<LinkerFlavor>, v: Option<&str>) -> bool {
            match v.and_then(LinkerFlavor::from_str) {
                Some(lf) => *slote = Some(lf),
                _ => return false,
            }
            true
        }

        fn parse_optimization_fuel(slot: &mut Option<(String, u64)>, v: Option<&str>) -> bool {
            match v {
                None => false,
                Some(s) => {
                    let parts = s.split('=').collect::<Vec<_>>();
                    if parts.len() != 2 { return false; }
                    let crate_name = parts[0].to_string();
                    let fuel = parts[1].parse::<u64>();
                    if fuel.is_err() { return false; }
                    *slot = Some((crate_name, fuel.unwrap()));
                    true
                }
            }
        }

        fn parse_unpretty(slot: &mut Option<String>, v: Option<&str>) -> bool {
            match v {
                None => false,
                Some(s) if s.split('=').count() <= 2 => {
                    *slot = Some(s.to_string());
                    true
                }
                _ => false,
            }
        }

        fn parse_mir_spanview(slot: &mut Option<MirSpanview>, v: Option<&str>) -> bool {
            if v.is_some() {
                let mut bool_arg = None;
                if parse_opt_bool(&mut bool_arg, v) {
                    *slot = if bool_arg.unwrap() {
                        Some(MirSpanview::Statement)
                    } else {
                        None
                    };
                    return true
                }
            }

            let v = match v {
                None => {
                    *slot = Some(MirSpanview::Statement);
                    return true;
                }
                Some(v) => v,
            };

            *slot = Some(match v.trim_end_matches("s") {
                "statement" | "stmt" => MirSpanview::Statement,
                "terminator" | "term" => MirSpanview::Terminator,
                "block" | "basicblock" => MirSpanview::Block,
                _ => return false,
            });
            true
        }

        fn parse_treat_err_as_bug(slot: &mut Option<usize>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = s.parse().ok().filter(|&x| x != 0); slot.unwrap_or(0) != 0 }
                None => { *slot = Some(1); true }
            }
        }

        fn parse_lto(slot: &mut LtoCli, v: Option<&str>) -> bool {
            if v.is_some() {
                let mut bool_arg = None;
                if parse_opt_bool(&mut bool_arg, v) {
                    *slot = if bool_arg.unwrap() {
                        LtoCli::Yes
                    } else {
                        LtoCli::No
                    };
                    return true
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

        fn parse_linker_plugin_lto(slot: &mut LinkerPluginLto, v: Option<&str>) -> bool {
            if v.is_some() {
                let mut bool_arg = None;
                if parse_opt_bool(&mut bool_arg, v) {
                    *slot = if bool_arg.unwrap() {
                        LinkerPluginLto::LinkerPluginAuto
                    } else {
                        LinkerPluginLto::Disabled
                    };
                    return true
                }
            }

            *slot = match v {
                None => LinkerPluginLto::LinkerPluginAuto,
                Some(path) => LinkerPluginLto::LinkerPlugin(PathBuf::from(path)),
            };
            true
        }

        fn parse_switch_with_opt_path(slot: &mut SwitchWithOptPath, v: Option<&str>) -> bool {
            *slot = match v {
                None => SwitchWithOptPath::Enabled(None),
                Some(path) => SwitchWithOptPath::Enabled(Some(PathBuf::from(path))),
            };
            true
        }

        fn parse_merge_functions(slot: &mut Option<MergeFunctions>, v: Option<&str>) -> bool {
            match v.and_then(|s| MergeFunctions::from_str(s).ok()) {
                Some(mergefunc) => *slot = Some(mergefunc),
                _ => return false,
            }
            true
        }

        fn parse_relocation_model(slot: &mut Option<RelocModel>, v: Option<&str>) -> bool {
            match v.and_then(|s| RelocModel::from_str(s).ok()) {
                Some(relocation_model) => *slot = Some(relocation_model),
                None if v == Some("default") => *slot = None,
                _ => return false,
            }
            true
        }

        fn parse_code_model(slot: &mut Option<CodeModel>, v: Option<&str>) -> bool {
            match v.and_then(|s| CodeModel::from_str(s).ok()) {
                Some(code_model) => *slot = Some(code_model),
                _ => return false,
            }
            true
        }

        fn parse_tls_model(slot: &mut Option<TlsModel>, v: Option<&str>) -> bool {
            match v.and_then(|s| TlsModel::from_str(s).ok()) {
                Some(tls_model) => *slot = Some(tls_model),
                _ => return false,
            }
            true
        }

        fn parse_symbol_mangling_version(
            slot: &mut SymbolManglingVersion,
            v: Option<&str>,
        ) -> bool {
            *slot = match v {
                Some("legacy") => SymbolManglingVersion::Legacy,
                Some("v0") => SymbolManglingVersion::V0,
                _ => return false,
            };
            true
        }

        fn parse_src_file_hash(slot: &mut Option<SourceFileHashAlgorithm>, v: Option<&str>) -> bool {
            match v.and_then(|s| SourceFileHashAlgorithm::from_str(s).ok()) {
                Some(hash_kind) => *slot = Some(hash_kind),
                _ => return false,
            }
            true
        }

        fn parse_target_feature(slot: &mut String, v: Option<&str>) -> bool {
            match v {
                Some(s) => {
                    if !slot.is_empty() {
                        slot.push_str(",");
                    }
                    slot.push_str(s);
                    true
                }
                None => false,
            }
        }
    }
) }

options! {CodegenOptions, CodegenSetter, basic_codegen_options,
          build_codegen_options, "C", "codegen",
          CG_OPTIONS, cg_type_desc, cgsetters,

    // This list is in alphabetical order.
    //
    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/rustc/src/codegen-options/index.md

    ar: String = (String::new(), parse_string, [UNTRACKED],
        "this option is deprecated and does nothing"),
    code_model: Option<CodeModel> = (None, parse_code_model, [TRACKED],
        "choose the code model to use (`rustc --print code-models` for details)"),
    codegen_units: Option<usize> = (None, parse_opt_uint, [UNTRACKED],
        "divide crate into N units to optimize in parallel"),
    control_flow_guard: CFGuard = (CFGuard::Disabled, parse_cfguard, [TRACKED],
        "use Windows Control Flow Guard (default: no)"),
    debug_assertions: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "explicitly enable the `cfg(debug_assertions)` directive"),
    debuginfo: usize = (0, parse_uint, [TRACKED],
        "debug info emission level (0 = no debug info, 1 = line tables only, \
        2 = full debug info with variable and type information; default: 0)"),
    default_linker_libraries: bool = (false, parse_bool, [UNTRACKED],
        "allow the linker to link its default libraries (default: no)"),
    embed_bitcode: bool = (true, parse_bool, [TRACKED],
        "emit bitcode in rlibs (default: yes)"),
    extra_filename: String = (String::new(), parse_string, [UNTRACKED],
        "extra data to put in each output filename"),
    force_frame_pointers: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "force use of the frame pointers"),
    force_unwind_tables: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "force use of unwind tables"),
    incremental: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "enable incremental compilation"),
    inline_threshold: Option<usize> = (None, parse_opt_uint, [TRACKED],
        "set the threshold for inlining a function"),
    link_arg: (/* redirected to link_args */) = ((), parse_string_push, [UNTRACKED],
        "a single extra argument to append to the linker invocation (can be used several times)"),
    link_args: Vec<String> = (Vec::new(), parse_list, [UNTRACKED],
        "extra arguments to append to the linker invocation (space separated)"),
    link_dead_code: Option<bool> = (None, parse_opt_bool, [UNTRACKED],
        "keep dead code at link time (useful for code coverage) (default: no)"),
    link_self_contained: Option<bool> = (None, parse_opt_bool, [UNTRACKED],
        "control whether to link Rust provided C objects/libraries or rely
        on C toolchain installed in the system"),
    linker: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "system linker to link outputs with"),
    linker_flavor: Option<LinkerFlavor> = (None, parse_linker_flavor, [UNTRACKED],
        "linker flavor"),
    linker_plugin_lto: LinkerPluginLto = (LinkerPluginLto::Disabled,
        parse_linker_plugin_lto, [TRACKED],
        "generate build artifacts that are compatible with linker-based LTO"),
    llvm_args: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of arguments to pass to LLVM (space separated)"),
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
    overflow_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "use overflow checks for integer arithmetic"),
    panic: Option<PanicStrategy> = (None, parse_panic_strategy, [TRACKED],
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
    relocation_model: Option<RelocModel> = (None, parse_relocation_model, [TRACKED],
        "control generation of position-independent code (PIC) \
        (`rustc --print relocation-models` for details)"),
    remark: Passes = (Passes::Some(Vec::new()), parse_passes, [UNTRACKED],
        "print remarks for these optimization passes (space separated, or \"all\")"),
    rpath: bool = (false, parse_bool, [UNTRACKED],
        "set rpath values in libs/exes (default: no)"),
    save_temps: bool = (false, parse_bool, [UNTRACKED],
        "save all temporary output files during compilation (default: no)"),
    soft_float: bool = (false, parse_bool, [TRACKED],
        "use soft float ABI (*eabihf targets only) (default: no)"),
    target_cpu: Option<String> = (None, parse_opt_string, [TRACKED],
        "select target processor (`rustc --print target-cpus` for details)"),
    target_feature: String = (String::new(), parse_target_feature, [TRACKED],
        "target specific attributes. (`rustc --print target-features` for details). \
        This feature is unsafe."),

    // This list is in alphabetical order.
    //
    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/rustc/src/codegen-options/index.md
}

options! {DebuggingOptions, DebuggingSetter, basic_debugging_options,
          build_debugging_options, "Z", "debugging",
          DB_OPTIONS, db_type_desc, dbsetters,

    // This list is in alphabetical order.
    //
    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs

    allow_features: Option<Vec<String>> = (None, parse_opt_comma_list, [TRACKED],
        "only allow the listed language features to be enabled in code (space separated)"),
    always_encode_mir: bool = (false, parse_bool, [TRACKED],
        "encode MIR of all functions into the crate metadata (default: no)"),
    asm_comments: bool = (false, parse_bool, [TRACKED],
        "generate comments into the assembly (may change behavior) (default: no)"),
    ast_json: bool = (false, parse_bool, [UNTRACKED],
        "print the AST as JSON and halt (default: no)"),
    ast_json_noexpand: bool = (false, parse_bool, [UNTRACKED],
        "print the pre-expansion AST as JSON and halt (default: no)"),
    binary_dep_depinfo: bool = (false, parse_bool, [TRACKED],
        "include artifacts (sysroot, crate dependencies) used during compilation in dep-info \
        (default: no)"),
    borrowck: String = ("migrate".to_string(), parse_string, [UNTRACKED],
        "select which borrowck is used (`mir` or `migrate`) (default: `migrate`)"),
    borrowck_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather borrowck statistics (default: no)"),
    cgu_partitioning_strategy: Option<String> = (None, parse_opt_string, [TRACKED],
        "the codegen unit partitioning strategy to use"),
    chalk: bool = (false, parse_bool, [TRACKED],
        "enable the experimental Chalk-based trait solving engine"),
    codegen_backend: Option<String> = (None, parse_opt_string, [TRACKED],
        "the backend to use"),
    combine_cgu: bool = (false, parse_bool, [TRACKED],
        "combine CGUs into a single one"),
    crate_attr: Vec<String> = (Vec::new(), parse_string_push, [TRACKED],
        "inject the given attribute in the crate"),
    debug_macros: bool = (false, parse_bool, [TRACKED],
        "emit line numbers debug info inside macros (default: no)"),
    deduplicate_diagnostics: bool = (true, parse_bool, [UNTRACKED],
        "deduplicate identical diagnostics (default: yes)"),
    dep_info_omit_d_target: bool = (false, parse_bool, [TRACKED],
        "in dep-info output, omit targets for tracking dependencies of the dep-info files \
        themselves (default: no)"),
    dep_tasks: bool = (false, parse_bool, [UNTRACKED],
        "print tasks that execute and the color their dep node gets (requires debug build) \
        (default: no)"),
    dont_buffer_diagnostics: bool = (false, parse_bool, [UNTRACKED],
        "emit diagnostics rather than buffering (breaks NLL error downgrading, sorting) \
        (default: no)"),
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
        "in addition to `.mir` files, create graphviz `.dot` files (and with \
        `-Z instrument-coverage`, also create a `.dot` file for the MIR-derived \
        coverage graph) (default: no)"),
    dump_mir_spanview: Option<MirSpanview> = (None, parse_mir_spanview, [UNTRACKED],
        "in addition to `.mir` files, create `.html` files to view spans for \
        all `statement`s (including terminators), only `terminator` spans, or \
        computed `block` spans (one span encompassing a block's terminator and \
        all statements). If `-Z instrument-coverage` is also enabled, create \
        an additional `.html` file showing the computed coverage spans."),
    emit_future_incompat_report: bool = (false, parse_bool, [UNTRACKED],
        "emits a future-incompatibility report for lints (RFC 2834)"),
    emit_stack_sizes: bool = (false, parse_bool, [UNTRACKED],
        "emit a section containing stack size metadata (default: no)"),
    fewer_names: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "reduce memory use by retaining fewer names within compilation artifacts (LLVM-IR) \
        (default: no)"),
    force_overflow_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "force overflow checks on or off"),
    force_unstable_if_unmarked: bool = (false, parse_bool, [TRACKED],
        "force all crates to be `rustc_private` unstable (default: no)"),
    fuel: Option<(String, u64)> = (None, parse_optimization_fuel, [TRACKED],
        "set the optimization fuel quota for a crate"),
    function_sections: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether each function should go in its own section"),
    graphviz_dark_mode: bool = (false, parse_bool, [UNTRACKED],
        "use dark-themed colors in graphviz output (default: no)"),
    graphviz_font: String = ("Courier, monospace".to_string(), parse_string, [UNTRACKED],
        "use the given `fontname` in graphviz output; can be overridden by setting \
        environment variable `RUSTC_GRAPHVIZ_FONT` (default: `Courier, monospace`)"),
    hir_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some statistics about AST and HIR (default: no)"),
    human_readable_cgu_names: bool = (false, parse_bool, [TRACKED],
        "generate human-readable, predictable names for codegen units (default: no)"),
    identify_regions: bool = (false, parse_bool, [UNTRACKED],
        "display unnamed regions as `'<id>`, using a non-ident unique id (default: no)"),
    incremental_ignore_spans: bool = (false, parse_bool, [UNTRACKED],
        "ignore spans during ICH computation -- used for testing (default: no)"),
    incremental_info: bool = (false, parse_bool, [UNTRACKED],
        "print high-level information about incremental reuse (or the lack thereof) \
        (default: no)"),
    incremental_verify_ich: bool = (false, parse_bool, [UNTRACKED],
        "verify incr. comp. hashes of green query instances (default: no)"),
    inline_mir_threshold: usize = (50, parse_uint, [TRACKED],
        "a default MIR inlining threshold (default: 50)"),
    inline_mir_hint_threshold: usize = (100, parse_uint, [TRACKED],
        "inlining threshold for functions with inline hint (default: 100)"),
    inline_in_all_cgus: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "control whether `#[inline]` functions are in all CGUs"),
    input_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather statistics about the input (default: no)"),
    insert_sideeffect: bool = (false, parse_bool, [TRACKED],
        "fix undefined behavior when a thread doesn't eventually make progress \
        (such as entering an empty infinite loop) by inserting llvm.sideeffect \
        (default: no)"),
    instrument_coverage: bool = (false, parse_bool, [TRACKED],
        "instrument the generated code to support LLVM source-based code coverage \
        reports (note, the compiler build config must include `profiler = true`, \
        and is mutually exclusive with `-C profile-generate`/`-C profile-use`); \
        implies `-Z symbol-mangling-version=v0`; disables/overrides some Rust \
        optimizations (default: no)"),
    instrument_mcount: bool = (false, parse_bool, [TRACKED],
        "insert function instrument code for mcount-based tracing (default: no)"),
    keep_hygiene_data: bool = (false, parse_bool, [UNTRACKED],
        "keep hygiene data after analysis (default: no)"),
    link_native_libraries: bool = (true, parse_bool, [UNTRACKED],
        "link native libraries in the linker invocation (default: yes)"),
    link_only: bool = (false, parse_bool, [TRACKED],
        "link the `.rlink` file generated by `-Z no-link` (default: no)"),
    llvm_time_trace: bool = (false, parse_bool, [UNTRACKED],
        "generate JSON tracing data file from LLVM data (default: no)"),
    ls: bool = (false, parse_bool, [UNTRACKED],
        "list the symbols defined by a library crate (default: no)"),
    macro_backtrace: bool = (false, parse_bool, [UNTRACKED],
        "show macro backtraces (default: no)"),
    merge_functions: Option<MergeFunctions> = (None, parse_merge_functions, [TRACKED],
        "control the operation of the MergeFunctions LLVM pass, taking \
        the same values as the target option of the same name"),
    meta_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather metadata statistics (default: no)"),
    mir_emit_retag: bool = (false, parse_bool, [TRACKED],
        "emit Retagging MIR statements, interpreted e.g., by miri; implies -Zmir-opt-level=0 \
        (default: no)"),
    mir_opt_level: usize = (1, parse_uint, [TRACKED],
        "MIR optimization level (0-3; default: 1)"),
    mutable_noalias: bool = (false, parse_bool, [TRACKED],
        "emit noalias metadata for mutable references (default: no)"),
    new_llvm_pass_manager: bool = (false, parse_bool, [TRACKED],
        "use new LLVM pass manager (default: no)"),
    nll_facts: bool = (false, parse_bool, [UNTRACKED],
        "dump facts from NLL analysis into side files (default: no)"),
    nll_facts_dir: String = ("nll-facts".to_string(), parse_string, [UNTRACKED],
        "the directory the NLL facts are dumped into (default: `nll-facts`)"),
    no_analysis: bool = (false, parse_no_flag, [UNTRACKED],
        "parse and expand the source, but run no analysis"),
    no_codegen: bool = (false, parse_no_flag, [TRACKED],
        "run all passes except codegen; no output"),
    no_generate_arange_section: bool = (false, parse_no_flag, [TRACKED],
        "omit DWARF address ranges that give faster lookups"),
    no_interleave_lints: bool = (false, parse_no_flag, [UNTRACKED],
        "execute lints separately; allows benchmarking individual lints"),
    no_leak_check: bool = (false, parse_no_flag, [UNTRACKED],
        "disable the 'leak check' for subtyping; unsound, but useful for tests"),
    no_link: bool = (false, parse_no_flag, [TRACKED],
        "compile without linking"),
    no_parallel_llvm: bool = (false, parse_no_flag, [UNTRACKED],
        "run LLVM in non-parallel mode (while keeping codegen-units and ThinLTO)"),
    no_profiler_runtime: bool = (false, parse_no_flag, [TRACKED],
        "prevent automatic injection of the profiler_builtins crate"),
    normalize_docs: bool = (false, parse_bool, [TRACKED],
        "normalize associated items in rustdoc when generating documentation"),
    osx_rpath_install_name: bool = (false, parse_bool, [TRACKED],
        "pass `-install_name @rpath/...` to the macOS linker (default: no)"),
    panic_abort_tests: bool = (false, parse_bool, [TRACKED],
        "support compiling tests with panic=abort (default: no)"),
    parse_only: bool = (false, parse_bool, [UNTRACKED],
        "parse only; do not compile, assemble, or link (default: no)"),
    perf_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some performance-related statistics (default: no)"),
    plt: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether to use the PLT when calling into shared libraries;
        only has effect for PIC code on systems with ELF binaries
        (default: PLT is disabled if full relro is enabled)"),
    polonius: bool = (false, parse_bool, [TRACKED],
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
    print_fuel: Option<String> = (None, parse_opt_string, [TRACKED],
        "make rustc print the total optimization fuel used by a crate"),
    print_link_args: bool = (false, parse_bool, [UNTRACKED],
        "print the arguments passed to the linker (default: no)"),
    print_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "print the LLVM optimization passes being run (default: no)"),
    print_mono_items: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "print the result of the monomorphization collection pass"),
    print_type_sizes: bool = (false, parse_bool, [UNTRACKED],
        "print layout information for each type encountered (default: no)"),
    proc_macro_backtrace: bool = (false, parse_bool, [UNTRACKED],
         "show backtraces for panics during proc-macro execution (default: no)"),
    profile: bool = (false, parse_bool, [TRACKED],
        "insert profiling code (default: no)"),
    profile_emit: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "file path to emit profiling data at runtime when using 'profile' \
        (default based on relative source path)"),
    query_dep_graph: bool = (false, parse_bool, [UNTRACKED],
        "enable queries of the dependency graph for regression testing (default: no)"),
    query_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some statistics about the query system (default: no)"),
    relax_elf_relocations: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "whether ELF relocations can be relaxed"),
    relro_level: Option<RelroLevel> = (None, parse_relro_level, [TRACKED],
        "choose which RELRO level to use"),
    report_delayed_bugs: bool = (false, parse_bool, [TRACKED],
        "immediately print bugs registered with `delay_span_bug` (default: no)"),
    // The default historical behavior was to always run dsymutil, so we're
    // preserving that temporarily, but we're likely to switch the default
    // soon.
    run_dsymutil: bool = (true, parse_bool, [TRACKED],
        "if on Mac, run `dsymutil` and delete intermediate object files (default: yes)"),
    sanitizer: SanitizerSet = (SanitizerSet::empty(), parse_sanitizers, [TRACKED],
        "use a sanitizer"),
    sanitizer_memory_track_origins: usize = (0, parse_sanitizer_memory_track_origins, [TRACKED],
        "enable origins tracking in MemorySanitizer"),
    sanitizer_recover: SanitizerSet = (SanitizerSet::empty(), parse_sanitizers, [TRACKED],
        "enable recovery for selected sanitizers"),
    saturating_float_casts: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "make float->int casts UB-free: numbers outside the integer type's range are clipped to \
        the max/min integer respectively, and NaN is mapped to 0 (default: yes)"),
    save_analysis: bool = (false, parse_bool, [UNTRACKED],
        "write syntax and type analysis (in JSON format) information, in \
        addition to normal output (default: no)"),
    self_profile: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [UNTRACKED],
        "run the self profiler and output the raw event data"),
    // keep this in sync with the event filter names in librustc_data_structures/profiling.rs
    self_profile_events: Option<Vec<String>> = (None, parse_opt_comma_list, [UNTRACKED],
        "specify the events recorded by the self profiler;
        for example: `-Z self-profile-events=default,query-keys`
        all options: none, all, default, generic-activity, query-provider, query-cache-hit
                     query-blocked, incr-cache-load, query-keys, function-args, args, llvm"),
    share_generics: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "make the current crate share its generic instantiations"),
    show_span: Option<String> = (None, parse_opt_string, [TRACKED],
        "show spans for compiler debugging (expr|pat|ty)"),
    span_debug: bool = (false, parse_bool, [UNTRACKED],
        "forward proc_macro::Span's `Debug` impl to `Span`"),
    // o/w tests have closure@path
    span_free_formats: bool = (false, parse_bool, [UNTRACKED],
        "exclude spans when debug-printing compiler state (default: no)"),
    src_hash_algorithm: Option<SourceFileHashAlgorithm> = (None, parse_src_file_hash, [TRACKED],
        "hash algorithm of source files in debug info (`md5`, `sha1`, or `sha256`)"),
    strip: Strip = (Strip::None, parse_strip, [UNTRACKED],
        "tell the linker which information to strip (`none` (default), `debuginfo` or `symbols`)"),
    symbol_mangling_version: SymbolManglingVersion = (SymbolManglingVersion::Legacy,
        parse_symbol_mangling_version, [TRACKED],
        "which mangling version to use for symbol names"),
    teach: bool = (false, parse_bool, [TRACKED],
        "show extended diagnostic help (default: no)"),
    terminal_width: Option<usize> = (None, parse_opt_uint, [UNTRACKED],
        "set the current terminal width"),
    tune_cpu: Option<String> = (None, parse_opt_string, [TRACKED],
        "select processor to schedule for (`rustc --print target-cpus` for details)"),
    thinlto: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable ThinLTO when possible"),
    // We default to 1 here since we want to behave like
    // a sequential compiler for now. This'll likely be adjusted
    // in the future. Note that -Zthreads=0 is the way to get
    // the num_cpus behavior.
    threads: usize = (1, parse_threads, [UNTRACKED],
        "use a thread pool with N threads"),
    time: bool = (false, parse_bool, [UNTRACKED],
        "measure time of rustc processes (default: no)"),
    time_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each LLVM pass (default: no)"),
    time_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each rustc pass (default: no)"),
    tls_model: Option<TlsModel> = (None, parse_tls_model, [TRACKED],
        "choose the TLS model to use (`rustc --print tls-models` for details)"),
    trace_macros: bool = (false, parse_bool, [UNTRACKED],
        "for every macro invocation, print its name and arguments (default: no)"),
    trap_unreachable: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "generate trap instructions for unreachable intrinsics (default: use target setting, usually yes)"),
    treat_err_as_bug: Option<usize> = (None, parse_treat_err_as_bug, [TRACKED],
        "treat error number `val` that occurs as bug"),
    trim_diagnostic_paths: bool = (true, parse_bool, [UNTRACKED],
        "in diagnostics, use heuristics to shorten paths referring to items"),
    ui_testing: bool = (false, parse_bool, [UNTRACKED],
        "emit compiler diagnostics in a form suitable for UI testing (default: no)"),
    unleash_the_miri_inside_of_you: bool = (false, parse_bool, [TRACKED],
        "take the brakes off const evaluation. NOTE: this is unsound (default: no)"),
    unpretty: Option<String> = (None, parse_unpretty, [UNTRACKED],
        "present the input source, unstable (and less-pretty) variants;
        valid types are any of the types for `--pretty`, as well as:
        `expanded`, `expanded,identified`,
        `expanded,hygiene` (with internal representations),
        `everybody_loops` (all function bodies replaced with `loop {}`),
        `hir` (the HIR), `hir,identified`,
        `hir,typed` (HIR with types for each node),
        `hir-tree` (dump the raw HIR),
        `mir` (the MIR), or `mir-cfg` (graphviz formatted MIR)"),
    unsound_mir_opts: bool = (false, parse_bool, [TRACKED],
        "enable unsound and buggy MIR optimizations (default: no)"),
    unstable_options: bool = (false, parse_bool, [UNTRACKED],
        "adds unstable command line options to rustc interface (default: no)"),
    use_ctors_section: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "use legacy .ctors section for initializers rather than .init_array"),
    validate_mir: bool = (false, parse_bool, [UNTRACKED],
        "validate MIR after each transformation"),
    verbose: bool = (false, parse_bool, [UNTRACKED],
        "in general, enable more debug printouts (default: no)"),
    verify_llvm_ir: bool = (false, parse_bool, [TRACKED],
        "verify LLVM IR (default: no)"),

    // This list is in alphabetical order.
    //
    // If you add a new option, please update:
    // - src/librustc_interface/tests.rs
}
