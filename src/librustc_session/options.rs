use crate::config::*;

use crate::early_error;
use crate::lint;
use crate::search_paths::SearchPath;
use crate::utils::NativeLibraryKind;

use rustc_target::spec::TargetTriple;
use rustc_target::spec::{LinkerFlavor, MergeFunctions, PanicStrategy, RelroLevel};

use rustc_feature::UnstableFeatures;
use rustc_span::edition::Edition;

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
        libs: Vec<(String, Option<String>, Option<NativeLibraryKind>)> [TRACKED],
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
            for &(candidate, setter, opt_type_desc, _) in $stat {
                if option_to_lookup != candidate { continue }
                if !setter(&mut op, value) {
                    match (value, opt_type_desc) {
                        (Some(..), None) => {
                            early_error(error_format, &format!("{} option `{}` takes no \
                                                                value", $outputname, key))
                        }
                        (None, Some(type_desc)) => {
                            early_error(error_format, &format!("{0} option `{1}` requires \
                                                                {2} ({3} {1}=<value>)",
                                                               $outputname, key,
                                                               type_desc, $prefix))
                        }
                        (Some(value), Some(type_desc)) => {
                            early_error(error_format, &format!("incorrect value `{}` for {} \
                                                                option `{}` - {} was expected",
                                                               value, $outputname,
                                                               key, type_desc))
                        }
                        (None, None) => panic!()
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
    pub const $stat: &[(&str, $setter_name, Option<&str>, &str)] =
        &[ $( (stringify!($opt), $mod_set::$opt, $mod_desc::$parse, $desc) ),* ];

    #[allow(non_upper_case_globals, dead_code)]
    mod $mod_desc {
        pub const parse_bool: Option<&str> = None;
        pub const parse_opt_bool: Option<&str> =
            Some("one of: `y`, `yes`, `on`, `n`, `no`, or `off`");
        pub const parse_string: Option<&str> = Some("a string");
        pub const parse_string_push: Option<&str> = Some("a string");
        pub const parse_pathbuf_push: Option<&str> = Some("a path");
        pub const parse_opt_string: Option<&str> = Some("a string");
        pub const parse_opt_pathbuf: Option<&str> = Some("a path");
        pub const parse_list: Option<&str> = Some("a space-separated list of strings");
        pub const parse_opt_list: Option<&str> = Some("a space-separated list of strings");
        pub const parse_opt_comma_list: Option<&str> = Some("a comma-separated list of strings");
        pub const parse_threads: Option<&str> = Some("a number");
        pub const parse_uint: Option<&str> = Some("a number");
        pub const parse_passes: Option<&str> =
            Some("a space-separated list of passes, or `all`");
        pub const parse_opt_uint: Option<&str> =
            Some("a number");
        pub const parse_panic_strategy: Option<&str> =
            Some("either `unwind` or `abort`");
        pub const parse_relro_level: Option<&str> =
            Some("one of: `full`, `partial`, or `off`");
        pub const parse_sanitizer: Option<&str> =
            Some("one of: `address`, `leak`, `memory` or `thread`");
        pub const parse_sanitizer_list: Option<&str> =
            Some("comma separated list of sanitizers");
        pub const parse_sanitizer_memory_track_origins: Option<&str> = None;
        pub const parse_cfguard: Option<&str> =
            Some("either `disabled`, `nochecks`, or `checks`");
        pub const parse_linker_flavor: Option<&str> =
            Some(::rustc_target::spec::LinkerFlavor::one_of());
        pub const parse_optimization_fuel: Option<&str> =
            Some("crate=integer");
        pub const parse_unpretty: Option<&str> =
            Some("`string` or `string=string`");
        pub const parse_treat_err_as_bug: Option<&str> =
            Some("either no value or a number bigger than 0");
        pub const parse_lto: Option<&str> =
            Some("either a boolean (`yes`, `no`, `on`, `off`, etc), `thin`, \
                  `fat`, or omitted");
        pub const parse_linker_plugin_lto: Option<&str> =
            Some("either a boolean (`yes`, `no`, `on`, `off`, etc), \
                  or the path to the linker plugin");
        pub const parse_switch_with_opt_path: Option<&str> =
            Some("an optional path to the profiling data output directory");
        pub const parse_merge_functions: Option<&str> =
            Some("one of: `disabled`, `trampolines`, or `aliases`");
        pub const parse_symbol_mangling_version: Option<&str> =
            Some("either `legacy` or `v0` (RFC 2603)");
    }

    #[allow(dead_code)]
    mod $mod_set {
        use super::{$struct_name, Passes, Sanitizer, LtoCli, LinkerPluginLto, SwitchWithOptPath,
            SymbolManglingVersion, CFGuard};
        use rustc_target::spec::{LinkerFlavor, MergeFunctions, PanicStrategy, RelroLevel};
        use std::path::PathBuf;
        use std::str::FromStr;

        $(
            pub fn $opt(cg: &mut $struct_name, v: Option<&str>) -> bool {
                $parse(&mut cg.$opt, v)
            }
        )*

        fn parse_bool(slot: &mut bool, v: Option<&str>) -> bool {
            match v {
                Some(..) => false,
                None => { *slot = true; true }
            }
        }

        fn parse_opt_bool(slot: &mut Option<bool>, v: Option<&str>) -> bool {
            match v {
                Some(s) => {
                    match s {
                        "n" | "no" | "off" => {
                            *slot = Some(false);
                        }
                        "y" | "yes" | "on" => {
                            *slot = Some(true);
                        }
                        _ => { return false; }
                    }

                    true
                },
                None => { *slot = Some(true); true }
            }
        }

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

        fn parse_string(slot: &mut String, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = s.to_string(); true },
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

        fn parse_uint(slot: &mut usize, v: Option<&str>) -> bool {
            match v.and_then(|s| s.parse().ok()) {
                Some(i) => { *slot = i; true },
                None => false
            }
        }

        fn parse_opt_uint(slot: &mut Option<usize>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = s.parse().ok(); slot.is_some() }
                None => { *slot = None; false }
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

        fn parse_sanitizer(slot: &mut Option<Sanitizer>, v: Option<&str>) -> bool {
            if let Some(Ok(s)) =  v.map(str::parse) {
                *slot = Some(s);
                true
            } else {
                false
            }
        }

        fn parse_sanitizer_list(slot: &mut Vec<Sanitizer>, v: Option<&str>) -> bool {
            if let Some(v) = v {
                for s in v.split(',').map(str::parse) {
                    if let Ok(s) = s {
                        if !slot.contains(&s) {
                            slot.push(s);
                        }
                    } else {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }

        fn parse_sanitizer_memory_track_origins(slot: &mut usize, v: Option<&str>) -> bool {
            match v.map(|s| s.parse()) {
                None => {
                    *slot = 2;
                    true
                }
                Some(Ok(i)) if i <= 2 => {
                    *slot = i;
                    true
                }
                _ => {
                    false
                }
            }
        }

        fn parse_cfguard(slot: &mut CFGuard, v: Option<&str>) -> bool {
            match v {
                Some("disabled") => *slot = CFGuard::Disabled,
                Some("nochecks") => *slot = CFGuard::NoChecks,
                Some("checks") => *slot = CFGuard::Checks,
                _ => return false,
            }
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
    }
) }

options! {CodegenOptions, CodegenSetter, basic_codegen_options,
          build_codegen_options, "C", "codegen",
          CG_OPTIONS, cg_type_desc, cgsetters,
    ar: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "this option is deprecated and does nothing"),
    linker: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "system linker to link outputs with"),
    link_arg: Vec<String> = (vec![], parse_string_push, [UNTRACKED],
        "a single extra argument to append to the linker invocation (can be used several times)"),
    link_args: Option<Vec<String>> = (None, parse_opt_list, [UNTRACKED],
        "extra arguments to append to the linker invocation (space separated)"),
    link_dead_code: bool = (false, parse_bool, [UNTRACKED],
        "don't let linker strip dead code (turning it on can be used for code coverage)"),
    lto: LtoCli = (LtoCli::Unspecified, parse_lto, [TRACKED],
        "perform LLVM link-time optimizations"),
    target_cpu: Option<String> = (None, parse_opt_string, [TRACKED],
        "select target processor (`rustc --print target-cpus` for details)"),
    target_feature: String = (String::new(), parse_string, [TRACKED],
        "target specific attributes. (`rustc --print target-features` for details). \
        This feature is unsafe."),
    passes: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of extra LLVM passes to run (space separated)"),
    llvm_args: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of arguments to pass to LLVM (space separated)"),
    save_temps: bool = (false, parse_bool, [UNTRACKED],
        "save all temporary output files during compilation"),
    rpath: bool = (false, parse_bool, [UNTRACKED],
        "set rpath values in libs/exes"),
    overflow_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "use overflow checks for integer arithmetic"),
    no_prepopulate_passes: bool = (false, parse_bool, [TRACKED],
        "don't pre-populate the pass manager with a list of passes"),
    no_vectorize_loops: bool = (false, parse_bool, [TRACKED],
        "don't run the loop vectorization optimization passes"),
    no_vectorize_slp: bool = (false, parse_bool, [TRACKED],
        "don't run LLVM's SLP vectorization pass"),
    soft_float: bool = (false, parse_bool, [TRACKED],
        "use soft float ABI (*eabihf targets only)"),
    prefer_dynamic: bool = (false, parse_bool, [TRACKED],
        "prefer dynamic linking to static linking"),
    no_integrated_as: bool = (false, parse_bool, [TRACKED],
        "use an external assembler rather than LLVM's integrated one"),
    no_redzone: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "disable the use of the redzone"),
    relocation_model: Option<String> = (None, parse_opt_string, [TRACKED],
        "choose the relocation model to use (`rustc --print relocation-models` for details)"),
    code_model: Option<String> = (None, parse_opt_string, [TRACKED],
        "choose the code model to use (`rustc --print code-models` for details)"),
    metadata: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "metadata to mangle symbol names with"),
    extra_filename: String = (String::new(), parse_string, [UNTRACKED],
        "extra data to put in each output filename"),
    codegen_units: Option<usize> = (None, parse_opt_uint, [UNTRACKED],
        "divide crate into N units to optimize in parallel"),
    remark: Passes = (Passes::Some(Vec::new()), parse_passes, [UNTRACKED],
        "print remarks for these optimization passes (space separated, or \"all\")"),
    no_stack_check: bool = (false, parse_bool, [UNTRACKED],
        "the `--no-stack-check` flag is deprecated and does nothing"),
    debuginfo: Option<usize> = (None, parse_opt_uint, [TRACKED],
        "debug info emission level, 0 = no debug info, 1 = line tables only, \
         2 = full debug info with variable and type information"),
    opt_level: Option<String> = (None, parse_opt_string, [TRACKED],
        "optimize with possible levels 0-3, s, or z"),
    force_frame_pointers: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "force use of the frame pointers"),
    debug_assertions: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "explicitly enable the `cfg(debug_assertions)` directive"),
    inline_threshold: Option<usize> = (None, parse_opt_uint, [TRACKED],
        "set the threshold for inlining a function (default: 225)"),
    panic: Option<PanicStrategy> = (None, parse_panic_strategy,
        [TRACKED], "panic strategy to compile crate with"),
    incremental: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "enable incremental compilation"),
    default_linker_libraries: Option<bool> = (None, parse_opt_bool, [UNTRACKED],
        "allow the linker to link its default libraries"),
    linker_flavor: Option<LinkerFlavor> = (None, parse_linker_flavor, [UNTRACKED],
                                           "linker flavor"),
    linker_plugin_lto: LinkerPluginLto = (LinkerPluginLto::Disabled,
        parse_linker_plugin_lto, [TRACKED],
        "generate build artifacts that are compatible with linker-based LTO."),
    profile_generate: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [TRACKED],
        "compile the program with profiling instrumentation"),
    profile_use: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "use the given `.profdata` file for profile-guided optimization"),
}

options! {DebuggingOptions, DebuggingSetter, basic_debugging_options,
          build_debugging_options, "Z", "debugging",
          DB_OPTIONS, db_type_desc, dbsetters,
    codegen_backend: Option<String> = (None, parse_opt_string, [TRACKED],
        "the backend to use"),
    verbose: bool = (false, parse_bool, [UNTRACKED],
        "in general, enable more debug printouts"),
    span_free_formats: bool = (false, parse_bool, [UNTRACKED],
        "when debug-printing compiler state, do not include spans"), // o/w tests have closure@path
    identify_regions: bool = (false, parse_bool, [UNTRACKED],
        "make unnamed regions display as '# (where # is some non-ident unique id)"),
    borrowck: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "select which borrowck is used (`mir` or `migrate`)"),
    time_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each rustc pass"),
    time: bool = (false, parse_bool, [UNTRACKED],
        "measure time of rustc processes"),
    time_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each LLVM pass"),
    llvm_time_trace: bool = (false, parse_bool, [UNTRACKED],
        "generate JSON tracing data file from LLVM data"),
    input_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather statistics about the input"),
    asm_comments: bool = (false, parse_bool, [TRACKED],
        "generate comments into the assembly (may change behavior)"),
    verify_llvm_ir: bool = (false, parse_bool, [TRACKED],
        "verify LLVM IR"),
    borrowck_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather borrowck statistics"),
    no_landing_pads: bool = (false, parse_bool, [TRACKED],
        "omit landing pads for unwinding"),
    fewer_names: bool = (false, parse_bool, [TRACKED],
        "reduce memory use by retaining fewer names within compilation artifacts (LLVM-IR)"),
    meta_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather metadata statistics"),
    print_link_args: bool = (false, parse_bool, [UNTRACKED],
        "print the arguments passed to the linker"),
    print_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "prints the LLVM optimization passes being run"),
    ast_json: bool = (false, parse_bool, [UNTRACKED],
        "print the AST as JSON and halt"),
    // We default to 1 here since we want to behave like
    // a sequential compiler for now. This'll likely be adjusted
    // in the future. Note that -Zthreads=0 is the way to get
    // the num_cpus behavior.
    threads: usize = (1, parse_threads, [UNTRACKED],
        "use a thread pool with N threads"),
    ast_json_noexpand: bool = (false, parse_bool, [UNTRACKED],
        "print the pre-expansion AST as JSON and halt"),
    ls: bool = (false, parse_bool, [UNTRACKED],
        "list the symbols defined by a library crate"),
    save_analysis: bool = (false, parse_bool, [UNTRACKED],
        "write syntax and type analysis (in JSON format) information, in \
         addition to normal output"),
    print_region_graph: bool = (false, parse_bool, [UNTRACKED],
        "prints region inference graph. \
         Use with RUST_REGION_GRAPH=help for more info"),
    parse_only: bool = (false, parse_bool, [UNTRACKED],
        "parse only; do not compile, assemble, or link"),
    dual_proc_macros: bool = (false, parse_bool, [TRACKED],
        "load proc macros for both target and host, but only link to the target"),
    no_codegen: bool = (false, parse_bool, [TRACKED],
        "run all passes except codegen; no output"),
    treat_err_as_bug: Option<usize> = (None, parse_treat_err_as_bug, [TRACKED],
        "treat error number `val` that occurs as bug"),
    report_delayed_bugs: bool = (false, parse_bool, [TRACKED],
        "immediately print bugs registered with `delay_span_bug`"),
    external_macro_backtrace: bool = (false, parse_bool, [UNTRACKED],
        "show macro backtraces even for non-local macros"),
    teach: bool = (false, parse_bool, [TRACKED],
        "show extended diagnostic help"),
    terminal_width: Option<usize> = (None, parse_opt_uint, [UNTRACKED],
        "set the current terminal width"),
    panic_abort_tests: bool = (false, parse_bool, [TRACKED],
        "support compiling tests with panic=abort"),
    dep_tasks: bool = (false, parse_bool, [UNTRACKED],
        "print tasks that execute and the color their dep node gets (requires debug build)"),
    incremental: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "enable incremental compilation (experimental)"),
    incremental_queries: bool = (true, parse_bool, [UNTRACKED],
        "enable incremental compilation support for queries (experimental)"),
    incremental_info: bool = (false, parse_bool, [UNTRACKED],
        "print high-level information about incremental reuse (or the lack thereof)"),
    incremental_dump_hash: bool = (false, parse_bool, [UNTRACKED],
        "dump hash information in textual format to stdout"),
    incremental_verify_ich: bool = (false, parse_bool, [UNTRACKED],
        "verify incr. comp. hashes of green query instances"),
    incremental_ignore_spans: bool = (false, parse_bool, [UNTRACKED],
        "ignore spans during ICH computation -- used for testing"),
    instrument_mcount: bool = (false, parse_bool, [TRACKED],
        "insert function instrument code for mcount-based tracing"),
    dump_dep_graph: bool = (false, parse_bool, [UNTRACKED],
        "dump the dependency graph to $RUST_DEP_GRAPH (default: /tmp/dep_graph.gv)"),
    query_dep_graph: bool = (false, parse_bool, [UNTRACKED],
        "enable queries of the dependency graph for regression testing"),
    no_analysis: bool = (false, parse_bool, [UNTRACKED],
        "parse and expand the source, but run no analysis"),
    unstable_options: bool = (false, parse_bool, [UNTRACKED],
        "adds unstable command line options to rustc interface"),
    force_overflow_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "force overflow checks on or off"),
    trace_macros: bool = (false, parse_bool, [UNTRACKED],
        "for every macro invocation, print its name and arguments"),
    debug_macros: bool = (false, parse_bool, [TRACKED],
        "emit line numbers debug info inside macros"),
    generate_arange_section: bool = (true, parse_bool, [TRACKED],
        "generate DWARF address ranges for faster lookups"),
    keep_hygiene_data: bool = (false, parse_bool, [UNTRACKED],
        "don't clear the hygiene data after analysis"),
    keep_ast: bool = (false, parse_bool, [UNTRACKED],
        "keep the AST after lowering it to HIR"),
    show_span: Option<String> = (None, parse_opt_string, [TRACKED],
        "show spans for compiler debugging (expr|pat|ty)"),
    print_type_sizes: bool = (false, parse_bool, [UNTRACKED],
        "print layout information for each type encountered"),
    print_mono_items: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "print the result of the monomorphization collection pass"),
    mir_opt_level: usize = (1, parse_uint, [TRACKED],
        "set the MIR optimization level (0-3, default: 1)"),
    mutable_noalias: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "emit noalias metadata for mutable references (default: no)"),
    dump_mir: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "dump MIR state to file.
        `val` is used to select which passes and functions to dump. For example:
        `all` matches all passes and functions,
        `foo` matches all passes for functions whose name contains 'foo',
        `foo & ConstProp` only the 'ConstProp' pass for function names containing 'foo',
        `foo | bar` all passes for function names containing 'foo' or 'bar'."),

    dump_mir_dir: String = (String::from("mir_dump"), parse_string, [UNTRACKED],
        "the directory the MIR is dumped into"),
    dump_mir_graphviz: bool = (false, parse_bool, [UNTRACKED],
        "in addition to `.mir` files, create graphviz `.dot` files"),
    dump_mir_exclude_pass_number: bool = (false, parse_bool, [UNTRACKED],
        "if set, exclude the pass number when dumping MIR (used in tests)"),
    mir_emit_retag: bool = (false, parse_bool, [TRACKED],
        "emit Retagging MIR statements, interpreted e.g., by miri; implies -Zmir-opt-level=0"),
    perf_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some performance-related statistics"),
    query_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some statistics about the query system"),
    hir_stats: bool = (false, parse_bool, [UNTRACKED],
        "print some statistics about AST and HIR"),
    always_encode_mir: bool = (false, parse_bool, [TRACKED],
        "encode MIR of all functions into the crate metadata"),
    json_rendered: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "describes how to render the `rendered` field of json diagnostics"),
    unleash_the_miri_inside_of_you: bool = (false, parse_bool, [TRACKED],
        "take the breaks off const evaluation. NOTE: this is unsound"),
    osx_rpath_install_name: bool = (false, parse_bool, [TRACKED],
        "pass `-install_name @rpath/...` to the macOS linker"),
    sanitizer: Option<Sanitizer> = (None, parse_sanitizer, [TRACKED],
                                    "use a sanitizer"),
    sanitizer_recover: Vec<Sanitizer> = (vec![], parse_sanitizer_list, [TRACKED],
        "Enable recovery for selected sanitizers"),
    sanitizer_memory_track_origins: usize = (0, parse_sanitizer_memory_track_origins, [TRACKED],
        "Enable origins tracking in MemorySanitizer"),
    fuel: Option<(String, u64)> = (None, parse_optimization_fuel, [TRACKED],
        "set the optimization fuel quota for a crate"),
    print_fuel: Option<String> = (None, parse_opt_string, [TRACKED],
        "make rustc print the total optimization fuel used by a crate"),
    force_unstable_if_unmarked: bool = (false, parse_bool, [TRACKED],
        "force all crates to be `rustc_private` unstable"),
    pre_link_arg: Vec<String> = (vec![], parse_string_push, [UNTRACKED],
        "a single extra argument to prepend the linker invocation (can be used several times)"),
    pre_link_args: Option<Vec<String>> = (None, parse_opt_list, [UNTRACKED],
        "extra arguments to prepend to the linker invocation (space separated)"),
    profile: bool = (false, parse_bool, [TRACKED],
                     "insert profiling code"),
    relro_level: Option<RelroLevel> = (None, parse_relro_level, [TRACKED],
        "choose which RELRO level to use"),
    nll_facts: bool = (false, parse_bool, [UNTRACKED],
                       "dump facts from NLL analysis into side files"),
    dont_buffer_diagnostics: bool = (false, parse_bool, [UNTRACKED],
        "emit diagnostics rather than buffering (breaks NLL error downgrading, sorting)."),
    polonius: bool = (false, parse_bool, [UNTRACKED],
        "enable polonius-based borrow-checker"),
    codegen_time_graph: bool = (false, parse_bool, [UNTRACKED],
        "generate a graphical HTML report of time spent in codegen and LLVM"),
    thinlto: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "enable ThinLTO when possible"),
    inline_in_all_cgus: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "control whether `#[inline]` functions are in all CGUs"),
    tls_model: Option<String> = (None, parse_opt_string, [TRACKED],
        "choose the TLS model to use (`rustc --print tls-models` for details)"),
    saturating_float_casts: bool = (false, parse_bool, [TRACKED],
        "make float->int casts UB-free: numbers outside the integer type's range are clipped to \
         the max/min integer respectively, and NaN is mapped to 0"),
    human_readable_cgu_names: bool = (false, parse_bool, [TRACKED],
        "generate human-readable, predictable names for codegen units"),
    dep_info_omit_d_target: bool = (false, parse_bool, [TRACKED],
        "in dep-info output, omit targets for tracking dependencies of the dep-info files \
         themselves"),
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
    run_dsymutil: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "run `dsymutil` and delete intermediate object files"),
    ui_testing: Option<bool> = (None, parse_opt_bool, [UNTRACKED],
        "format compiler diagnostics in a way that's better suitable for UI testing"),
    embed_bitcode: bool = (false, parse_bool, [TRACKED],
        "embed LLVM bitcode in object files"),
    strip_debuginfo_if_disabled: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "tell the linker to strip debuginfo when building without debuginfo enabled."),
    share_generics: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "make the current crate share its generic instantiations"),
    chalk: bool = (false, parse_bool, [TRACKED],
        "enable the experimental Chalk-based trait solving engine"),
    no_parallel_llvm: bool = (false, parse_bool, [UNTRACKED],
        "don't run LLVM in parallel (while keeping codegen-units and ThinLTO)"),
    no_leak_check: bool = (false, parse_bool, [UNTRACKED],
        "disables the 'leak check' for subtyping; unsound, but useful for tests"),
    no_interleave_lints: bool = (false, parse_bool, [UNTRACKED],
        "don't interleave execution of lints; allows benchmarking individual lints"),
    crate_attr: Vec<String> = (Vec::new(), parse_string_push, [TRACKED],
        "inject the given attribute in the crate"),
    self_profile: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [UNTRACKED],
        "run the self profiler and output the raw event data"),
    // keep this in sync with the event filter names in librustc_data_structures/profiling.rs
    self_profile_events: Option<Vec<String>> = (None, parse_opt_comma_list, [UNTRACKED],
        "specifies which kinds of events get recorded by the self profiler;
        for example: `-Z self-profile-events=default,query-keys`
        all options: none, all, default, generic-activity, query-provider, query-cache-hit
                     query-blocked, incr-cache-load, query-keys"),
    emit_stack_sizes: bool = (false, parse_bool, [UNTRACKED],
        "emits a section containing stack size metadata"),
    plt: Option<bool> = (None, parse_opt_bool, [TRACKED],
          "whether to use the PLT when calling into shared libraries;
          only has effect for PIC code on systems with ELF binaries
          (default: PLT is disabled if full relro is enabled)"),
    merge_functions: Option<MergeFunctions> = (None, parse_merge_functions, [TRACKED],
        "control the operation of the MergeFunctions LLVM pass, taking
         the same values as the target option of the same name"),
    allow_features: Option<Vec<String>> = (None, parse_opt_comma_list, [TRACKED],
        "only allow the listed language features to be enabled in code (space separated)"),
    symbol_mangling_version: SymbolManglingVersion = (SymbolManglingVersion::Legacy,
        parse_symbol_mangling_version, [TRACKED],
        "which mangling version to use for symbol names"),
    binary_dep_depinfo: bool = (false, parse_bool, [TRACKED],
        "include artifacts (sysroot, crate dependencies) used during compilation in dep-info"),
    insert_sideeffect: bool = (false, parse_bool, [TRACKED],
        "fix undefined behavior when a thread doesn't eventually make progress \
         (such as entering an empty infinite loop) by inserting llvm.sideeffect"),
    deduplicate_diagnostics: Option<bool> = (None, parse_opt_bool, [UNTRACKED],
        "deduplicate identical diagnostics"),
    control_flow_guard: CFGuard = (CFGuard::Disabled, parse_cfguard, [UNTRACKED],
        "use Windows Control Flow Guard (`disabled`, `nochecks` or `checks`)"),
    no_link: bool = (false, parse_bool, [TRACKED],
        "compile without linking"),
}
