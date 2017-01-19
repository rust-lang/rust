// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Contains infrastructure for configuring the compiler, including parsing
//! command line options.

pub use self::EntryFnType::*;
pub use self::CrateType::*;
pub use self::Passes::*;
pub use self::DebugInfoLevel::*;

use session::{early_error, early_warn, Session};
use session::search_paths::SearchPaths;

use rustc_back::PanicStrategy;
use rustc_back::target::Target;
use lint;
use middle::cstore;

use syntax::ast::{self, IntTy, UintTy};
use syntax::parse::token;
use syntax::parse;
use syntax::symbol::Symbol;
use syntax::feature_gate::UnstableFeatures;

use errors::{ColorConfig, FatalError, Handler};

use getopts;
use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::Iter as BTreeMapIter;
use std::collections::btree_map::Keys as BTreeMapKeysIter;
use std::collections::btree_map::Values as BTreeMapValuesIter;

use std::fmt;
use std::hash::Hasher;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::path::PathBuf;

pub struct Config {
    pub target: Target,
    pub int_type: IntTy,
    pub uint_type: UintTy,
}

#[derive(Clone, Copy, PartialEq, Hash)]
pub enum OptLevel {
    No, // -O0
    Less, // -O1
    Default, // -O2
    Aggressive, // -O3
    Size, // -Os
    SizeMin, // -Oz
}

#[derive(Clone, Copy, PartialEq, Hash)]
pub enum DebugInfoLevel {
    NoDebugInfo,
    LimitedDebugInfo,
    FullDebugInfo,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord,
         RustcEncodable, RustcDecodable)]
pub enum OutputType {
    Bitcode,
    Assembly,
    LlvmAssembly,
    Metadata,
    Object,
    Exe,
    DepInfo,
}

impl OutputType {
    fn is_compatible_with_codegen_units_and_single_output_file(&self) -> bool {
        match *self {
            OutputType::Exe |
            OutputType::DepInfo => true,
            OutputType::Bitcode |
            OutputType::Assembly |
            OutputType::LlvmAssembly |
            OutputType::Object |
            OutputType::Metadata => false,
        }
    }

    fn shorthand(&self) -> &'static str {
        match *self {
            OutputType::Bitcode => "llvm-bc",
            OutputType::Assembly => "asm",
            OutputType::LlvmAssembly => "llvm-ir",
            OutputType::Object => "obj",
            OutputType::Metadata => "metadata",
            OutputType::Exe => "link",
            OutputType::DepInfo => "dep-info",
        }
    }

    pub fn extension(&self) -> &'static str {
        match *self {
            OutputType::Bitcode => "bc",
            OutputType::Assembly => "s",
            OutputType::LlvmAssembly => "ll",
            OutputType::Object => "o",
            OutputType::Metadata => "rmeta",
            OutputType::DepInfo => "d",
            OutputType::Exe => "",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorOutputType {
    HumanReadable(ColorConfig),
    Json,
}

impl Default for ErrorOutputType {
    fn default() -> ErrorOutputType {
        ErrorOutputType::HumanReadable(ColorConfig::Auto)
    }
}

// Use tree-based collections to cheaply get a deterministic Hash implementation.
// DO NOT switch BTreeMap out for an unsorted container type! That would break
// dependency tracking for commandline arguments.
#[derive(Clone, Hash)]
pub struct OutputTypes(BTreeMap<OutputType, Option<PathBuf>>);

impl OutputTypes {
    pub fn new(entries: &[(OutputType, Option<PathBuf>)]) -> OutputTypes {
        OutputTypes(BTreeMap::from_iter(entries.iter()
                                               .map(|&(k, ref v)| (k, v.clone()))))
    }

    pub fn get(&self, key: &OutputType) -> Option<&Option<PathBuf>> {
        self.0.get(key)
    }

    pub fn contains_key(&self, key: &OutputType) -> bool {
        self.0.contains_key(key)
    }

    pub fn keys<'a>(&'a self) -> BTreeMapKeysIter<'a, OutputType, Option<PathBuf>> {
        self.0.keys()
    }

    pub fn values<'a>(&'a self) -> BTreeMapValuesIter<'a, OutputType, Option<PathBuf>> {
        self.0.values()
    }

    // True if any of the output types require codegen or linking.
    pub fn should_trans(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::Bitcode |
            OutputType::Assembly |
            OutputType::LlvmAssembly |
            OutputType::Object |
            OutputType::Exe => true,
            OutputType::Metadata |
            OutputType::DepInfo => false,
        })
    }
}


// Use tree-based collections to cheaply get a deterministic Hash implementation.
// DO NOT switch BTreeMap or BTreeSet out for an unsorted container type! That
// would break dependency tracking for commandline arguments.
#[derive(Clone, Hash)]
pub struct Externs(BTreeMap<String, BTreeSet<String>>);

impl Externs {
    pub fn new(data: BTreeMap<String, BTreeSet<String>>) -> Externs {
        Externs(data)
    }

    pub fn get(&self, key: &str) -> Option<&BTreeSet<String>> {
        self.0.get(key)
    }

    pub fn iter<'a>(&'a self) -> BTreeMapIter<'a, String, BTreeSet<String>> {
        self.0.iter()
    }
}

macro_rules! hash_option {
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, [UNTRACKED]) => ({});
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, [TRACKED]) => ({
        if $sub_hashes.insert(stringify!($opt_name),
                              $opt_expr as &dep_tracking::DepTrackingHash).is_some() {
            bug!("Duplicate key in CLI DepTrackingHash: {}", stringify!($opt_name))
        }
    });
    ($opt_name:ident,
     $opt_expr:expr,
     $sub_hashes:expr,
     [UNTRACKED_WITH_WARNING $warn_val:expr, $warn_text:expr, $error_format:expr]) => ({
        if *$opt_expr == $warn_val {
            early_warn($error_format, $warn_text)
        }
    });
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

// The top-level commandline options struct
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
// [UNTRACKED_WITH_WARNING(val, warning)]
// The option is incompatible with incremental compilation in some way. If it
// has the value `val`, the string `warning` is emitted as a warning.
//
// If you add a new option to this struct or one of the sub-structs like
// CodegenOptions, think about how it influences incremental compilation. If in
// doubt, specify [TRACKED], which is always "correct" but might lead to
// unnecessary re-compilation.
top_level_options!(
    pub struct Options {
        // The crate config requested for the session, which may be combined
        // with additional crate configurations during the compile process
        crate_types: Vec<CrateType> [TRACKED],
        optimize: OptLevel [TRACKED],
        // Include the debug_assertions flag into dependency tracking, since it
        // can influence whether overflow checks are done or not.
        debug_assertions: bool [TRACKED],
        debuginfo: DebugInfoLevel [TRACKED],
        lint_opts: Vec<(String, lint::Level)> [TRACKED],
        lint_cap: Option<lint::Level> [TRACKED],
        describe_lints: bool [UNTRACKED],
        output_types: OutputTypes [TRACKED],
        // FIXME(mw): We track this for now but it actually doesn't make too
        //            much sense: The search path can stay the same while the
        //            things discovered there might have changed on disk.
        search_paths: SearchPaths [TRACKED],
        libs: Vec<(String, Option<String>, cstore::NativeLibraryKind)> [TRACKED],
        maybe_sysroot: Option<PathBuf> [TRACKED],

        target_triple: String [TRACKED],

        test: bool [TRACKED],
        error_format: ErrorOutputType [UNTRACKED],

        // if Some, enable incremental compilation, using the given
        // directory to store intermediate results
        incremental: Option<PathBuf> [UNTRACKED],

        debugging_opts: DebuggingOptions [TRACKED],
        prints: Vec<PrintRequest> [UNTRACKED],
        cg: CodegenOptions [TRACKED],
        // FIXME(mw): We track this for now but it actually doesn't make too
        //            much sense: The value of this option can stay the same
        //            while the files they refer to might have changed on disk.
        externs: Externs [TRACKED],
        crate_name: Option<String> [TRACKED],
        // An optional name to use as the crate for std during std injection,
        // written `extern crate std = "name"`. Default to "std". Used by
        // out-of-tree drivers.
        alt_std_name: Option<String> [TRACKED],
        // Indicates how the compiler should treat unstable features
        unstable_features: UnstableFeatures [TRACKED],

        // Indicates whether this run of the compiler is actually rustdoc. This
        // is currently just a hack and will be removed eventually, so please
        // try to not rely on this too much.
        actually_rustdoc: bool [TRACKED],
    }
);

#[derive(Clone, PartialEq, Eq)]
pub enum PrintRequest {
    FileNames,
    Sysroot,
    CrateName,
    Cfg,
    TargetList,
    TargetCPUs,
    TargetFeatures,
    RelocationModels,
    CodeModels,
    TargetSpec,
}

pub enum Input {
    /// Load source from file
    File(PathBuf),
    Str {
        /// String that is shown in place of a filename
        name: String,
        /// Anonymous source string
        input: String,
    },
}

impl Input {
    pub fn filestem(&self) -> String {
        match *self {
            Input::File(ref ifile) => ifile.file_stem().unwrap()
                                           .to_str().unwrap().to_string(),
            Input::Str { .. } => "rust_out".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct OutputFilenames {
    pub out_directory: PathBuf,
    pub out_filestem: String,
    pub single_output_file: Option<PathBuf>,
    pub extra: String,
    pub outputs: OutputTypes,
}

/// Codegen unit names generated by the numbered naming scheme will contain this
/// marker right before the index of the codegen unit.
pub const NUMBERED_CODEGEN_UNIT_MARKER: &'static str = ".cgu-";

impl OutputFilenames {
    pub fn path(&self, flavor: OutputType) -> PathBuf {
        self.outputs.get(&flavor).and_then(|p| p.to_owned())
            .or_else(|| self.single_output_file.clone())
            .unwrap_or_else(|| self.temp_path(flavor, None))
    }

    /// Get the path where a compilation artifact of the given type for the
    /// given codegen unit should be placed on disk. If codegen_unit_name is
    /// None, a path distinct from those of any codegen unit will be generated.
    pub fn temp_path(&self,
                     flavor: OutputType,
                     codegen_unit_name: Option<&str>)
                     -> PathBuf {
        let extension = flavor.extension();
        self.temp_path_ext(extension, codegen_unit_name)
    }

    /// Like temp_path, but also supports things where there is no corresponding
    /// OutputType, like no-opt-bitcode or lto-bitcode.
    pub fn temp_path_ext(&self,
                         ext: &str,
                         codegen_unit_name: Option<&str>)
                         -> PathBuf {
        let base = self.out_directory.join(&self.filestem());

        let mut extension = String::new();

        if let Some(codegen_unit_name) = codegen_unit_name {
            if codegen_unit_name.contains(NUMBERED_CODEGEN_UNIT_MARKER) {
                // If we use the numbered naming scheme for modules, we don't want
                // the files to look like <crate-name><extra>.<crate-name>.<index>.<ext>
                // but simply <crate-name><extra>.<index>.<ext>
                let marker_offset = codegen_unit_name.rfind(NUMBERED_CODEGEN_UNIT_MARKER)
                                                     .unwrap();
                let index_offset = marker_offset + NUMBERED_CODEGEN_UNIT_MARKER.len();
                extension.push_str(&codegen_unit_name[index_offset .. ]);
            } else {
                extension.push_str(codegen_unit_name);
            };
        }

        if !ext.is_empty() {
            if !extension.is_empty() {
                extension.push_str(".");
            }

            extension.push_str(ext);
        }

        let path = base.with_extension(&extension[..]);
        path
    }

    pub fn with_extension(&self, extension: &str) -> PathBuf {
        self.out_directory.join(&self.filestem()).with_extension(extension)
    }

    pub fn filestem(&self) -> String {
        format!("{}{}", self.out_filestem, self.extra)
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
    (option_env!("CFG_COMPILER_HOST_TRIPLE")).
        expect("CFG_COMPILER_HOST_TRIPLE")
}

/// Some reasonable defaults
pub fn basic_options() -> Options {
    Options {
        crate_types: Vec::new(),
        optimize: OptLevel::No,
        debuginfo: NoDebugInfo,
        lint_opts: Vec::new(),
        lint_cap: None,
        describe_lints: false,
        output_types: OutputTypes(BTreeMap::new()),
        search_paths: SearchPaths::new(),
        maybe_sysroot: None,
        target_triple: host_triple().to_string(),
        test: false,
        incremental: None,
        debugging_opts: basic_debugging_options(),
        prints: Vec::new(),
        cg: basic_codegen_options(),
        error_format: ErrorOutputType::default(),
        externs: Externs(BTreeMap::new()),
        crate_name: None,
        alt_std_name: None,
        libs: Vec::new(),
        unstable_features: UnstableFeatures::Disallow,
        debug_assertions: true,
        actually_rustdoc: false,
    }
}

impl Options {
    /// True if there is a reason to build the dep graph.
    pub fn build_dep_graph(&self) -> bool {
        self.incremental.is_some() ||
            self.debugging_opts.dump_dep_graph ||
            self.debugging_opts.query_dep_graph
    }

    pub fn single_codegen_unit(&self) -> bool {
        self.incremental.is_none() ||
        self.cg.codegen_units == 1
    }
}

// The type of entry function, so
// users can have their own entry
// functions that don't start a
// scheduler
#[derive(Copy, Clone, PartialEq)]
pub enum EntryFnType {
    EntryMain,
    EntryStart,
    EntryNone,
}

#[derive(Copy, PartialEq, PartialOrd, Clone, Ord, Eq, Hash, Debug)]
pub enum CrateType {
    CrateTypeExecutable,
    CrateTypeDylib,
    CrateTypeRlib,
    CrateTypeStaticlib,
    CrateTypeCdylib,
    CrateTypeProcMacro,
}

#[derive(Clone, Hash)]
pub enum Passes {
    SomePasses(Vec<String>),
    AllPasses,
}

impl Passes {
    pub fn is_empty(&self) -> bool {
        match *self {
            SomePasses(ref v) => v.is_empty(),
            AllPasses => false,
        }
    }
}

/// Declare a macro that will define all CodegenOptions/DebuggingOptions fields and parsers all
/// at once. The goal of this macro is to define an interface that can be
/// programmatically used by the option parser in order to initialize the struct
/// without hardcoding field names all over the place.
///
/// The goal is to invoke this macro once with the correct fields, and then this
/// macro generates all necessary code. The main gotcha of this macro is the
/// cgsetters module which is a bunch of generated code to parse an option into
/// its respective field in the struct. There are a few hand-written parsers for
/// parsing specific types of values in this module.
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
                        (None, None) => bug!()
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

    impl<'a> dep_tracking::DepTrackingHash for $struct_name {

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
    pub const $stat: &'static [(&'static str, $setter_name,
                                     Option<&'static str>, &'static str)] =
        &[ $( (stringify!($opt), $mod_set::$opt, $mod_desc::$parse, $desc) ),* ];

    #[allow(non_upper_case_globals, dead_code)]
    mod $mod_desc {
        pub const parse_bool: Option<&'static str> = None;
        pub const parse_opt_bool: Option<&'static str> =
            Some("one of: `y`, `yes`, `on`, `n`, `no`, or `off`");
        pub const parse_string: Option<&'static str> = Some("a string");
        pub const parse_string_push: Option<&'static str> = Some("a string");
        pub const parse_opt_string: Option<&'static str> = Some("a string");
        pub const parse_list: Option<&'static str> = Some("a space-separated list of strings");
        pub const parse_opt_list: Option<&'static str> = Some("a space-separated list of strings");
        pub const parse_uint: Option<&'static str> = Some("a number");
        pub const parse_passes: Option<&'static str> =
            Some("a space-separated list of passes, or `all`");
        pub const parse_opt_uint: Option<&'static str> =
            Some("a number");
        pub const parse_panic_strategy: Option<&'static str> =
            Some("either `panic` or `abort`");
    }

    #[allow(dead_code)]
    mod $mod_set {
        use super::{$struct_name, Passes, SomePasses, AllPasses};
        use rustc_back::PanicStrategy;

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

        fn parse_list(slot: &mut Vec<String>, v: Option<&str>)
                      -> bool {
            match v {
                Some(s) => {
                    for s in s.split_whitespace() {
                        slot.push(s.to_string());
                    }
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

        fn parse_uint(slot: &mut usize, v: Option<&str>) -> bool {
            match v.and_then(|s| s.parse().ok()) {
                Some(i) => { *slot = i; true },
                None => false
            }
        }

        fn parse_opt_uint(slot: &mut Option<usize>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = s.parse().ok(); slot.is_some() }
                None => { *slot = None; true }
            }
        }

        fn parse_passes(slot: &mut Passes, v: Option<&str>) -> bool {
            match v {
                Some("all") => {
                    *slot = AllPasses;
                    true
                }
                v => {
                    let mut passes = vec![];
                    if parse_list(&mut passes, v) {
                        *slot = SomePasses(passes);
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
    }
) }

options! {CodegenOptions, CodegenSetter, basic_codegen_options,
         build_codegen_options, "C", "codegen",
         CG_OPTIONS, cg_type_desc, cgsetters,
    ar: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "tool to assemble archives with"),
    linker: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "system linker to link outputs with"),
    link_arg: Vec<String> = (vec![], parse_string_push, [UNTRACKED],
        "a single extra argument to pass to the linker (can be used several times)"),
    link_args: Option<Vec<String>> = (None, parse_opt_list, [UNTRACKED],
        "extra arguments to pass to the linker (space separated)"),
    link_dead_code: bool = (false, parse_bool, [UNTRACKED],
        "don't let linker strip dead code (turning it on can be used for code coverage)"),
    lto: bool = (false, parse_bool, [TRACKED],
        "perform LLVM link-time optimizations"),
    target_cpu: Option<String> = (None, parse_opt_string, [TRACKED],
        "select target processor (rustc --print target-cpus for details)"),
    target_feature: String = ("".to_string(), parse_string, [TRACKED],
        "target specific attributes (rustc --print target-features for details)"),
    passes: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of extra LLVM passes to run (space separated)"),
    llvm_args: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of arguments to pass to llvm (space separated)"),
    save_temps: bool = (false, parse_bool, [UNTRACKED_WITH_WARNING(true,
        "`-C save-temps` might not produce all requested temporary products \
         when incremental compilation is enabled.")],
        "save all temporary output files during compilation"),
    rpath: bool = (false, parse_bool, [UNTRACKED],
        "set rpath values in libs/exes"),
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
         "choose the relocation model to use (rustc --print relocation-models for details)"),
    code_model: Option<String> = (None, parse_opt_string, [TRACKED],
         "choose the code model to use (rustc --print code-models for details)"),
    metadata: Vec<String> = (Vec::new(), parse_list, [TRACKED],
         "metadata to mangle symbol names with"),
    extra_filename: String = ("".to_string(), parse_string, [UNTRACKED],
         "extra data to put in each output filename"),
    codegen_units: usize = (1, parse_uint, [UNTRACKED],
        "divide crate into N units to optimize in parallel"),
    remark: Passes = (SomePasses(Vec::new()), parse_passes, [UNTRACKED],
        "print remarks for these optimization passes (space separated, or \"all\")"),
    no_stack_check: bool = (false, parse_bool, [UNTRACKED],
        "the --no-stack-check flag is deprecated and does nothing"),
    debuginfo: Option<usize> = (None, parse_opt_uint, [TRACKED],
        "debug info emission level, 0 = no debug info, 1 = line tables only, \
         2 = full debug info with variable and type information"),
    opt_level: Option<String> = (None, parse_opt_string, [TRACKED],
        "optimize with possible levels 0-3, s, or z"),
    debug_assertions: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "explicitly enable the cfg(debug_assertions) directive"),
    inline_threshold: Option<usize> = (None, parse_opt_uint, [TRACKED],
        "set the inlining threshold for"),
    panic: Option<PanicStrategy> = (None, parse_panic_strategy,
        [TRACKED], "panic strategy to compile crate with"),
}

options! {DebuggingOptions, DebuggingSetter, basic_debugging_options,
         build_debugging_options, "Z", "debugging",
         DB_OPTIONS, db_type_desc, dbsetters,
    verbose: bool = (false, parse_bool, [UNTRACKED],
        "in general, enable more debug printouts"),
    time_passes: bool = (false, parse_bool, [UNTRACKED],
        "measure time of each rustc pass"),
    count_llvm_insns: bool = (false, parse_bool,
        [UNTRACKED_WITH_WARNING(true,
        "The output generated by `-Z count_llvm_insns` might not be reliable \
         when used with incremental compilation")],
        "count where LLVM instrs originate"),
    time_llvm_passes: bool = (false, parse_bool, [UNTRACKED_WITH_WARNING(true,
        "The output of `-Z time-llvm-passes` will only reflect timings of \
         re-translated modules when used with incremental compilation" )],
        "measure time of each LLVM pass"),
    input_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather statistics about the input"),
    trans_stats: bool = (false, parse_bool, [UNTRACKED_WITH_WARNING(true,
        "The output of `-Z trans-stats` might not be accurate when incremental \
         compilation is enabled")],
        "gather trans statistics"),
    asm_comments: bool = (false, parse_bool, [TRACKED],
        "generate comments into the assembly (may change behavior)"),
    no_verify: bool = (false, parse_bool, [TRACKED],
        "skip LLVM verification"),
    borrowck_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather borrowck statistics"),
    no_landing_pads: bool = (false, parse_bool, [TRACKED],
        "omit landing pads for unwinding"),
    debug_llvm: bool = (false, parse_bool, [UNTRACKED],
        "enable debug output from LLVM"),
    meta_stats: bool = (false, parse_bool, [UNTRACKED],
        "gather metadata statistics"),
    print_link_args: bool = (false, parse_bool, [UNTRACKED],
        "print the arguments passed to the linker"),
    print_llvm_passes: bool = (false, parse_bool, [UNTRACKED],
        "prints the llvm optimization passes being run"),
    ast_json: bool = (false, parse_bool, [UNTRACKED],
        "print the AST as JSON and halt"),
    ast_json_noexpand: bool = (false, parse_bool, [UNTRACKED],
        "print the pre-expansion AST as JSON and halt"),
    ls: bool = (false, parse_bool, [UNTRACKED],
        "list the symbols defined by a library crate"),
    save_analysis: bool = (false, parse_bool, [UNTRACKED],
        "write syntax and type analysis (in JSON format) information, in \
         addition to normal output"),
    save_analysis_csv: bool = (false, parse_bool, [UNTRACKED],
        "write syntax and type analysis (in CSV format) information, in addition to normal output"),
    save_analysis_api: bool = (false, parse_bool, [UNTRACKED],
        "write syntax and type analysis information for opaque libraries (in JSON format), \
         in addition to normal output"),
    print_move_fragments: bool = (false, parse_bool, [UNTRACKED],
        "print out move-fragment data for every fn"),
    flowgraph_print_loans: bool = (false, parse_bool, [UNTRACKED],
        "include loan analysis data in --unpretty flowgraph output"),
    flowgraph_print_moves: bool = (false, parse_bool, [UNTRACKED],
        "include move analysis data in --unpretty flowgraph output"),
    flowgraph_print_assigns: bool = (false, parse_bool, [UNTRACKED],
        "include assignment analysis data in --unpretty flowgraph output"),
    flowgraph_print_all: bool = (false, parse_bool, [UNTRACKED],
        "include all dataflow analysis data in --unpretty flowgraph output"),
    print_region_graph: bool = (false, parse_bool, [UNTRACKED],
         "prints region inference graph. \
          Use with RUST_REGION_GRAPH=help for more info"),
    parse_only: bool = (false, parse_bool, [UNTRACKED],
          "parse only; do not compile, assemble, or link"),
    no_trans: bool = (false, parse_bool, [TRACKED],
          "run all passes except translation; no output"),
    treat_err_as_bug: bool = (false, parse_bool, [TRACKED],
          "treat all errors that occur as bugs"),
    continue_parse_after_error: bool = (false, parse_bool, [TRACKED],
          "attempt to recover from parse errors (experimental)"),
    incremental: Option<String> = (None, parse_opt_string, [UNTRACKED],
          "enable incremental compilation (experimental)"),
    incremental_info: bool = (false, parse_bool, [UNTRACKED],
        "print high-level information about incremental reuse (or the lack thereof)"),
    incremental_dump_hash: bool = (false, parse_bool, [UNTRACKED],
        "dump hash information in textual format to stdout"),
    dump_dep_graph: bool = (false, parse_bool, [UNTRACKED],
          "dump the dependency graph to $RUST_DEP_GRAPH (default: /tmp/dep_graph.gv)"),
    query_dep_graph: bool = (false, parse_bool, [UNTRACKED],
          "enable queries of the dependency graph for regression testing"),
    no_analysis: bool = (false, parse_bool, [UNTRACKED],
          "parse and expand the source, but run no analysis"),
    extra_plugins: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "load extra plugins"),
    unstable_options: bool = (false, parse_bool, [UNTRACKED],
          "adds unstable command line options to rustc interface"),
    force_overflow_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
          "force overflow checks on or off"),
    trace_macros: bool = (false, parse_bool, [UNTRACKED],
          "for every macro invocation, print its name and arguments"),
    debug_macros: bool = (false, parse_bool, [TRACKED],
          "emit line numbers debug info inside macros"),
    enable_nonzeroing_move_hints: bool = (false, parse_bool, [TRACKED],
          "force nonzeroing move optimization on"),
    keep_hygiene_data: bool = (false, parse_bool, [UNTRACKED],
          "don't clear the hygiene data after analysis"),
    keep_ast: bool = (false, parse_bool, [UNTRACKED],
          "keep the AST after lowering it to HIR"),
    show_span: Option<String> = (None, parse_opt_string, [TRACKED],
          "show spans for compiler debugging (expr|pat|ty)"),
    print_type_sizes: bool = (false, parse_bool, [UNTRACKED],
          "print layout information for each type encountered"),
    print_trans_items: Option<String> = (None, parse_opt_string, [UNTRACKED],
          "print the result of the translation item collection pass"),
    mir_opt_level: usize = (1, parse_uint, [TRACKED],
          "set the MIR optimization level (0-3, default: 1)"),
    dump_mir: Option<String> = (None, parse_opt_string, [UNTRACKED],
          "dump MIR state at various points in translation"),
    dump_mir_dir: Option<String> = (None, parse_opt_string, [UNTRACKED],
          "the directory the MIR is dumped into"),
    perf_stats: bool = (false, parse_bool, [UNTRACKED],
          "print some performance-related statistics"),
    hir_stats: bool = (false, parse_bool, [UNTRACKED],
          "print some statistics about AST and HIR"),
    mir_stats: bool = (false, parse_bool, [UNTRACKED],
          "print some statistics about MIR"),
    always_encode_mir: bool = (false, parse_bool, [TRACKED],
          "encode MIR of all functions into the crate metadata"),
    osx_rpath_install_name: bool = (false, parse_bool, [TRACKED],
          "pass `-install_name @rpath/...` to the OSX linker"),
}

pub fn default_lib_output() -> CrateType {
    CrateTypeRlib
}

pub fn default_configuration(sess: &Session) -> ast::CrateConfig {
    let end = &sess.target.target.target_endian;
    let arch = &sess.target.target.arch;
    let wordsz = &sess.target.target.target_pointer_width;
    let os = &sess.target.target.target_os;
    let env = &sess.target.target.target_env;
    let vendor = &sess.target.target.target_vendor;
    let min_atomic_width = sess.target.target.min_atomic_width();
    let max_atomic_width = sess.target.target.max_atomic_width();

    let mut ret = HashSet::new();
    // Target bindings.
    ret.insert((Symbol::intern("target_os"), Some(Symbol::intern(os))));
    if let Some(ref fam) = sess.target.target.options.target_family {
        ret.insert((Symbol::intern("target_family"), Some(Symbol::intern(fam))));
        if fam == "windows" || fam == "unix" {
            ret.insert((Symbol::intern(fam), None));
        }
    }
    ret.insert((Symbol::intern("target_arch"), Some(Symbol::intern(arch))));
    ret.insert((Symbol::intern("target_endian"), Some(Symbol::intern(end))));
    ret.insert((Symbol::intern("target_pointer_width"), Some(Symbol::intern(wordsz))));
    ret.insert((Symbol::intern("target_env"), Some(Symbol::intern(env))));
    ret.insert((Symbol::intern("target_vendor"), Some(Symbol::intern(vendor))));
    if sess.target.target.options.has_elf_tls {
        ret.insert((Symbol::intern("target_thread_local"), None));
    }
    for &i in &[8, 16, 32, 64, 128] {
        if i >= min_atomic_width && i <= max_atomic_width {
            let s = i.to_string();
            ret.insert((Symbol::intern("target_has_atomic"), Some(Symbol::intern(&s))));
            if &s == wordsz {
                ret.insert((Symbol::intern("target_has_atomic"), Some(Symbol::intern("ptr"))));
            }
        }
    }
    if sess.opts.debug_assertions {
        ret.insert((Symbol::intern("debug_assertions"), None));
    }
    if sess.opts.crate_types.contains(&CrateTypeProcMacro) {
        ret.insert((Symbol::intern("proc_macro"), None));
    }
    return ret;
}

pub fn build_configuration(sess: &Session,
                           mut user_cfg: ast::CrateConfig)
                           -> ast::CrateConfig {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items
    let default_cfg = default_configuration(sess);
    // If the user wants a test runner, then add the test cfg
    if sess.opts.test {
        user_cfg.insert((Symbol::intern("test"), None));
    }
    user_cfg.extend(default_cfg.iter().cloned());
    user_cfg
}

pub fn build_target_config(opts: &Options, sp: &Handler) -> Config {
    let target = match Target::search(&opts.target_triple) {
        Ok(t) => t,
        Err(e) => {
            sp.struct_fatal(&format!("Error loading target specification: {}", e))
                .help("Use `--print target-list` for a list of built-in targets")
                .emit();
            panic!(FatalError);
        }
    };

    let (int_type, uint_type) = match &target.target_pointer_width[..] {
        "16" => (ast::IntTy::I16, ast::UintTy::U16),
        "32" => (ast::IntTy::I32, ast::UintTy::U32),
        "64" => (ast::IntTy::I64, ast::UintTy::U64),
        w    => panic!(sp.fatal(&format!("target specification was invalid: \
                                          unrecognized target-pointer-width {}", w))),
    };

    Config {
        target: target,
        int_type: int_type,
        uint_type: uint_type,
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OptionStability {
    Stable,

    // FIXME: historically there were some options which were either `-Z` or
    //        required the `-Z unstable-options` flag, which were all intended
    //        to be unstable. Unfortunately we didn't actually gate usage of
    //        these options on the stable compiler, so we still allow them there
    //        today. There are some warnings printed out about this in the
    //        driver.
    UnstableButNotReally,

    Unstable,
}

#[derive(Clone, PartialEq, Eq)]
pub struct RustcOptGroup {
    pub opt_group: getopts::OptGroup,
    pub stability: OptionStability,
}

impl RustcOptGroup {
    pub fn is_stable(&self) -> bool {
        self.stability == OptionStability::Stable
    }

    pub fn stable(g: getopts::OptGroup) -> RustcOptGroup {
        RustcOptGroup { opt_group: g, stability: OptionStability::Stable }
    }

    #[allow(dead_code)] // currently we have no "truly unstable" options
    pub fn unstable(g: getopts::OptGroup) -> RustcOptGroup {
        RustcOptGroup { opt_group: g, stability: OptionStability::Unstable }
    }

    fn unstable_bnr(g: getopts::OptGroup) -> RustcOptGroup {
        RustcOptGroup {
            opt_group: g,
            stability: OptionStability::UnstableButNotReally,
        }
    }
}

// The `opt` local module holds wrappers around the `getopts` API that
// adds extra rustc-specific metadata to each option; such metadata
// is exposed by .  The public
// functions below ending with `_u` are the functions that return
// *unstable* options, i.e. options that are only enabled when the
// user also passes the `-Z unstable-options` debugging flag.
mod opt {
    // The `fn opt_u` etc below are written so that we can use them
    // in the future; do not warn about them not being used right now.
    #![allow(dead_code)]

    use getopts;
    use super::RustcOptGroup;

    pub type R = RustcOptGroup;
    pub type S<'a> = &'a str;

    fn stable(g: getopts::OptGroup) -> R { RustcOptGroup::stable(g) }
    fn unstable(g: getopts::OptGroup) -> R { RustcOptGroup::unstable(g) }
    fn unstable_bnr(g: getopts::OptGroup) -> R { RustcOptGroup::unstable_bnr(g) }

    pub fn opt_s(a: S, b: S, c: S, d: S) -> R {
        stable(getopts::optopt(a, b, c, d))
    }
    pub fn multi_s(a: S, b: S, c: S, d: S) -> R {
        stable(getopts::optmulti(a, b, c, d))
    }
    pub fn flag_s(a: S, b: S, c: S) -> R {
        stable(getopts::optflag(a, b, c))
    }
    pub fn flagopt_s(a: S, b: S, c: S, d: S) -> R {
        stable(getopts::optflagopt(a, b, c, d))
    }
    pub fn flagmulti_s(a: S, b: S, c: S) -> R {
        stable(getopts::optflagmulti(a, b, c))
    }

    pub fn opt(a: S, b: S, c: S, d: S) -> R {
        unstable(getopts::optopt(a, b, c, d))
    }
    pub fn multi(a: S, b: S, c: S, d: S) -> R {
        unstable(getopts::optmulti(a, b, c, d))
    }
    pub fn flag(a: S, b: S, c: S) -> R {
        unstable(getopts::optflag(a, b, c))
    }
    pub fn flagopt(a: S, b: S, c: S, d: S) -> R {
        unstable(getopts::optflagopt(a, b, c, d))
    }
    pub fn flagmulti(a: S, b: S, c: S) -> R {
        unstable(getopts::optflagmulti(a, b, c))
    }

    // Do not use these functions for any new options added to the compiler, all
    // new options should use the `*_u` variants above to be truly unstable.
    pub fn opt_ubnr(a: S, b: S, c: S, d: S) -> R {
        unstable_bnr(getopts::optopt(a, b, c, d))
    }
    pub fn multi_ubnr(a: S, b: S, c: S, d: S) -> R {
        unstable_bnr(getopts::optmulti(a, b, c, d))
    }
    pub fn flag_ubnr(a: S, b: S, c: S) -> R {
        unstable_bnr(getopts::optflag(a, b, c))
    }
    pub fn flagopt_ubnr(a: S, b: S, c: S, d: S) -> R {
        unstable_bnr(getopts::optflagopt(a, b, c, d))
    }
    pub fn flagmulti_ubnr(a: S, b: S, c: S) -> R {
        unstable_bnr(getopts::optflagmulti(a, b, c))
    }
}

/// Returns the "short" subset of the rustc command line options,
/// including metadata for each option, such as whether the option is
/// part of the stable long-term interface for rustc.
pub fn rustc_short_optgroups() -> Vec<RustcOptGroup> {
    let mut print_opts = vec!["crate-name", "file-names", "sysroot", "cfg",
                              "target-list", "target-cpus", "target-features",
                              "relocation-models", "code-models"];
    if nightly_options::is_nightly_build() {
        print_opts.push("target-spec-json");
    }

    vec![
        opt::flag_s("h", "help", "Display this message"),
        opt::multi_s("", "cfg", "Configure the compilation environment", "SPEC"),
        opt::multi_s("L", "",   "Add a directory to the library search path. The
                             optional KIND can be one of dependency, crate, native,
                             framework or all (the default).", "[KIND=]PATH"),
        opt::multi_s("l", "",   "Link the generated crate(s) to the specified native
                             library NAME. The optional KIND can be one of
                             static, dylib, or framework. If omitted, dylib is
                             assumed.", "[KIND=]NAME"),
        opt::multi_s("", "crate-type", "Comma separated list of types of crates
                                    for the compiler to emit",
                   "[bin|lib|rlib|dylib|cdylib|staticlib|proc-macro]"),
        opt::opt_s("", "crate-name", "Specify the name of the crate being built",
               "NAME"),
        opt::multi_s("", "emit", "Comma separated list of types of output for \
                              the compiler to emit",
                 "[asm|llvm-bc|llvm-ir|obj|metadata|link|dep-info]"),
        opt::multi_s("", "print", "Comma separated list of compiler information to \
                               print on stdout", &format!("[{}]",
                               &print_opts.join("|"))),
        opt::flagmulti_s("g",  "",  "Equivalent to -C debuginfo=2"),
        opt::flagmulti_s("O", "", "Equivalent to -C opt-level=2"),
        opt::opt_s("o", "", "Write output to <filename>", "FILENAME"),
        opt::opt_s("",  "out-dir", "Write output to compiler-chosen filename \
                                in <dir>", "DIR"),
        opt::opt_s("", "explain", "Provide a detailed explanation of an error \
                               message", "OPT"),
        opt::flag_s("", "test", "Build a test harness"),
        opt::opt_s("", "target", "Target triple for which the code is compiled", "TARGET"),
        opt::multi_s("W", "warn", "Set lint warnings", "OPT"),
        opt::multi_s("A", "allow", "Set lint allowed", "OPT"),
        opt::multi_s("D", "deny", "Set lint denied", "OPT"),
        opt::multi_s("F", "forbid", "Set lint forbidden", "OPT"),
        opt::multi_s("", "cap-lints", "Set the most restrictive lint level. \
                                     More restrictive lints are capped at this \
                                     level", "LEVEL"),
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
    opts.extend_from_slice(&[
        opt::multi_s("", "extern", "Specify where an external rust library is located",
                     "NAME=PATH"),
        opt::opt_s("", "sysroot", "Override the system root", "PATH"),
        opt::multi_ubnr("Z", "", "Set internal debugging options", "FLAG"),
        opt::opt_s("", "error-format",
                      "How errors and other messages are produced",
                      "human|json"),
        opt::opt_s("", "color", "Configure coloring of output:
                                 auto   = colorize, if output goes to a tty (default);
                                 always = always colorize output;
                                 never  = never colorize output", "auto|always|never"),

        opt::flagopt_ubnr("", "pretty",
                          "Pretty-print the input instead of compiling;
                           valid types are: `normal` (un-annotated source),
                           `expanded` (crates expanded), or
                           `expanded,identified` (fully parenthesized, AST nodes with IDs).",
                          "TYPE"),
        opt::flagopt_ubnr("", "unpretty",
                          "Present the input source, unstable (and less-pretty) variants;
                           valid types are any of the types for `--pretty`, as well as:
                           `flowgraph=<nodeid>` (graphviz formatted flowgraph for node),
                           `everybody_loops` (all function bodies replaced with `loop {}`),
                           `hir` (the HIR), `hir,identified`, or
                           `hir,typed` (HIR with types for each node).",
                          "TYPE"),

        // new options here should **not** use the `_ubnr` functions, all new
        // unstable options should use the short variants to indicate that they
        // are truly unstable. All `_ubnr` flags are just that way because they
        // were so historically.
        //
        // You may also wish to keep this comment at the bottom of this list to
        // ensure that others see it.
    ]);
    opts
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
pub fn parse_cfgspecs(cfgspecs: Vec<String> ) -> ast::CrateConfig {
    cfgspecs.into_iter().map(|s| {
        let sess = parse::ParseSess::new();
        let mut parser =
            parse::new_parser_from_source_str(&sess, "cfgspec".to_string(), s.to_string());

        let meta_item = panictry!(parser.parse_meta_item());

        if parser.token != token::Eof {
            early_error(ErrorOutputType::default(), &format!("invalid --cfg argument: {}", s))
        } else if meta_item.is_meta_item_list() {
            let msg =
                format!("invalid predicate in --cfg command line argument: `{}`", meta_item.name());
            early_error(ErrorOutputType::default(), &msg)
        }

        (meta_item.name(), meta_item.value_str())
    }).collect::<ast::CrateConfig>()
}

pub fn build_session_options_and_crate_config(matches: &getopts::Matches)
                                              -> (Options, ast::CrateConfig) {
    let color = match matches.opt_str("color").as_ref().map(|s| &s[..]) {
        Some("auto")   => ColorConfig::Auto,
        Some("always") => ColorConfig::Always,
        Some("never")  => ColorConfig::Never,

        None => ColorConfig::Auto,

        Some(arg) => {
            early_error(ErrorOutputType::default(), &format!("argument for --color must be auto, \
                                                              always or never (instead was `{}`)",
                                                            arg))
        }
    };

    // We need the opts_present check because the driver will send us Matches
    // with only stable options if no unstable options are used. Since error-format
    // is unstable, it will not be present. We have to use opts_present not
    // opt_present because the latter will panic.
    let error_format = if matches.opts_present(&["error-format".to_owned()]) {
        match matches.opt_str("error-format").as_ref().map(|s| &s[..]) {
            Some("human")   => ErrorOutputType::HumanReadable(color),
            Some("json") => ErrorOutputType::Json,

            None => ErrorOutputType::HumanReadable(color),

            Some(arg) => {
                early_error(ErrorOutputType::HumanReadable(color),
                            &format!("argument for --error-format must be human or json (instead \
                                      was `{}`)",
                                     arg))
            }
        }
    } else {
        ErrorOutputType::HumanReadable(color)
    };

    let unparsed_crate_types = matches.opt_strs("crate-type");
    let (crate_types, emit_metadata) = parse_crate_types_from_list(unparsed_crate_types)
        .unwrap_or_else(|e| early_error(error_format, &e[..]));

    let mut lint_opts = vec![];
    let mut describe_lints = false;

    for &level in &[lint::Allow, lint::Warn, lint::Deny, lint::Forbid] {
        for lint_name in matches.opt_strs(level.as_str()) {
            if lint_name == "help" {
                describe_lints = true;
            } else {
                lint_opts.push((lint_name.replace("-", "_"), level));
            }
        }
    }

    let lint_cap = matches.opt_str("cap-lints").map(|cap| {
        lint::Level::from_str(&cap).unwrap_or_else(|| {
            early_error(error_format, &format!("unknown lint level: `{}`", cap))
        })
    });

    let debugging_opts = build_debugging_options(matches, error_format);

    let mut output_types = BTreeMap::new();
    if !debugging_opts.parse_only {
        for list in matches.opt_strs("emit") {
            for output_type in list.split(',') {
                let mut parts = output_type.splitn(2, '=');
                let output_type = match parts.next().unwrap() {
                    "asm" => OutputType::Assembly,
                    "llvm-ir" => OutputType::LlvmAssembly,
                    "llvm-bc" => OutputType::Bitcode,
                    "obj" => OutputType::Object,
                    "metadata" => OutputType::Metadata,
                    "link" => OutputType::Exe,
                    "dep-info" => OutputType::DepInfo,
                    part => {
                        early_error(error_format, &format!("unknown emission type: `{}`",
                                                    part))
                    }
                };
                let path = parts.next().map(PathBuf::from);
                output_types.insert(output_type, path);
            }
        }
    };
    if emit_metadata {
        output_types.insert(OutputType::Metadata, None);
    } else if output_types.is_empty() {
        output_types.insert(OutputType::Exe, None);
    }

    let mut cg = build_codegen_options(matches, error_format);

    // Issue #30063: if user requests llvm-related output to one
    // particular path, disable codegen-units.
    if matches.opt_present("o") && cg.codegen_units != 1 {
        let incompatible: Vec<_> = output_types.iter()
            .map(|ot_path| ot_path.0)
            .filter(|ot| {
                !ot.is_compatible_with_codegen_units_and_single_output_file()
            }).collect();
        if !incompatible.is_empty() {
            for ot in &incompatible {
                early_warn(error_format, &format!("--emit={} with -o incompatible with \
                                                 -C codegen-units=N for N > 1",
                                                ot.shorthand()));
            }
            early_warn(error_format, "resetting to default -C codegen-units=1");
            cg.codegen_units = 1;
        }
    }

    if cg.codegen_units < 1 {
        early_error(error_format, "Value for codegen units must be a positive nonzero integer");
    }

    let mut prints = Vec::<PrintRequest>::new();
    if cg.target_cpu.as_ref().map_or(false, |s| s == "help") {
        prints.push(PrintRequest::TargetCPUs);
        cg.target_cpu = None;
    };
    if cg.target_feature == "help" {
        prints.push(PrintRequest::TargetFeatures);
        cg.target_feature = "".to_string();
    }
    if cg.relocation_model.as_ref().map_or(false, |s| s == "help") {
        prints.push(PrintRequest::RelocationModels);
        cg.relocation_model = None;
    }
    if cg.code_model.as_ref().map_or(false, |s| s == "help") {
        prints.push(PrintRequest::CodeModels);
        cg.code_model = None;
    }

    let cg = cg;

    let sysroot_opt = matches.opt_str("sysroot").map(|m| PathBuf::from(&m));
    let target = matches.opt_str("target").unwrap_or(
        host_triple().to_string());
    let opt_level = {
        if matches.opt_present("O") {
            if cg.opt_level.is_some() {
                early_error(error_format, "-O and -C opt-level both provided");
            }
            OptLevel::Default
        } else {
            match (cg.opt_level.as_ref().map(String::as_ref),
                   nightly_options::is_nightly_build()) {
                (None, _) => OptLevel::No,
                (Some("0"), _) => OptLevel::No,
                (Some("1"), _) => OptLevel::Less,
                (Some("2"), _) => OptLevel::Default,
                (Some("3"), _) => OptLevel::Aggressive,
                (Some("s"), true) => OptLevel::Size,
                (Some("z"), true) => OptLevel::SizeMin,
                (Some("s"), false) | (Some("z"), false) => {
                    early_error(error_format, &format!("the optimizations s or z are only \
                                                        accepted on the nightly compiler"));
                },
                (Some(arg), _) => {
                    early_error(error_format, &format!("optimization level needs to be \
                                                      between 0-3 (instead was `{}`)",
                                                     arg));
                }
            }
        }
    };
    let debug_assertions = cg.debug_assertions.unwrap_or(opt_level == OptLevel::No);
    let debuginfo = if matches.opt_present("g") {
        if cg.debuginfo.is_some() {
            early_error(error_format, "-g and -C debuginfo both provided");
        }
        FullDebugInfo
    } else {
        match cg.debuginfo {
            None | Some(0) => NoDebugInfo,
            Some(1) => LimitedDebugInfo,
            Some(2) => FullDebugInfo,
            Some(arg) => {
                early_error(error_format, &format!("debug info level needs to be between \
                                                  0-2 (instead was `{}`)",
                                                 arg));
            }
        }
    };

    let mut search_paths = SearchPaths::new();
    for s in &matches.opt_strs("L") {
        search_paths.add_path(&s[..], error_format);
    }

    let libs = matches.opt_strs("l").into_iter().map(|s| {
        // Parse string of the form "[KIND=]lib[:new_name]",
        // where KIND is one of "dylib", "framework", "static".
        let mut parts = s.splitn(2, '=');
        let kind = parts.next().unwrap();
        let (name, kind) = match (parts.next(), kind) {
            (None, name) |
            (Some(name), "dylib") => (name, cstore::NativeUnknown),
            (Some(name), "framework") => (name, cstore::NativeFramework),
            (Some(name), "static") => (name, cstore::NativeStatic),
            (_, s) => {
                early_error(error_format, &format!("unknown library kind `{}`, expected \
                                                  one of dylib, framework, or static",
                                                 s));
            }
        };
        let mut name_parts = name.splitn(2, ':');
        let name = name_parts.next().unwrap();
        let new_name = name_parts.next();
        (name.to_string(), new_name.map(|n| n.to_string()), kind)
    }).collect();

    let cfg = parse_cfgspecs(matches.opt_strs("cfg"));
    let test = matches.opt_present("test");

    prints.extend(matches.opt_strs("print").into_iter().map(|s| {
        match &*s {
            "crate-name" => PrintRequest::CrateName,
            "file-names" => PrintRequest::FileNames,
            "sysroot" => PrintRequest::Sysroot,
            "cfg" => PrintRequest::Cfg,
            "target-list" => PrintRequest::TargetList,
            "target-cpus" => PrintRequest::TargetCPUs,
            "target-features" => PrintRequest::TargetFeatures,
            "relocation-models" => PrintRequest::RelocationModels,
            "code-models" => PrintRequest::CodeModels,
            "target-spec-json" if nightly_options::is_unstable_enabled(matches)
                => PrintRequest::TargetSpec,
            req => {
                early_error(error_format, &format!("unknown print request `{}`", req))
            }
        }
    }));

    if !cg.remark.is_empty() && debuginfo == NoDebugInfo {
        early_warn(error_format, "-C remark will not show source locations without \
                                --debuginfo");
    }

    let mut externs = BTreeMap::new();
    for arg in &matches.opt_strs("extern") {
        let mut parts = arg.splitn(2, '=');
        let name = match parts.next() {
            Some(s) => s,
            None => early_error(error_format, "--extern value must not be empty"),
        };
        let location = match parts.next() {
            Some(s) => s,
            None => early_error(error_format, "--extern value must be of the format `foo=bar`"),
        };

        externs.entry(name.to_string())
               .or_insert_with(BTreeSet::new)
               .insert(location.to_string());
    }

    let crate_name = matches.opt_str("crate-name");

    let incremental = debugging_opts.incremental.as_ref().map(|m| PathBuf::from(m));

    (Options {
        crate_types: crate_types,
        optimize: opt_level,
        debuginfo: debuginfo,
        lint_opts: lint_opts,
        lint_cap: lint_cap,
        describe_lints: describe_lints,
        output_types: OutputTypes(output_types),
        search_paths: search_paths,
        maybe_sysroot: sysroot_opt,
        target_triple: target,
        test: test,
        incremental: incremental,
        debugging_opts: debugging_opts,
        prints: prints,
        cg: cg,
        error_format: error_format,
        externs: Externs(externs),
        crate_name: crate_name,
        alt_std_name: None,
        libs: libs,
        unstable_features: UnstableFeatures::from_environment(),
        debug_assertions: debug_assertions,
        actually_rustdoc: false,
    },
    cfg)
}

pub fn parse_crate_types_from_list(list_list: Vec<String>)
                                   -> Result<(Vec<CrateType>, bool), String> {
    let mut crate_types: Vec<CrateType> = Vec::new();
    let mut emit_metadata = false;
    for unparsed_crate_type in &list_list {
        for part in unparsed_crate_type.split(',') {
            let new_part = match part {
                "lib"       => default_lib_output(),
                "rlib"      => CrateTypeRlib,
                "staticlib" => CrateTypeStaticlib,
                "dylib"     => CrateTypeDylib,
                "cdylib"    => CrateTypeCdylib,
                "bin"       => CrateTypeExecutable,
                "proc-macro" => CrateTypeProcMacro,
                // FIXME(#38640) remove this when Cargo is fixed.
                "metadata"  => {
                    early_warn(ErrorOutputType::default(), "--crate-type=metadata is deprecated, \
                                                            prefer --emit=metadata");
                    emit_metadata = true;
                    CrateTypeRlib
                }
                _ => {
                    return Err(format!("unknown crate type: `{}`",
                                       part));
                }
            };
            if !crate_types.contains(&new_part) {
                crate_types.push(new_part)
            }
        }
    }

    return Ok((crate_types, emit_metadata));
}

pub mod nightly_options {
    use getopts;
    use syntax::feature_gate::UnstableFeatures;
    use super::{ErrorOutputType, OptionStability, RustcOptGroup};
    use session::{early_error, early_warn};

    pub fn is_unstable_enabled(matches: &getopts::Matches) -> bool {
        is_nightly_build() && matches.opt_strs("Z").iter().any(|x| *x == "unstable-options")
    }

    pub fn is_nightly_build() -> bool {
        UnstableFeatures::from_environment().is_nightly_build()
    }

    pub fn check_nightly_options(matches: &getopts::Matches, flags: &[RustcOptGroup]) {
        let has_z_unstable_option = matches.opt_strs("Z").iter().any(|x| *x == "unstable-options");
        let really_allows_unstable_options = UnstableFeatures::from_environment()
            .is_nightly_build();

        for opt in flags.iter() {
            if opt.stability == OptionStability::Stable {
                continue
            }
            let opt_name = if opt.opt_group.long_name.is_empty() {
                &opt.opt_group.short_name
            } else {
                &opt.opt_group.long_name
            };
            if !matches.opt_present(opt_name) {
                continue
            }
            if opt_name != "Z" && !has_z_unstable_option {
                early_error(ErrorOutputType::default(),
                            &format!("the `-Z unstable-options` flag must also be passed to enable \
                                      the flag `{}`",
                                     opt_name));
            }
            if really_allows_unstable_options {
                continue
            }
            match opt.stability {
                OptionStability::Unstable => {
                    let msg = format!("the option `{}` is only accepted on the \
                                       nightly compiler", opt_name);
                    early_error(ErrorOutputType::default(), &msg);
                }
                OptionStability::UnstableButNotReally => {
                    let msg = format!("the option `{}` is unstable and should \
                                       only be used on the nightly compiler, but \
                                       it is currently accepted for backwards \
                                       compatibility; this will soon change, \
                                       see issue #31847 for more details",
                                      opt_name);
                    early_warn(ErrorOutputType::default(), &msg);
                }
                OptionStability::Stable => {}
            }
        }
    }
}

impl fmt::Display for CrateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CrateTypeExecutable => "bin".fmt(f),
            CrateTypeDylib => "dylib".fmt(f),
            CrateTypeRlib => "rlib".fmt(f),
            CrateTypeStaticlib => "staticlib".fmt(f),
            CrateTypeCdylib => "cdylib".fmt(f),
            CrateTypeProcMacro => "proc-macro".fmt(f),
        }
    }
}

/// Commandline arguments passed to the compiler have to be incorporated with
/// the dependency tracking system for incremental compilation. This module
/// provides some utilities to make this more convenient.
///
/// The values of all commandline arguments that are relevant for dependency
/// tracking are hashed into a single value that determines whether the
/// incremental compilation cache can be re-used or not. This hashing is done
/// via the DepTrackingHash trait defined below, since the standard Hash
/// implementation might not be suitable (e.g. arguments are stored in a Vec,
/// the hash of which is order dependent, but we might not want the order of
/// arguments to make a difference for the hash).
///
/// However, since the value provided by Hash::hash often *is* suitable,
/// especially for primitive types, there is the
/// impl_dep_tracking_hash_via_hash!() macro that allows to simply reuse the
/// Hash implementation for DepTrackingHash. It's important though that
/// we have an opt-in scheme here, so one is hopefully forced to think about
/// how the hash should be calculated when adding a new commandline argument.
mod dep_tracking {
    use lint;
    use middle::cstore;
    use session::search_paths::{PathKind, SearchPaths};
    use std::collections::BTreeMap;
    use std::hash::Hash;
    use std::path::PathBuf;
    use std::collections::hash_map::DefaultHasher;
    use super::{Passes, CrateType, OptLevel, DebugInfoLevel,
                OutputTypes, Externs, ErrorOutputType};
    use syntax::feature_gate::UnstableFeatures;
    use rustc_back::PanicStrategy;

    pub trait DepTrackingHash {
        fn hash(&self, &mut DefaultHasher, ErrorOutputType);
    }

    macro_rules! impl_dep_tracking_hash_via_hash {
        ($t:ty) => (
            impl DepTrackingHash for $t {
                fn hash(&self, hasher: &mut DefaultHasher, _: ErrorOutputType) {
                    Hash::hash(self, hasher);
                }
            }
        )
    }

    macro_rules! impl_dep_tracking_hash_for_sortable_vec_of {
        ($t:ty) => (
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
        );
    }

    impl_dep_tracking_hash_via_hash!(bool);
    impl_dep_tracking_hash_via_hash!(usize);
    impl_dep_tracking_hash_via_hash!(String);
    impl_dep_tracking_hash_via_hash!(lint::Level);
    impl_dep_tracking_hash_via_hash!(Option<bool>);
    impl_dep_tracking_hash_via_hash!(Option<usize>);
    impl_dep_tracking_hash_via_hash!(Option<String>);
    impl_dep_tracking_hash_via_hash!(Option<PanicStrategy>);
    impl_dep_tracking_hash_via_hash!(Option<lint::Level>);
    impl_dep_tracking_hash_via_hash!(Option<PathBuf>);
    impl_dep_tracking_hash_via_hash!(CrateType);
    impl_dep_tracking_hash_via_hash!(PanicStrategy);
    impl_dep_tracking_hash_via_hash!(Passes);
    impl_dep_tracking_hash_via_hash!(OptLevel);
    impl_dep_tracking_hash_via_hash!(DebugInfoLevel);
    impl_dep_tracking_hash_via_hash!(UnstableFeatures);
    impl_dep_tracking_hash_via_hash!(Externs);
    impl_dep_tracking_hash_via_hash!(OutputTypes);
    impl_dep_tracking_hash_via_hash!(cstore::NativeLibraryKind);

    impl_dep_tracking_hash_for_sortable_vec_of!(String);
    impl_dep_tracking_hash_for_sortable_vec_of!(CrateType);
    impl_dep_tracking_hash_for_sortable_vec_of!((String, lint::Level));
    impl_dep_tracking_hash_for_sortable_vec_of!((String, Option<String>,
                                                 cstore::NativeLibraryKind));
    impl DepTrackingHash for SearchPaths {
        fn hash(&self, hasher: &mut DefaultHasher, _: ErrorOutputType) {
            let mut elems: Vec<_> = self
                .iter(PathKind::All)
                .collect();
            elems.sort();
            Hash::hash(&elems, hasher);
        }
    }

    impl<T1, T2> DepTrackingHash for (T1, T2)
        where T1: DepTrackingHash,
              T2: DepTrackingHash
    {
        fn hash(&self, hasher: &mut DefaultHasher, error_format: ErrorOutputType) {
            Hash::hash(&0, hasher);
            DepTrackingHash::hash(&self.0, hasher, error_format);
            Hash::hash(&1, hasher);
            DepTrackingHash::hash(&self.1, hasher, error_format);
        }
    }

    impl<T1, T2, T3> DepTrackingHash for (T1, T2, T3)
        where T1: DepTrackingHash,
              T2: DepTrackingHash,
              T3: DepTrackingHash
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
    pub fn stable_hash(sub_hashes: BTreeMap<&'static str, &DepTrackingHash>,
                       hasher: &mut DefaultHasher,
                       error_format: ErrorOutputType) {
        for (key, sub_hash) in sub_hashes {
            // Using Hash::hash() instead of DepTrackingHash::hash() is fine for
            // the keys, as they are just plain strings
            Hash::hash(&key.len(), hasher);
            Hash::hash(key, hasher);
            sub_hash.hash(hasher, error_format);
        }
    }
}

#[cfg(test)]
mod tests {
    use dep_graph::DepGraph;
    use errors;
    use getopts::{getopts, OptGroup};
    use lint;
    use middle::cstore::{self, DummyCrateStore};
    use session::config::{build_configuration, build_session_options_and_crate_config};
    use session::build_session;
    use std::collections::{BTreeMap, BTreeSet};
    use std::iter::FromIterator;
    use std::path::PathBuf;
    use std::rc::Rc;
    use super::{OutputType, OutputTypes, Externs};
    use rustc_back::PanicStrategy;
    use syntax::symbol::Symbol;

    fn optgroups() -> Vec<OptGroup> {
        super::rustc_optgroups().into_iter()
                                .map(|a| a.opt_group)
                                .collect()
    }

    fn mk_map<K: Ord, V>(entries: Vec<(K, V)>) -> BTreeMap<K, V> {
        BTreeMap::from_iter(entries.into_iter())
    }

    fn mk_set<V: Ord>(entries: Vec<V>) -> BTreeSet<V> {
        BTreeSet::from_iter(entries.into_iter())
    }

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let dep_graph = DepGraph::new(false);
        let matches =
            &match getopts(&["--test".to_string()], &optgroups()) {
              Ok(m) => m,
              Err(f) => panic!("test_switch_implies_cfg_test: {}", f)
            };
        let registry = errors::registry::Registry::new(&[]);
        let (sessopts, cfg) = build_session_options_and_crate_config(matches);
        let sess = build_session(sessopts, &dep_graph, None, registry, Rc::new(DummyCrateStore));
        let cfg = build_configuration(&sess, cfg);
        assert!(cfg.contains(&(Symbol::intern("test"), None)));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let dep_graph = DepGraph::new(false);
        let matches =
            &match getopts(&["--test".to_string(), "--cfg=test".to_string()],
                           &optgroups()) {
              Ok(m) => m,
              Err(f) => {
                panic!("test_switch_implies_cfg_test_unless_cfg_test: {}", f)
              }
            };
        let registry = errors::registry::Registry::new(&[]);
        let (sessopts, cfg) = build_session_options_and_crate_config(matches);
        let sess = build_session(sessopts, &dep_graph, None, registry,
                                 Rc::new(DummyCrateStore));
        let cfg = build_configuration(&sess, cfg);
        let mut test_items = cfg.iter().filter(|&&(name, _)| name == "test");
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    }

    #[test]
    fn test_can_print_warnings() {
        let dep_graph = DepGraph::new(false);
        {
            let matches = getopts(&[
                "-Awarnings".to_string()
            ], &optgroups()).unwrap();
            let registry = errors::registry::Registry::new(&[]);
            let (sessopts, _) = build_session_options_and_crate_config(&matches);
            let sess = build_session(sessopts, &dep_graph, None, registry,
                                     Rc::new(DummyCrateStore));
            assert!(!sess.diagnostic().can_emit_warnings);
        }

        {
            let matches = getopts(&[
                "-Awarnings".to_string(),
                "-Dwarnings".to_string()
            ], &optgroups()).unwrap();
            let registry = errors::registry::Registry::new(&[]);
            let (sessopts, _) = build_session_options_and_crate_config(&matches);
            let sess = build_session(sessopts, &dep_graph, None, registry,
                                     Rc::new(DummyCrateStore));
            assert!(sess.diagnostic().can_emit_warnings);
        }

        {
            let matches = getopts(&[
                "-Adead_code".to_string()
            ], &optgroups()).unwrap();
            let registry = errors::registry::Registry::new(&[]);
            let (sessopts, _) = build_session_options_and_crate_config(&matches);
            let sess = build_session(sessopts, &dep_graph, None, registry,
                                     Rc::new(DummyCrateStore));
            assert!(sess.diagnostic().can_emit_warnings);
        }
    }

    #[test]
    fn test_output_types_tracking_hash_different_paths() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();
        let mut v3 = super::basic_options();

        v1.output_types = OutputTypes::new(&[(OutputType::Exe,
                                              Some(PathBuf::from("./some/thing")))]);
        v2.output_types = OutputTypes::new(&[(OutputType::Exe,
                                              Some(PathBuf::from("/some/thing")))]);
        v3.output_types = OutputTypes::new(&[(OutputType::Exe, None)]);

        assert!(v1.dep_tracking_hash() != v2.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() != v3.dep_tracking_hash());
        assert!(v2.dep_tracking_hash() != v3.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
        assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
    }

    #[test]
    fn test_output_types_tracking_hash_different_construction_order() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();

        v1.output_types = OutputTypes::new(&[
            (OutputType::Exe, Some(PathBuf::from("./some/thing"))),
            (OutputType::Bitcode, Some(PathBuf::from("./some/thing.bc"))),
        ]);

        v2.output_types = OutputTypes::new(&[
            (OutputType::Bitcode, Some(PathBuf::from("./some/thing.bc"))),
            (OutputType::Exe, Some(PathBuf::from("./some/thing"))),
        ]);

        assert_eq!(v1.dep_tracking_hash(), v2.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
    }

    #[test]
    fn test_externs_tracking_hash_different_values() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();
        let mut v3 = super::basic_options();

        v1.externs = Externs::new(mk_map(vec![
            (String::from("a"), mk_set(vec![String::from("b"),
                                            String::from("c")])),
            (String::from("d"), mk_set(vec![String::from("e"),
                                            String::from("f")])),
        ]));

        v2.externs = Externs::new(mk_map(vec![
            (String::from("a"), mk_set(vec![String::from("b"),
                                            String::from("c")])),
            (String::from("X"), mk_set(vec![String::from("e"),
                                            String::from("f")])),
        ]));

        v3.externs = Externs::new(mk_map(vec![
            (String::from("a"), mk_set(vec![String::from("b"),
                                            String::from("c")])),
            (String::from("d"), mk_set(vec![String::from("X"),
                                            String::from("f")])),
        ]));

        assert!(v1.dep_tracking_hash() != v2.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() != v3.dep_tracking_hash());
        assert!(v2.dep_tracking_hash() != v3.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
        assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
    }

    #[test]
    fn test_externs_tracking_hash_different_construction_order() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();
        let mut v3 = super::basic_options();

        v1.externs = Externs::new(mk_map(vec![
            (String::from("a"), mk_set(vec![String::from("b"),
                                            String::from("c")])),
            (String::from("d"), mk_set(vec![String::from("e"),
                                            String::from("f")])),
        ]));

        v2.externs = Externs::new(mk_map(vec![
            (String::from("d"), mk_set(vec![String::from("e"),
                                            String::from("f")])),
            (String::from("a"), mk_set(vec![String::from("b"),
                                            String::from("c")])),
        ]));

        v3.externs = Externs::new(mk_map(vec![
            (String::from("a"), mk_set(vec![String::from("b"),
                                            String::from("c")])),
            (String::from("d"), mk_set(vec![String::from("f"),
                                            String::from("e")])),
        ]));

        assert_eq!(v1.dep_tracking_hash(), v2.dep_tracking_hash());
        assert_eq!(v1.dep_tracking_hash(), v3.dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v3.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
        assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
    }

    #[test]
    fn test_lints_tracking_hash_different_values() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();
        let mut v3 = super::basic_options();

        v1.lint_opts = vec![(String::from("a"), lint::Allow),
                            (String::from("b"), lint::Warn),
                            (String::from("c"), lint::Deny),
                            (String::from("d"), lint::Forbid)];

        v2.lint_opts = vec![(String::from("a"), lint::Allow),
                            (String::from("b"), lint::Warn),
                            (String::from("X"), lint::Deny),
                            (String::from("d"), lint::Forbid)];

        v3.lint_opts = vec![(String::from("a"), lint::Allow),
                            (String::from("b"), lint::Warn),
                            (String::from("c"), lint::Forbid),
                            (String::from("d"), lint::Deny)];

        assert!(v1.dep_tracking_hash() != v2.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() != v3.dep_tracking_hash());
        assert!(v2.dep_tracking_hash() != v3.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
        assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
    }

    #[test]
    fn test_lints_tracking_hash_different_construction_order() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();

        v1.lint_opts = vec![(String::from("a"), lint::Allow),
                            (String::from("b"), lint::Warn),
                            (String::from("c"), lint::Deny),
                            (String::from("d"), lint::Forbid)];

        v2.lint_opts = vec![(String::from("a"), lint::Allow),
                            (String::from("c"), lint::Deny),
                            (String::from("b"), lint::Warn),
                            (String::from("d"), lint::Forbid)];

        assert_eq!(v1.dep_tracking_hash(), v2.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
    }

    #[test]
    fn test_search_paths_tracking_hash_different_values() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();
        let mut v3 = super::basic_options();
        let mut v4 = super::basic_options();
        let mut v5 = super::basic_options();

        // Reference
        v1.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v1.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v1.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v1.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);
        v1.search_paths.add_path("all=mno", super::ErrorOutputType::Json);

        // Native changed
        v2.search_paths.add_path("native=XXX", super::ErrorOutputType::Json);
        v2.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v2.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v2.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);
        v2.search_paths.add_path("all=mno", super::ErrorOutputType::Json);

        // Crate changed
        v2.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v2.search_paths.add_path("crate=XXX", super::ErrorOutputType::Json);
        v2.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v2.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);
        v2.search_paths.add_path("all=mno", super::ErrorOutputType::Json);

        // Dependency changed
        v3.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v3.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v3.search_paths.add_path("dependency=XXX", super::ErrorOutputType::Json);
        v3.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);
        v3.search_paths.add_path("all=mno", super::ErrorOutputType::Json);

        // Framework changed
        v4.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v4.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v4.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v4.search_paths.add_path("framework=XXX", super::ErrorOutputType::Json);
        v4.search_paths.add_path("all=mno", super::ErrorOutputType::Json);

        // All changed
        v5.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v5.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v5.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v5.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);
        v5.search_paths.add_path("all=XXX", super::ErrorOutputType::Json);

        assert!(v1.dep_tracking_hash() != v2.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() != v3.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() != v4.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() != v5.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
        assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
        assert_eq!(v4.dep_tracking_hash(), v4.clone().dep_tracking_hash());
        assert_eq!(v5.dep_tracking_hash(), v5.clone().dep_tracking_hash());
    }

    #[test]
    fn test_search_paths_tracking_hash_different_order() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();
        let mut v3 = super::basic_options();
        let mut v4 = super::basic_options();

        // Reference
        v1.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v1.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v1.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v1.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);
        v1.search_paths.add_path("all=mno", super::ErrorOutputType::Json);

        v2.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v2.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v2.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v2.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);
        v2.search_paths.add_path("all=mno", super::ErrorOutputType::Json);

        v3.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v3.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);
        v3.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v3.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v3.search_paths.add_path("all=mno", super::ErrorOutputType::Json);

        v4.search_paths.add_path("all=mno", super::ErrorOutputType::Json);
        v4.search_paths.add_path("native=abc", super::ErrorOutputType::Json);
        v4.search_paths.add_path("crate=def", super::ErrorOutputType::Json);
        v4.search_paths.add_path("dependency=ghi", super::ErrorOutputType::Json);
        v4.search_paths.add_path("framework=jkl", super::ErrorOutputType::Json);

        assert!(v1.dep_tracking_hash() == v2.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() == v3.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() == v4.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
        assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
        assert_eq!(v4.dep_tracking_hash(), v4.clone().dep_tracking_hash());
    }

    #[test]
    fn test_native_libs_tracking_hash_different_values() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();
        let mut v3 = super::basic_options();
        let mut v4 = super::basic_options();

        // Reference
        v1.libs = vec![(String::from("a"), None, cstore::NativeStatic),
                       (String::from("b"), None, cstore::NativeFramework),
                       (String::from("c"), None, cstore::NativeUnknown)];

        // Change label
        v2.libs = vec![(String::from("a"), None, cstore::NativeStatic),
                       (String::from("X"), None, cstore::NativeFramework),
                       (String::from("c"), None, cstore::NativeUnknown)];

        // Change kind
        v3.libs = vec![(String::from("a"), None, cstore::NativeStatic),
                       (String::from("b"), None, cstore::NativeStatic),
                       (String::from("c"), None, cstore::NativeUnknown)];

        // Change new-name
        v4.libs = vec![(String::from("a"), None, cstore::NativeStatic),
                       (String::from("b"), Some(String::from("X")), cstore::NativeFramework),
                       (String::from("c"), None, cstore::NativeUnknown)];

        assert!(v1.dep_tracking_hash() != v2.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() != v3.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() != v4.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
        assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
        assert_eq!(v4.dep_tracking_hash(), v4.clone().dep_tracking_hash());
    }

    #[test]
    fn test_native_libs_tracking_hash_different_order() {
        let mut v1 = super::basic_options();
        let mut v2 = super::basic_options();
        let mut v3 = super::basic_options();

        // Reference
        v1.libs = vec![(String::from("a"), None, cstore::NativeStatic),
                       (String::from("b"), None, cstore::NativeFramework),
                       (String::from("c"), None, cstore::NativeUnknown)];

        v2.libs = vec![(String::from("b"), None, cstore::NativeFramework),
                       (String::from("a"), None, cstore::NativeStatic),
                       (String::from("c"), None, cstore::NativeUnknown)];

        v3.libs = vec![(String::from("c"), None, cstore::NativeUnknown),
                       (String::from("a"), None, cstore::NativeStatic),
                       (String::from("b"), None, cstore::NativeFramework)];

        assert!(v1.dep_tracking_hash() == v2.dep_tracking_hash());
        assert!(v1.dep_tracking_hash() == v3.dep_tracking_hash());
        assert!(v2.dep_tracking_hash() == v3.dep_tracking_hash());

        // Check clone
        assert_eq!(v1.dep_tracking_hash(), v1.clone().dep_tracking_hash());
        assert_eq!(v2.dep_tracking_hash(), v2.clone().dep_tracking_hash());
        assert_eq!(v3.dep_tracking_hash(), v3.clone().dep_tracking_hash());
    }

    #[test]
    fn test_codegen_options_tracking_hash() {
        let reference = super::basic_options();
        let mut opts = super::basic_options();

        // Make sure the changing an [UNTRACKED] option leaves the hash unchanged
        opts.cg.ar = Some(String::from("abc"));
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        opts.cg.linker = Some(String::from("linker"));
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        opts.cg.link_args = Some(vec![String::from("abc"), String::from("def")]);
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        opts.cg.link_dead_code = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        opts.cg.rpath = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        opts.cg.extra_filename = String::from("extra-filename");
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        opts.cg.codegen_units = 42;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        opts.cg.remark = super::SomePasses(vec![String::from("pass1"),
                                                String::from("pass2")]);
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        opts.cg.save_temps = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());


        // Make sure changing a [TRACKED] option changes the hash
        opts = reference.clone();
        opts.cg.lto = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.target_cpu = Some(String::from("abc"));
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.target_feature = String::from("all the features, all of them");
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.passes = vec![String::from("1"), String::from("2")];
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.llvm_args = vec![String::from("1"), String::from("2")];
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.no_prepopulate_passes = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.no_vectorize_loops = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.no_vectorize_slp = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.soft_float = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.prefer_dynamic = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.no_integrated_as = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.no_redzone = Some(true);
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.relocation_model = Some(String::from("relocation model"));
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.code_model = Some(String::from("code model"));
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.metadata = vec![String::from("A"), String::from("B")];
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.debuginfo = Some(0xdeadbeef);
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.debuginfo = Some(0xba5eba11);
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.debug_assertions = Some(true);
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.inline_threshold = Some(0xf007ba11);
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.cg.panic = Some(PanicStrategy::Abort);
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());
    }

    #[test]
    fn test_debugging_options_tracking_hash() {
        let reference = super::basic_options();
        let mut opts = super::basic_options();

        // Make sure the changing an [UNTRACKED] option leaves the hash unchanged
        opts.debugging_opts.verbose = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.time_passes = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.count_llvm_insns = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.time_llvm_passes = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.input_stats = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.trans_stats = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.borrowck_stats = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.debug_llvm = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.meta_stats = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.print_link_args = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.print_llvm_passes = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.ast_json = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.ast_json_noexpand = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.ls = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.save_analysis = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.save_analysis_csv = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.save_analysis_api = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.print_move_fragments = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.flowgraph_print_loans = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.flowgraph_print_moves = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.flowgraph_print_assigns = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.flowgraph_print_all = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.print_region_graph = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.parse_only = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.incremental = Some(String::from("abc"));
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.dump_dep_graph = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.query_dep_graph = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.no_analysis = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.unstable_options = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.trace_macros = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.keep_hygiene_data = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.keep_ast = true;
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.print_trans_items = Some(String::from("abc"));
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.dump_mir = Some(String::from("abc"));
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());
        opts.debugging_opts.dump_mir_dir = Some(String::from("abc"));
        assert_eq!(reference.dep_tracking_hash(), opts.dep_tracking_hash());

        // Make sure changing a [TRACKED] option changes the hash
        opts = reference.clone();
        opts.debugging_opts.asm_comments = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.no_verify = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.no_landing_pads = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.no_trans = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.treat_err_as_bug = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.continue_parse_after_error = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.extra_plugins = vec![String::from("plugin1"), String::from("plugin2")];
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.force_overflow_checks = Some(true);
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.enable_nonzeroing_move_hints = true;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.show_span = Some(String::from("abc"));
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());

        opts = reference.clone();
        opts.debugging_opts.mir_opt_level = 3;
        assert!(reference.dep_tracking_hash() != opts.dep_tracking_hash());
    }
}
