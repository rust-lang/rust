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
pub use self::OptLevel::*;
pub use self::OutputType::*;
pub use self::DebugInfoLevel::*;

use session::{early_error, Session};

use rustc_back::target::Target;
use lint;
use metadata::cstore;

use syntax::ast;
use syntax::ast::{IntTy, UintTy};
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::diagnostic::{ColorConfig, Auto, Always, Never, SpanHandler};
use syntax::parse;
use syntax::parse::token::InternedString;

use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use getopts;
use std::cell::{RefCell};
use std::fmt;

use llvm;

pub struct Config {
    pub target: Target,
    pub int_type: IntTy,
    pub uint_type: UintTy,
}

#[deriving(Clone, Copy, PartialEq)]
pub enum OptLevel {
    No, // -O0
    Less, // -O1
    Default, // -O2
    Aggressive // -O3
}

#[deriving(Clone, Copy, PartialEq)]
pub enum DebugInfoLevel {
    NoDebugInfo,
    LimitedDebugInfo,
    FullDebugInfo,
}

#[deriving(Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
pub enum OutputType {
    OutputTypeBitcode,
    OutputTypeAssembly,
    OutputTypeLlvmAssembly,
    OutputTypeObject,
    OutputTypeExe,
    OutputTypeDepInfo,
}

#[deriving(Clone)]
pub struct Options {
    // The crate config requested for the session, which may be combined
    // with additional crate configurations during the compile process
    pub crate_types: Vec<CrateType>,

    pub gc: bool,
    pub optimize: OptLevel,
    pub debuginfo: DebugInfoLevel,
    pub lint_opts: Vec<(String, lint::Level)>,
    pub describe_lints: bool,
    pub output_types: Vec<OutputType> ,
    // This was mutable for rustpkg, which updates search paths based on the
    // parsed code. It remains mutable in case its replacements wants to use
    // this.
    pub addl_lib_search_paths: RefCell<Vec<Path>>,
    pub libs: Vec<(String, cstore::NativeLibraryKind)>,
    pub maybe_sysroot: Option<Path>,
    pub target_triple: String,
    // User-specified cfg meta items. The compiler itself will add additional
    // items to the crate config, and during parsing the entire crate config
    // will be added to the crate AST node.  This should not be used for
    // anything except building the full crate config prior to parsing.
    pub cfg: ast::CrateConfig,
    pub test: bool,
    pub parse_only: bool,
    pub no_trans: bool,
    pub no_analysis: bool,
    pub debugging_opts: u64,
    /// Whether to write dependency files. It's (enabled, optional filename).
    pub write_dependency_info: (bool, Option<Path>),
    pub prints: Vec<PrintRequest>,
    pub cg: CodegenOptions,
    pub color: ColorConfig,
    pub externs: HashMap<String, Vec<String>>,
    pub crate_name: Option<String>,
    /// An optional name to use as the crate for std during std injection,
    /// written `extern crate std = "name"`. Default to "std". Used by
    /// out-of-tree drivers.
    pub alt_std_name: Option<String>
}

#[deriving(Clone, PartialEq, Eq)]
#[allow(missing_copy_implementations)]
pub enum PrintRequest {
    FileNames,
    Sysroot,
    CrateName,
}

pub enum Input {
    /// Load source from file
    File(Path),
    /// The string is the source
    Str(String)
}

impl Input {
    pub fn filestem(&self) -> String {
        match *self {
            Input::File(ref ifile) => ifile.filestem_str().unwrap().to_string(),
            Input::Str(_) => "rust_out".to_string(),
        }
    }
}

#[deriving(Clone)]
pub struct OutputFilenames {
    pub out_directory: Path,
    pub out_filestem: String,
    pub single_output_file: Option<Path>,
    pub extra: String,
}

impl OutputFilenames {
    pub fn path(&self, flavor: OutputType) -> Path {
        match self.single_output_file {
            Some(ref path) => return path.clone(),
            None => {}
        }
        self.temp_path(flavor)
    }

    pub fn temp_path(&self, flavor: OutputType) -> Path {
        let base = self.out_directory.join(self.filestem());
        match flavor {
            OutputTypeBitcode => base.with_extension("bc"),
            OutputTypeAssembly => base.with_extension("s"),
            OutputTypeLlvmAssembly => base.with_extension("ll"),
            OutputTypeObject => base.with_extension("o"),
            OutputTypeDepInfo => base.with_extension("d"),
            OutputTypeExe => base,
        }
    }

    pub fn with_extension(&self, extension: &str) -> Path {
        self.out_directory.join(self.filestem()).with_extension(extension)
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
        gc: false,
        optimize: No,
        debuginfo: NoDebugInfo,
        lint_opts: Vec::new(),
        describe_lints: false,
        output_types: Vec::new(),
        addl_lib_search_paths: RefCell::new(Vec::new()),
        maybe_sysroot: None,
        target_triple: host_triple().to_string(),
        cfg: Vec::new(),
        test: false,
        parse_only: false,
        no_trans: false,
        no_analysis: false,
        debugging_opts: 0,
        write_dependency_info: (false, None),
        prints: Vec::new(),
        cg: basic_codegen_options(),
        color: Auto,
        externs: HashMap::new(),
        crate_name: None,
        alt_std_name: None,
        libs: Vec::new(),
    }
}

// The type of entry function, so
// users can have their own entry
// functions that don't start a
// scheduler
#[deriving(Copy, PartialEq)]
pub enum EntryFnType {
    EntryMain,
    EntryStart,
    EntryNone,
}

#[deriving(Copy, PartialEq, PartialOrd, Clone, Ord, Eq, Hash)]
pub enum CrateType {
    CrateTypeExecutable,
    CrateTypeDylib,
    CrateTypeRlib,
    CrateTypeStaticlib,
}

macro_rules! debugging_opts {
    ([ $opt:ident ] $cnt:expr ) => (
        pub const $opt: u64 = 1 << $cnt;
    );
    ([ $opt:ident, $($rest:ident),* ] $cnt:expr ) => (
        pub const $opt: u64 = 1 << $cnt;
        debugging_opts! { [ $($rest),* ] $cnt + 1 }
    )
}

debugging_opts! {
    [
        VERBOSE,
        TIME_PASSES,
        COUNT_LLVM_INSNS,
        TIME_LLVM_PASSES,
        TRANS_STATS,
        ASM_COMMENTS,
        NO_VERIFY,
        BORROWCK_STATS,
        NO_LANDING_PADS,
        DEBUG_LLVM,
        SHOW_SPAN,
        COUNT_TYPE_SIZES,
        META_STATS,
        GC,
        PRINT_LINK_ARGS,
        PRINT_LLVM_PASSES,
        AST_JSON,
        AST_JSON_NOEXPAND,
        LS,
        SAVE_ANALYSIS,
        PRINT_MOVE_FRAGMENTS,
        FLOWGRAPH_PRINT_LOANS,
        FLOWGRAPH_PRINT_MOVES,
        FLOWGRAPH_PRINT_ASSIGNS,
        FLOWGRAPH_PRINT_ALL,
        PRINT_REGION_GRAPH,
        PARSE_ONLY,
        NO_TRANS,
        NO_ANALYSIS,
        UNSTABLE_OPTIONS
    ]
    0
}

pub fn debugging_opts_map() -> Vec<(&'static str, &'static str, u64)> {
    vec![("verbose", "in general, enable more debug printouts", VERBOSE),
     ("time-passes", "measure time of each rustc pass", TIME_PASSES),
     ("count-llvm-insns", "count where LLVM \
                           instrs originate", COUNT_LLVM_INSNS),
     ("time-llvm-passes", "measure time of each LLVM pass",
      TIME_LLVM_PASSES),
     ("trans-stats", "gather trans statistics", TRANS_STATS),
     ("asm-comments", "generate comments into the assembly (may change behavior)",
      ASM_COMMENTS),
     ("no-verify", "skip LLVM verification", NO_VERIFY),
     ("borrowck-stats", "gather borrowck statistics",  BORROWCK_STATS),
     ("no-landing-pads", "omit landing pads for unwinding",
      NO_LANDING_PADS),
     ("debug-llvm", "enable debug output from LLVM", DEBUG_LLVM),
     ("show-span", "show spans for compiler debugging", SHOW_SPAN),
     ("count-type-sizes", "count the sizes of aggregate types",
      COUNT_TYPE_SIZES),
     ("meta-stats", "gather metadata statistics", META_STATS),
     ("print-link-args", "Print the arguments passed to the linker",
      PRINT_LINK_ARGS),
     ("gc", "Garbage collect shared data (experimental)", GC),
     ("print-llvm-passes",
      "Prints the llvm optimization passes being run",
      PRINT_LLVM_PASSES),
     ("ast-json", "Print the AST as JSON and halt", AST_JSON),
     ("ast-json-noexpand", "Print the pre-expansion AST as JSON and halt", AST_JSON_NOEXPAND),
     ("ls", "List the symbols defined by a library crate", LS),
     ("save-analysis", "Write syntax and type analysis information \
                        in addition to normal output", SAVE_ANALYSIS),
     ("print-move-fragments", "Print out move-fragment data for every fn",
      PRINT_MOVE_FRAGMENTS),
     ("flowgraph-print-loans", "Include loan analysis data in \
                       --pretty flowgraph output", FLOWGRAPH_PRINT_LOANS),
     ("flowgraph-print-moves", "Include move analysis data in \
                       --pretty flowgraph output", FLOWGRAPH_PRINT_MOVES),
     ("flowgraph-print-assigns", "Include assignment analysis data in \
                       --pretty flowgraph output", FLOWGRAPH_PRINT_ASSIGNS),
     ("flowgraph-print-all", "Include all dataflow analysis data in \
                       --pretty flowgraph output", FLOWGRAPH_PRINT_ALL),
     ("print-region-graph", "Prints region inference graph. \
                             Use with RUST_REGION_GRAPH=help for more info",
      PRINT_REGION_GRAPH),
     ("parse-only", "Parse only; do not compile, assemble, or link", PARSE_ONLY),
     ("no-trans", "Run all passes except translation; no output", NO_TRANS),
     ("no-analysis", "Parse and expand the source, but run no analysis and",
      NO_TRANS),
     ("unstable-options", "Adds unstable command line options to rustc interface",
      UNSTABLE_OPTIONS)]
}

#[deriving(Clone)]
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

/// Declare a macro that will define all CodegenOptions fields and parsers all
/// at once. The goal of this macro is to define an interface that can be
/// programmatically used by the option parser in order to initialize the struct
/// without hardcoding field names all over the place.
///
/// The goal is to invoke this macro once with the correct fields, and then this
/// macro generates all necessary code. The main gotcha of this macro is the
/// cgsetters module which is a bunch of generated code to parse an option into
/// its respective field in the struct. There are a few hand-written parsers for
/// parsing specific types of values in this module.
macro_rules! cgoptions {
    ($($opt:ident : $t:ty = ($init:expr, $parse:ident, $desc:expr)),* ,) =>
(
    #[deriving(Clone)]
    pub struct CodegenOptions { $(pub $opt: $t),* }

    pub fn basic_codegen_options() -> CodegenOptions {
        CodegenOptions { $($opt: $init),* }
    }

    pub type CodegenSetter = fn(&mut CodegenOptions, v: Option<&str>) -> bool;
    pub const CG_OPTIONS: &'static [(&'static str, CodegenSetter,
                                     Option<&'static str>, &'static str)] =
        &[ $( (stringify!($opt), cgsetters::$opt, cg_type_descs::$parse, $desc) ),* ];

    #[allow(non_upper_case_globals)]
    mod cg_type_descs {
        pub const parse_bool: Option<&'static str> = None;
        pub const parse_opt_bool: Option<&'static str> = None;
        pub const parse_string: Option<&'static str> = Some("a string");
        pub const parse_opt_string: Option<&'static str> = Some("a string");
        pub const parse_list: Option<&'static str> = Some("a space-separated list of strings");
        pub const parse_opt_list: Option<&'static str> = Some("a space-separated list of strings");
        pub const parse_uint: Option<&'static str> = Some("a number");
        pub const parse_passes: Option<&'static str> =
            Some("a space-separated list of passes, or `all`");
        pub const parse_opt_uint: Option<&'static str> =
            Some("a number");
    }

    mod cgsetters {
        use super::{CodegenOptions, Passes, SomePasses, AllPasses};

        $(
            pub fn $opt(cg: &mut CodegenOptions, v: Option<&str>) -> bool {
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
                Some(..) => false,
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

        fn parse_list(slot: &mut Vec<String>, v: Option<&str>)
                      -> bool {
            match v {
                Some(s) => {
                    for s in s.words() {
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
                    let v = s.words().map(|s| s.to_string()).collect();
                    *slot = Some(v);
                    true
                },
                None => false,
            }
        }

        fn parse_uint(slot: &mut uint, v: Option<&str>) -> bool {
            match v.and_then(from_str) {
                Some(i) => { *slot = i; true },
                None => false
            }
        }

        fn parse_opt_uint(slot: &mut Option<uint>, v: Option<&str>) -> bool {
            match v {
                Some(s) => { *slot = from_str(s); slot.is_some() }
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
                    let mut passes = vec!();
                    if parse_list(&mut passes, v) {
                        *slot = SomePasses(passes);
                        true
                    } else {
                        false
                    }
                }
            }
        }
    }
) }

cgoptions! {
    ar: Option<String> = (None, parse_opt_string,
        "tool to assemble archives with"),
    linker: Option<String> = (None, parse_opt_string,
        "system linker to link outputs with"),
    link_args: Option<Vec<String>> = (None, parse_opt_list,
        "extra arguments to pass to the linker (space separated)"),
    lto: bool = (false, parse_bool,
        "perform LLVM link-time optimizations"),
    target_cpu: Option<String> = (None, parse_opt_string,
        "select target processor (llc -mcpu=help for details)"),
    target_feature: String = ("".to_string(), parse_string,
        "target specific attributes (llc -mattr=help for details)"),
    passes: Vec<String> = (Vec::new(), parse_list,
        "a list of extra LLVM passes to run (space separated)"),
    llvm_args: Vec<String> = (Vec::new(), parse_list,
        "a list of arguments to pass to llvm (space separated)"),
    save_temps: bool = (false, parse_bool,
        "save all temporary output files during compilation"),
    rpath: bool = (false, parse_bool,
        "set rpath values in libs/exes"),
    no_prepopulate_passes: bool = (false, parse_bool,
        "don't pre-populate the pass manager with a list of passes"),
    no_vectorize_loops: bool = (false, parse_bool,
        "don't run the loop vectorization optimization passes"),
    no_vectorize_slp: bool = (false, parse_bool,
        "don't run LLVM's SLP vectorization pass"),
    soft_float: bool = (false, parse_bool,
        "generate software floating point library calls"),
    prefer_dynamic: bool = (false, parse_bool,
        "prefer dynamic linking to static linking"),
    no_integrated_as: bool = (false, parse_bool,
        "use an external assembler rather than LLVM's integrated one"),
    no_redzone: Option<bool> = (None, parse_opt_bool,
        "disable the use of the redzone"),
    relocation_model: Option<String> = (None, parse_opt_string,
         "choose the relocation model to use (llc -relocation-model for details)"),
    code_model: Option<String> = (None, parse_opt_string,
         "choose the code model to use (llc -code-model for details)"),
    metadata: Vec<String> = (Vec::new(), parse_list,
         "metadata to mangle symbol names with"),
    extra_filename: String = ("".to_string(), parse_string,
         "extra data to put in each output filename"),
    codegen_units: uint = (1, parse_uint,
        "divide crate into N units to optimize in parallel"),
    remark: Passes = (SomePasses(Vec::new()), parse_passes,
        "print remarks for these optimization passes (space separated, or \"all\")"),
    no_stack_check: bool = (false, parse_bool,
        "disable checks for stack exhaustion (a memory-safety hazard!)"),
    debuginfo: Option<uint> = (None, parse_opt_uint,
        "debug info emission level, 0 = no debug info, 1 = line tables only, \
         2 = full debug info with variable and type information"),
    opt_level: Option<uint> = (None, parse_opt_uint,
        "Optimize with possible levels 0-3"),
}

pub fn build_codegen_options(matches: &getopts::Matches) -> CodegenOptions
{
    let mut cg = basic_codegen_options();
    for option in matches.opt_strs("C").into_iter() {
        let mut iter = option.splitn(1, '=');
        let key = iter.next().unwrap();
        let value = iter.next();
        let option_to_lookup = key.replace("-", "_");
        let mut found = false;
        for &(candidate, setter, opt_type_desc, _) in CG_OPTIONS.iter() {
            if option_to_lookup != candidate { continue }
            if !setter(&mut cg, value) {
                match (value, opt_type_desc) {
                    (Some(..), None) => {
                        early_error(format!("codegen option `{}` takes no \
                                             value", key)[])
                    }
                    (None, Some(type_desc)) => {
                        early_error(format!("codegen option `{0}` requires \
                                             {1} (-C {0}=<value>)",
                                            key, type_desc)[])
                    }
                    (Some(value), Some(type_desc)) => {
                        early_error(format!("incorrect value `{}` for codegen \
                                             option `{}` - {} was expected",
                                             value, key, type_desc)[])
                    }
                    (None, None) => unreachable!()
                }
            }
            found = true;
            break;
        }
        if !found {
            early_error(format!("unknown codegen option: `{}`",
                                key)[]);
        }
    }
    return cg;
}

pub fn default_lib_output() -> CrateType {
    CrateTypeRlib
}

pub fn default_configuration(sess: &Session) -> ast::CrateConfig {
    use syntax::parse::token::intern_and_get_ident as intern;

    let end = sess.target.target.target_endian[];
    let arch = sess.target.target.arch[];
    let wordsz = sess.target.target.target_word_size[];
    let os = sess.target.target.target_os[];

    let fam = match sess.target.target.options.is_like_windows {
        true  => InternedString::new("windows"),
        false => InternedString::new("unix")
    };

    let mk = attr::mk_name_value_item_str;
    return vec!(// Target bindings.
         attr::mk_word_item(fam.clone()),
         mk(InternedString::new("target_os"), intern(os)),
         mk(InternedString::new("target_family"), fam),
         mk(InternedString::new("target_arch"), intern(arch)),
         mk(InternedString::new("target_endian"), intern(end)),
         mk(InternedString::new("target_word_size"),
            intern(wordsz))
    );
}

pub fn append_configuration(cfg: &mut ast::CrateConfig,
                            name: InternedString) {
    if !cfg.iter().any(|mi| mi.name() == name) {
        cfg.push(attr::mk_word_item(name))
    }
}

pub fn build_configuration(sess: &Session) -> ast::CrateConfig {
    // Combine the configuration requested by the session (command line) with
    // some default and generated configuration items
    let default_cfg = default_configuration(sess);
    let mut user_cfg = sess.opts.cfg.clone();
    // If the user wants a test runner, then add the test cfg
    if sess.opts.test {
        append_configuration(&mut user_cfg, InternedString::new("test"))
    }
    let mut v = user_cfg.into_iter().collect::<Vec<_>>();
    v.push_all(default_cfg[]);
    v
}

pub fn build_target_config(opts: &Options, sp: &SpanHandler) -> Config {
    let target = match Target::search(opts.target_triple[]) {
        Ok(t) => t,
        Err(e) => {
            sp.handler().fatal((format!("Error loading target specification: {}", e))[]);
    }
    };

    let (int_type, uint_type) = match target.target_word_size[] {
        "32" => (ast::TyI32, ast::TyU32),
        "64" => (ast::TyI64, ast::TyU64),
        w    => sp.handler().fatal((format!("target specification was invalid: unrecognized \
                                            target-word-size {}", w))[])
    };

    Config {
        target: target,
        int_type: int_type,
        uint_type: uint_type,
    }
}

/// Returns the "short" subset of the stable rustc command line options.
pub fn short_optgroups() -> Vec<getopts::OptGroup> {
    rustc_short_optgroups().into_iter()
        .filter(|g|g.is_stable())
        .map(|g|g.opt_group)
        .collect()
}

/// Returns all of the stable rustc command line options.
pub fn optgroups() -> Vec<getopts::OptGroup> {
    rustc_optgroups().into_iter()
        .filter(|g|g.is_stable())
        .map(|g|g.opt_group)
        .collect()
}

#[deriving(Copy, Clone, PartialEq, Eq, Show)]
pub enum OptionStability { Stable, Unstable }

#[deriving(Clone, PartialEq, Eq)]
pub struct RustcOptGroup {
    pub opt_group: getopts::OptGroup,
    pub stability: OptionStability,
}

impl RustcOptGroup {
    pub fn is_stable(&self) -> bool {
        self.stability == OptionStability::Stable
    }

    fn stable(g: getopts::OptGroup) -> RustcOptGroup {
        RustcOptGroup { opt_group: g, stability: OptionStability::Stable }
    }

    fn unstable(g: getopts::OptGroup) -> RustcOptGroup {
        RustcOptGroup { opt_group: g, stability: OptionStability::Unstable }
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

    type R = RustcOptGroup;
    type S<'a> = &'a str;

    fn stable(g: getopts::OptGroup) -> R { RustcOptGroup::stable(g) }
    fn unstable(g: getopts::OptGroup) -> R { RustcOptGroup::unstable(g) }

    // FIXME (pnkfelix): We default to stable since the current set of
    // options is defacto stable.  However, it would be good to revise the
    // code so that a stable option is the thing that takes extra effort
    // to encode.

    pub fn     opt(a: S, b: S, c: S, d: S) -> R { stable(getopts::optopt(a, b, c, d)) }
    pub fn   multi(a: S, b: S, c: S, d: S) -> R { stable(getopts::optmulti(a, b, c, d)) }
    pub fn    flag(a: S, b: S, c: S)       -> R { stable(getopts::optflag(a, b, c)) }
    pub fn flagopt(a: S, b: S, c: S, d: S) -> R { stable(getopts::optflagopt(a, b, c, d)) }

    pub fn     opt_u(a: S, b: S, c: S, d: S) -> R { unstable(getopts::optopt(a, b, c, d)) }
    pub fn   multi_u(a: S, b: S, c: S, d: S) -> R { unstable(getopts::optmulti(a, b, c, d)) }
    pub fn    flag_u(a: S, b: S, c: S)       -> R { unstable(getopts::optflag(a, b, c)) }
    pub fn flagopt_u(a: S, b: S, c: S, d: S) -> R { unstable(getopts::optflagopt(a, b, c, d)) }
}

/// Returns the "short" subset of the rustc command line options,
/// including metadata for each option, such as whether the option is
/// part of the stable long-term interface for rustc.
pub fn rustc_short_optgroups() -> Vec<RustcOptGroup> {
    vec![
        opt::flag("h", "help", "Display this message"),
        opt::multi("", "cfg", "Configure the compilation environment", "SPEC"),
        opt::multi("L", "",   "Add a directory to the library search path", "PATH"),
        opt::multi("l", "",   "Link the generated crate(s) to the specified native
                             library NAME. The optional KIND can be one of,
                             static, dylib, or framework. If omitted, dylib is
                             assumed.", "NAME[:KIND]"),
        opt::multi("", "crate-type", "Comma separated list of types of crates
                                    for the compiler to emit",
                   "[bin|lib|rlib|dylib|staticlib]"),
        opt::opt("", "crate-name", "Specify the name of the crate being built",
               "NAME"),
        opt::multi("", "emit", "Comma separated list of types of output for \
                              the compiler to emit",
                 "[asm|llvm-bc|llvm-ir|obj|link|dep-info]"),
        opt::multi("", "print", "Comma separated list of compiler information to \
                               print on stdout",
                 "[crate-name|output-file-names|sysroot]"),
        opt::flag("g",  "",  "Equivalent to -C debuginfo=2"),
        opt::flag("O", "", "Equivalent to -C opt-level=2"),
        opt::opt("o", "", "Write output to <filename>", "FILENAME"),
        opt::opt("",  "out-dir", "Write output to compiler-chosen filename \
                                in <dir>", "DIR"),
        opt::opt("", "explain", "Provide a detailed explanation of an error \
                               message", "OPT"),
        opt::flag("", "test", "Build a test harness"),
        opt::opt("", "target", "Target triple cpu-manufacturer-kernel[-os] \
                              to compile for (see chapter 3.4 of \
                              http://www.sourceware.org/autobook/
                              for details)",
               "TRIPLE"),
        opt::multi("W", "warn", "Set lint warnings", "OPT"),
        opt::multi("A", "allow", "Set lint allowed", "OPT"),
        opt::multi("D", "deny", "Set lint denied", "OPT"),
        opt::multi("F", "forbid", "Set lint forbidden", "OPT"),
        opt::multi("C", "codegen", "Set a codegen option", "OPT[=VALUE]"),
        opt::flag("V", "version", "Print version info and exit"),
        opt::flag("v", "verbose", "Use verbose output"),
    ]
}

/// Returns all rustc command line options, including metadata for
/// each option, such as whether the option is part of the stable
/// long-term interface for rustc.
pub fn rustc_optgroups() -> Vec<RustcOptGroup> {
    let mut opts = rustc_short_optgroups();
    opts.push_all(&[
        opt::multi("", "extern", "Specify where an external rust library is \
                                located",
                 "NAME=PATH"),
        opt::opt("", "opt-level", "Optimize with possible levels 0-3", "LEVEL"),
        opt::opt("", "sysroot", "Override the system root", "PATH"),
        opt::multi("Z", "", "Set internal debugging options", "FLAG"),
        opt::opt("", "color", "Configure coloring of output:
            auto   = colorize, if output goes to a tty (default);
            always = always colorize output;
            never  = never colorize output", "auto|always|never"),

        // DEPRECATED
        opt::flag("", "print-crate-name", "Output the crate name and exit"),
        opt::flag("", "print-file-name", "Output the file(s) that would be \
                                        written if compilation \
                                        continued and exit"),
        opt::opt("",  "debuginfo",  "Emit DWARF debug info to the objects created:
             0 = no debug info,
             1 = line-tables only (for stacktraces and breakpoints),
             2 = full debug info with variable and type information \
                    (same as -g)", "LEVEL"),
        opt::flag("", "no-trans", "Run all passes except translation; no output"),
        opt::flag("", "no-analysis", "Parse and expand the source, but run no \
                                    analysis and produce no output"),
        opt::flag("", "parse-only", "Parse only; do not compile, assemble, \
                                   or link"),
        opt::flagopt("", "pretty",
                   "Pretty-print the input instead of compiling;
                   valid types are: `normal` (un-annotated source),
                   `expanded` (crates expanded),
                   `typed` (crates expanded, with type annotations), or
                   `expanded,identified` (fully parenthesized, AST nodes with IDs).",
                 "TYPE"),
        opt::flagopt_u("", "xpretty",
                     "Pretty-print the input instead of compiling, unstable variants;
                      valid types are any of the types for `--pretty`, as well as:
                      `flowgraph=<nodeid>` (graphviz formatted flowgraph for node), or
                      `everybody_loops` (all function bodies replaced with `loop {}`).",
                     "TYPE"),
        opt::flagopt("", "dep-info",
                 "Output dependency info to <filename> after compiling, \
                  in a format suitable for use by Makefiles", "FILENAME"),
    ]);
    opts
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
pub fn parse_cfgspecs(cfgspecs: Vec<String> ) -> ast::CrateConfig {
    cfgspecs.into_iter().map(|s| {
        parse::parse_meta_from_source_str("cfgspec".to_string(),
                                          s.to_string(),
                                          Vec::new(),
                                          &parse::new_parse_sess())
    }).collect::<ast::CrateConfig>()
}

pub fn build_session_options(matches: &getopts::Matches) -> Options {

    let unparsed_crate_types = matches.opt_strs("crate-type");
    let crate_types = parse_crate_types_from_list(unparsed_crate_types)
        .unwrap_or_else(|e| early_error(e[]));

    let mut lint_opts = vec!();
    let mut describe_lints = false;

    for &level in [lint::Allow, lint::Warn, lint::Deny, lint::Forbid].iter() {
        for lint_name in matches.opt_strs(level.as_str()).into_iter() {
            if lint_name == "help" {
                describe_lints = true;
            } else {
                lint_opts.push((lint_name.replace("-", "_"), level));
            }
        }
    }

    let mut debugging_opts = 0;
    let debug_flags = matches.opt_strs("Z");
    let debug_map = debugging_opts_map();
    for debug_flag in debug_flags.iter() {
        let mut this_bit = 0;
        for &(name, _, bit) in debug_map.iter() {
            if name == *debug_flag {
                this_bit = bit;
                break;
            }
        }
        if this_bit == 0 {
            early_error(format!("unknown debug flag: {}",
                                *debug_flag)[])
        }
        debugging_opts |= this_bit;
    }

    let parse_only = if matches.opt_present("parse-only") {
        // FIXME(acrichto) uncomment deprecation warning
        // early_warn("--parse-only is deprecated in favor of -Z parse-only");
        true
    } else {
        debugging_opts & PARSE_ONLY != 0
    };
    let no_trans = if matches.opt_present("no-trans") {
        // FIXME(acrichto) uncomment deprecation warning
        // early_warn("--no-trans is deprecated in favor of -Z no-trans");
        true
    } else {
        debugging_opts & NO_TRANS != 0
    };
    let no_analysis = if matches.opt_present("no-analysis") {
        // FIXME(acrichto) uncomment deprecation warning
        // early_warn("--no-analysis is deprecated in favor of -Z no-analysis");
        true
    } else {
        debugging_opts & NO_ANALYSIS != 0
    };

    if debugging_opts & DEBUG_LLVM != 0 {
        unsafe { llvm::LLVMSetDebug(1); }
    }

    let mut output_types = Vec::new();
    if !parse_only && !no_trans {
        let unparsed_output_types = matches.opt_strs("emit");
        for unparsed_output_type in unparsed_output_types.iter() {
            for part in unparsed_output_type.split(',') {
                let output_type = match part.as_slice() {
                    "asm" => OutputTypeAssembly,
                    "llvm-ir" => OutputTypeLlvmAssembly,
                    "llvm-bc" => OutputTypeBitcode,
                    "obj" => OutputTypeObject,
                    "link" => OutputTypeExe,
                    "dep-info" => OutputTypeDepInfo,
                    _ => {
                        early_error(format!("unknown emission type: `{}`",
                                            part)[])
                    }
                };
                output_types.push(output_type)
            }
        }
    };
    output_types.sort();
    output_types.dedup();
    if output_types.len() == 0 {
        output_types.push(OutputTypeExe);
    }

    let cg = build_codegen_options(matches);

    let sysroot_opt = matches.opt_str("sysroot").map(|m| Path::new(m));
    let target = matches.opt_str("target").unwrap_or(
        host_triple().to_string());
    let opt_level = {
        if matches.opt_present("O") {
            if matches.opt_present("opt-level") {
                early_error("-O and --opt-level both provided");
            }
            if cg.opt_level.is_some() {
                early_error("-O and -C opt-level both provided");
            }
            Default
        } else if matches.opt_present("opt-level") {
            // FIXME(acrichto) uncomment deprecation warning
            // early_warn("--opt-level=N is deprecated in favor of -C opt-level=N");
            match matches.opt_str("opt-level").as_ref().map(|s| s.as_slice()) {
                None      |
                Some("0") => No,
                Some("1") => Less,
                Some("2") => Default,
                Some("3") => Aggressive,
                Some(arg) => {
                    early_error(format!("optimization level needs to be \
                                         between 0-3 (instead was `{}`)",
                                        arg)[]);
                }
            }
        } else {
            match cg.opt_level {
                None => No,
                Some(0) => No,
                Some(1) => Less,
                Some(2) => Default,
                Some(3) => Aggressive,
                Some(arg) => {
                    early_error(format!("optimization level needs to be \
                                         between 0-3 (instead was `{}`)",
                                        arg).as_slice());
                }
            }
        }
    };
    let gc = debugging_opts & GC != 0;
    let debuginfo = if matches.opt_present("g") {
        if matches.opt_present("debuginfo") {
            early_error("-g and --debuginfo both provided");
        }
        if cg.debuginfo.is_some() {
            early_error("-g and -C debuginfo both provided");
        }
        FullDebugInfo
    } else if matches.opt_present("debuginfo") {
        // FIXME(acrichto) uncomment deprecation warning
        // early_warn("--debuginfo=N is deprecated in favor of -C debuginfo=N");
        match matches.opt_str("debuginfo").as_ref().map(|s| s.as_slice()) {
            Some("0") => NoDebugInfo,
            Some("1") => LimitedDebugInfo,
            None      |
            Some("2") => FullDebugInfo,
            Some(arg) => {
                early_error(format!("debug info level needs to be between \
                                     0-2 (instead was `{}`)",
                                    arg)[]);
            }
        }
    } else {
        match cg.debuginfo {
            None | Some(0) => NoDebugInfo,
            Some(1) => LimitedDebugInfo,
            Some(2) => FullDebugInfo,
            Some(arg) => {
                early_error(format!("debug info level needs to be between \
                                     0-2 (instead was `{}`)",
                                    arg).as_slice());
            }
        }
    };

    let addl_lib_search_paths = matches.opt_strs("L").iter().map(|s| {
        Path::new(s[])
    }).collect();

    let libs = matches.opt_strs("l").into_iter().map(|s| {
        let mut parts = s.rsplitn(1, ':');
        let kind = parts.next().unwrap();
        let (name, kind) = match (parts.next(), kind) {
            (None, name) |
            (Some(name), "dylib") => (name, cstore::NativeUnknown),
            (Some(name), "framework") => (name, cstore::NativeFramework),
            (Some(name), "static") => (name, cstore::NativeStatic),
            (_, s) => {
                early_error(format!("unknown library kind `{}`, expected \
                                     one of dylib, framework, or static",
                                    s)[]);
            }
        };
        (name.to_string(), kind)
    }).collect();

    let cfg = parse_cfgspecs(matches.opt_strs("cfg"));
    let test = matches.opt_present("test");
    let write_dependency_info = if matches.opt_present("dep-info") {
        // FIXME(acrichto) uncomment deprecation warning
        // early_warn("--dep-info has been deprecated in favor of --emit");
        (true, matches.opt_str("dep-info").map(|p| Path::new(p)))
    } else {
        (output_types.contains(&OutputTypeDepInfo), None)
    };

    let mut prints = matches.opt_strs("print").into_iter().map(|s| {
        match s.as_slice() {
            "crate-name" => PrintRequest::CrateName,
            "file-names" => PrintRequest::FileNames,
            "sysroot" => PrintRequest::Sysroot,
            req => {
                early_error(format!("unknown print request `{}`", req).as_slice())
            }
        }
    }).collect::<Vec<_>>();
    if matches.opt_present("print-crate-name") {
        // FIXME(acrichto) uncomment deprecation warning
        // early_warn("--print-crate-name has been deprecated in favor of \
        //             --print crate-name");
        prints.push(PrintRequest::CrateName);
    }
    if matches.opt_present("print-file-name") {
        // FIXME(acrichto) uncomment deprecation warning
        // early_warn("--print-file-name has been deprecated in favor of \
        //             --print file-names");
        prints.push(PrintRequest::FileNames);
    }

    if !cg.remark.is_empty() && debuginfo == NoDebugInfo {
        // FIXME(acrichto) uncomment deprecation warning
        // early_warn("-C remark will not show source locations without \
        //             --debuginfo");
    }

    let color = match matches.opt_str("color").as_ref().map(|s| s[]) {
        Some("auto")   => Auto,
        Some("always") => Always,
        Some("never")  => Never,

        None => Auto,

        Some(arg) => {
            early_error(format!("argument for --color must be auto, always \
                                 or never (instead was `{}`)",
                                arg)[])
        }
    };

    let mut externs = HashMap::new();
    for arg in matches.opt_strs("extern").iter() {
        let mut parts = arg.splitn(1, '=');
        let name = match parts.next() {
            Some(s) => s,
            None => early_error("--extern value must not be empty"),
        };
        let location = match parts.next() {
            Some(s) => s,
            None => early_error("--extern value must be of the format `foo=bar`"),
        };

        match externs.entry(name.to_string()) {
            Vacant(entry) => { entry.set(vec![location.to_string()]); },
            Occupied(mut entry) => { entry.get_mut().push(location.to_string()); },
        }
    }

    let crate_name = matches.opt_str("crate-name");

    Options {
        crate_types: crate_types,
        gc: gc,
        optimize: opt_level,
        debuginfo: debuginfo,
        lint_opts: lint_opts,
        describe_lints: describe_lints,
        output_types: output_types,
        addl_lib_search_paths: RefCell::new(addl_lib_search_paths),
        maybe_sysroot: sysroot_opt,
        target_triple: target,
        cfg: cfg,
        test: test,
        parse_only: parse_only,
        no_trans: no_trans,
        no_analysis: no_analysis,
        debugging_opts: debugging_opts,
        write_dependency_info: write_dependency_info,
        prints: prints,
        cg: cg,
        color: color,
        externs: externs,
        crate_name: crate_name,
        alt_std_name: None,
        libs: libs,
    }
}

pub fn parse_crate_types_from_list(list_list: Vec<String>) -> Result<Vec<CrateType>, String> {

    let mut crate_types: Vec<CrateType> = Vec::new();
    for unparsed_crate_type in list_list.iter() {
        for part in unparsed_crate_type.split(',') {
            let new_part = match part {
                "lib"       => default_lib_output(),
                "rlib"      => CrateTypeRlib,
                "staticlib" => CrateTypeStaticlib,
                "dylib"     => CrateTypeDylib,
                "bin"       => CrateTypeExecutable,
                _ => {
                    return Err(format!("unknown crate type: `{}`",
                                       part));
                }
            };
            crate_types.push(new_part)
        }
    }

    return Ok(crate_types);
}

impl fmt::Show for CrateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CrateTypeExecutable => "bin".fmt(f),
            CrateTypeDylib => "dylib".fmt(f),
            CrateTypeRlib => "rlib".fmt(f),
            CrateTypeStaticlib => "staticlib".fmt(f)
        }
    }
}

#[cfg(test)]
mod test {

    use session::config::{build_configuration, optgroups, build_session_options};
    use session::build_session;

    use getopts::getopts;
    use syntax::attr;
    use syntax::attr::AttrMetaMethods;
    use syntax::diagnostics;

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let matches =
            &match getopts(&["--test".to_string()], optgroups()[]) {
              Ok(m) => m,
              Err(f) => panic!("test_switch_implies_cfg_test: {}", f)
            };
        let registry = diagnostics::registry::Registry::new(&[]);
        let sessopts = build_session_options(matches);
        let sess = build_session(sessopts, None, registry);
        let cfg = build_configuration(&sess);
        assert!((attr::contains_name(cfg[], "test")));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let matches =
            &match getopts(&["--test".to_string(), "--cfg=test".to_string()],
                           optgroups()[]) {
              Ok(m) => m,
              Err(f) => {
                panic!("test_switch_implies_cfg_test_unless_cfg_test: {}", f)
              }
            };
        let registry = diagnostics::registry::Registry::new(&[]);
        let sessopts = build_session_options(matches);
        let sess = build_session(sessopts, None, registry);
        let cfg = build_configuration(&sess);
        let mut test_items = cfg.iter().filter(|m| m.name() == "test");
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    }

    #[test]
    fn test_can_print_warnings() {
        {
            let matches = getopts(&[
                "-Awarnings".to_string()
            ], optgroups()[]).unwrap();
            let registry = diagnostics::registry::Registry::new(&[]);
            let sessopts = build_session_options(&matches);
            let sess = build_session(sessopts, None, registry);
            assert!(!sess.can_print_warnings);
        }

        {
            let matches = getopts(&[
                "-Awarnings".to_string(),
                "-Dwarnings".to_string()
            ], optgroups()[]).unwrap();
            let registry = diagnostics::registry::Registry::new(&[]);
            let sessopts = build_session_options(&matches);
            let sess = build_session(sessopts, None, registry);
            assert!(sess.can_print_warnings);
        }

        {
            let matches = getopts(&[
                "-Adead_code".to_string()
            ], optgroups()[]).unwrap();
            let registry = diagnostics::registry::Registry::new(&[]);
            let sessopts = build_session_options(&matches);
            let sess = build_session(sessopts, None, registry);
            assert!(sess.can_print_warnings);
        }
    }
}
