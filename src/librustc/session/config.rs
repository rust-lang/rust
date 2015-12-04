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
pub use self::DebugInfoLevel::*;

use session::{early_error, early_warn, Session};
use session::search_paths::SearchPaths;

use rustc_back::target::Target;
use lint;
use middle::cstore;

use syntax::ast::{self, IntTy, UintTy};
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::diagnostic::{ColorConfig, Auto, Always, Never, SpanHandler};
use syntax::parse;
use syntax::parse::token::InternedString;
use syntax::feature_gate::UnstableFeatures;

use getopts;
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::path::PathBuf;

use llvm;

pub struct Config {
    pub target: Target,
    pub int_type: IntTy,
    pub uint_type: UintTy,
}

#[derive(Clone, Copy, PartialEq)]
pub enum OptLevel {
    No, // -O0
    Less, // -O1
    Default, // -O2
    Aggressive // -O3
}

#[derive(Clone, Copy, PartialEq)]
pub enum DebugInfoLevel {
    NoDebugInfo,
    LimitedDebugInfo,
    FullDebugInfo,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputType {
    Bitcode,
    Assembly,
    LlvmAssembly,
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
            OutputType::Object => false,
        }
    }

    fn shorthand(&self) -> &'static str {
        match *self {
            OutputType::Bitcode => "llvm-bc",
            OutputType::Assembly => "asm",
            OutputType::LlvmAssembly => "llvm-ir",
            OutputType::Object => "obj",
            OutputType::Exe => "link",
            OutputType::DepInfo => "dep-info",
        }
    }
}

#[derive(Clone)]
pub struct Options {
    // The crate config requested for the session, which may be combined
    // with additional crate configurations during the compile process
    pub crate_types: Vec<CrateType>,

    pub gc: bool,
    pub optimize: OptLevel,
    pub debug_assertions: bool,
    pub debuginfo: DebugInfoLevel,
    pub lint_opts: Vec<(String, lint::Level)>,
    pub lint_cap: Option<lint::Level>,
    pub describe_lints: bool,
    pub output_types: HashMap<OutputType, Option<PathBuf>>,
    // This was mutable for rustpkg, which updates search paths based on the
    // parsed code. It remains mutable in case its replacements wants to use
    // this.
    pub search_paths: SearchPaths,
    pub libs: Vec<(String, cstore::NativeLibraryKind)>,
    pub maybe_sysroot: Option<PathBuf>,
    pub target_triple: String,
    // User-specified cfg meta items. The compiler itself will add additional
    // items to the crate config, and during parsing the entire crate config
    // will be added to the crate AST node.  This should not be used for
    // anything except building the full crate config prior to parsing.
    pub cfg: ast::CrateConfig,
    pub test: bool,
    pub parse_only: bool,
    pub no_trans: bool,
    pub treat_err_as_bug: bool,
    pub no_analysis: bool,
    pub debugging_opts: DebuggingOptions,
    pub prints: Vec<PrintRequest>,
    pub cg: CodegenOptions,
    pub color: ColorConfig,
    pub show_span: Option<String>,
    pub externs: HashMap<String, Vec<String>>,
    pub crate_name: Option<String>,
    /// An optional name to use as the crate for std during std injection,
    /// written `extern crate std = "name"`. Default to "std". Used by
    /// out-of-tree drivers.
    pub alt_std_name: Option<String>,
    /// Indicates how the compiler should treat unstable features
    pub unstable_features: UnstableFeatures
}

#[derive(Clone, PartialEq, Eq)]
pub enum PrintRequest {
    FileNames,
    Sysroot,
    CrateName,
}

pub enum Input {
    /// Load source from file
    File(PathBuf),
    /// The string is the source
    Str(String)
}

impl Input {
    pub fn filestem(&self) -> String {
        match *self {
            Input::File(ref ifile) => ifile.file_stem().unwrap()
                                           .to_str().unwrap().to_string(),
            Input::Str(_) => "rust_out".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct OutputFilenames {
    pub out_directory: PathBuf,
    pub out_filestem: String,
    pub single_output_file: Option<PathBuf>,
    pub extra: String,
    pub outputs: HashMap<OutputType, Option<PathBuf>>,
}

impl OutputFilenames {
    pub fn path(&self, flavor: OutputType) -> PathBuf {
        self.outputs.get(&flavor).and_then(|p| p.to_owned())
            .or_else(|| self.single_output_file.clone())
            .unwrap_or_else(|| self.temp_path(flavor))
    }

    pub fn temp_path(&self, flavor: OutputType) -> PathBuf {
        let base = self.out_directory.join(&self.filestem());
        match flavor {
            OutputType::Bitcode => base.with_extension("bc"),
            OutputType::Assembly => base.with_extension("s"),
            OutputType::LlvmAssembly => base.with_extension("ll"),
            OutputType::Object => base.with_extension("o"),
            OutputType::DepInfo => base.with_extension("d"),
            OutputType::Exe => base,
        }
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
        gc: false,
        optimize: No,
        debuginfo: NoDebugInfo,
        lint_opts: Vec::new(),
        lint_cap: None,
        describe_lints: false,
        output_types: HashMap::new(),
        search_paths: SearchPaths::new(),
        maybe_sysroot: None,
        target_triple: host_triple().to_string(),
        cfg: Vec::new(),
        test: false,
        parse_only: false,
        no_trans: false,
        treat_err_as_bug: false,
        no_analysis: false,
        debugging_opts: basic_debugging_options(),
        prints: Vec::new(),
        cg: basic_codegen_options(),
        color: Auto,
        show_span: None,
        externs: HashMap::new(),
        crate_name: None,
        alt_std_name: None,
        libs: Vec::new(),
        unstable_features: UnstableFeatures::Disallow,
        debug_assertions: true,
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
}

#[derive(Clone)]
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
     $($opt:ident : $t:ty = ($init:expr, $parse:ident, $desc:expr)),* ,) =>
(
    #[derive(Clone)]
    pub struct $struct_name { $(pub $opt: $t),* }

    pub fn $defaultfn() -> $struct_name {
        $struct_name { $($opt: $init),* }
    }

    pub fn $buildfn(matches: &getopts::Matches, color: ColorConfig) -> $struct_name
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
                            early_error(color, &format!("{} option `{}` takes no \
                                                         value", $outputname, key))
                        }
                        (None, Some(type_desc)) => {
                            early_error(color, &format!("{0} option `{1}` requires \
                                                         {2} ({3} {1}=<value>)",
                                                        $outputname, key,
                                                        type_desc, $prefix))
                        }
                        (Some(value), Some(type_desc)) => {
                            early_error(color, &format!("incorrect value `{}` for {} \
                                                         option `{}` - {} was expected",
                                                        value, $outputname,
                                                        key, type_desc))
                        }
                        (None, None) => unreachable!()
                    }
                }
                found = true;
                break;
            }
            if !found {
                early_error(color, &format!("unknown {} option: `{}`",
                                            $outputname, key));
            }
        }
        return op;
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
        pub const parse_opt_string: Option<&'static str> = Some("a string");
        pub const parse_list: Option<&'static str> = Some("a space-separated list of strings");
        pub const parse_opt_list: Option<&'static str> = Some("a space-separated list of strings");
        pub const parse_uint: Option<&'static str> = Some("a number");
        pub const parse_passes: Option<&'static str> =
            Some("a space-separated list of passes, or `all`");
        pub const parse_opt_uint: Option<&'static str> =
            Some("a number");
    }

    #[allow(dead_code)]
    mod $mod_set {
        use super::{$struct_name, Passes, SomePasses, AllPasses};

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

options! {CodegenOptions, CodegenSetter, basic_codegen_options,
         build_codegen_options, "C", "codegen",
         CG_OPTIONS, cg_type_desc, cgsetters,
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
    codegen_units: usize = (1, parse_uint,
        "divide crate into N units to optimize in parallel"),
    remark: Passes = (SomePasses(Vec::new()), parse_passes,
        "print remarks for these optimization passes (space separated, or \"all\")"),
    no_stack_check: bool = (false, parse_bool,
        "disable checks for stack exhaustion (a memory-safety hazard!)"),
    debuginfo: Option<usize> = (None, parse_opt_uint,
        "debug info emission level, 0 = no debug info, 1 = line tables only, \
         2 = full debug info with variable and type information"),
    opt_level: Option<usize> = (None, parse_opt_uint,
        "optimize with possible levels 0-3"),
    debug_assertions: Option<bool> = (None, parse_opt_bool,
        "explicitly enable the cfg(debug_assertions) directive"),
    inline_threshold: Option<usize> = (None, parse_opt_uint,
        "set the inlining threshold for"),
}


options! {DebuggingOptions, DebuggingSetter, basic_debugging_options,
         build_debugging_options, "Z", "debugging",
         DB_OPTIONS, db_type_desc, dbsetters,
    verbose: bool = (false, parse_bool,
        "in general, enable more debug printouts"),
    time_passes: bool = (false, parse_bool,
        "measure time of each rustc pass"),
    count_llvm_insns: bool = (false, parse_bool,
        "count where LLVM instrs originate"),
    time_llvm_passes: bool = (false, parse_bool,
        "measure time of each LLVM pass"),
    input_stats: bool = (false, parse_bool,
        "gather statistics about the input"),
    trans_stats: bool = (false, parse_bool,
        "gather trans statistics"),
    asm_comments: bool = (false, parse_bool,
        "generate comments into the assembly (may change behavior)"),
    no_verify: bool = (false, parse_bool,
        "skip LLVM verification"),
    borrowck_stats: bool = (false, parse_bool,
        "gather borrowck statistics"),
    no_landing_pads: bool = (false, parse_bool,
        "omit landing pads for unwinding"),
    debug_llvm: bool = (false, parse_bool,
        "enable debug output from LLVM"),
    count_type_sizes: bool = (false, parse_bool,
        "count the sizes of aggregate types"),
    meta_stats: bool = (false, parse_bool,
        "gather metadata statistics"),
    print_link_args: bool = (false, parse_bool,
        "print the arguments passed to the linker"),
    gc: bool = (false, parse_bool,
        "garbage collect shared data (experimental)"),
    print_llvm_passes: bool = (false, parse_bool,
        "prints the llvm optimization passes being run"),
    ast_json: bool = (false, parse_bool,
        "print the AST as JSON and halt"),
    ast_json_noexpand: bool = (false, parse_bool,
        "print the pre-expansion AST as JSON and halt"),
    ls: bool = (false, parse_bool,
        "list the symbols defined by a library crate"),
    save_analysis: bool = (false, parse_bool,
        "write syntax and type analysis information in addition to normal output"),
    print_move_fragments: bool = (false, parse_bool,
        "print out move-fragment data for every fn"),
    flowgraph_print_loans: bool = (false, parse_bool,
        "include loan analysis data in --unpretty flowgraph output"),
    flowgraph_print_moves: bool = (false, parse_bool,
        "include move analysis data in --unpretty flowgraph output"),
    flowgraph_print_assigns: bool = (false, parse_bool,
        "include assignment analysis data in --unpretty flowgraph output"),
    flowgraph_print_all: bool = (false, parse_bool,
        "include all dataflow analysis data in --unpretty flowgraph output"),
    print_region_graph: bool = (false, parse_bool,
         "prints region inference graph. \
          Use with RUST_REGION_GRAPH=help for more info"),
    parse_only: bool = (false, parse_bool,
          "parse only; do not compile, assemble, or link"),
    no_trans: bool = (false, parse_bool,
          "run all passes except translation; no output"),
    treat_err_as_bug: bool = (false, parse_bool,
          "treat all errors that occur as bugs"),
    no_analysis: bool = (false, parse_bool,
          "parse and expand the source, but run no analysis"),
    extra_plugins: Vec<String> = (Vec::new(), parse_list,
        "load extra plugins"),
    unstable_options: bool = (false, parse_bool,
          "adds unstable command line options to rustc interface"),
    print_enum_sizes: bool = (false, parse_bool,
          "print the size of enums and their variants"),
    force_overflow_checks: Option<bool> = (None, parse_opt_bool,
          "force overflow checks on or off"),
    force_dropflag_checks: Option<bool> = (None, parse_opt_bool,
          "force drop flag checks on or off"),
    trace_macros: bool = (false, parse_bool,
          "for every macro invocation, print its name and arguments"),
    enable_nonzeroing_move_hints: bool = (false, parse_bool,
          "force nonzeroing move optimization on"),
    keep_mtwt_tables: bool = (false, parse_bool,
          "don't clear the resolution tables after analysis"),
}

pub fn default_lib_output() -> CrateType {
    CrateTypeRlib
}

pub fn default_configuration(sess: &Session) -> ast::CrateConfig {
    use syntax::parse::token::intern_and_get_ident as intern;

    let end = &sess.target.target.target_endian;
    let arch = &sess.target.target.arch;
    let wordsz = &sess.target.target.target_pointer_width;
    let os = &sess.target.target.target_os;
    let env = &sess.target.target.target_env;
    let vendor = &sess.target.target.target_vendor;

    let fam = if let Some(ref fam) = sess.target.target.options.target_family {
        intern(fam)
    } else if sess.target.target.options.is_like_windows {
        InternedString::new("windows")
    } else {
        InternedString::new("unix")
    };

    let mk = attr::mk_name_value_item_str;
    let mut ret = vec![ // Target bindings.
        mk(InternedString::new("target_os"), intern(os)),
        mk(InternedString::new("target_family"), fam.clone()),
        mk(InternedString::new("target_arch"), intern(arch)),
        mk(InternedString::new("target_endian"), intern(end)),
        mk(InternedString::new("target_pointer_width"), intern(wordsz)),
        mk(InternedString::new("target_env"), intern(env)),
        mk(InternedString::new("target_vendor"), intern(vendor)),
    ];
    match &fam[..] {
        "windows" | "unix" => ret.push(attr::mk_word_item(fam)),
        _ => (),
    }
    if sess.opts.debug_assertions {
        ret.push(attr::mk_word_item(InternedString::new("debug_assertions")));
    }
    return ret;
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
    v.push_all(&default_cfg[..]);
    v
}

pub fn build_target_config(opts: &Options, sp: &SpanHandler) -> Config {
    let target = match Target::search(&opts.target_triple) {
        Ok(t) => t,
        Err(e) => {
            panic!(sp.handler().fatal(&format!("Error loading target specification: {}", e)));
        }
    };

    let (int_type, uint_type) = match &target.target_pointer_width[..] {
        "32" => (ast::TyI32, ast::TyU32),
        "64" => (ast::TyI64, ast::TyU64),
        w    => panic!(sp.handler().fatal(&format!("target specification was invalid: \
                                                    unrecognized target-pointer-width {}", w))),
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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OptionStability { Stable, Unstable }

#[derive(Clone, PartialEq, Eq)]
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

    pub type R = RustcOptGroup;
    pub type S<'a> = &'a str;

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
    pub fn flagmulti(a: S, b: S, c: S)     -> R { stable(getopts::optflagmulti(a, b, c)) }


    pub fn     opt_u(a: S, b: S, c: S, d: S) -> R { unstable(getopts::optopt(a, b, c, d)) }
    pub fn   multi_u(a: S, b: S, c: S, d: S) -> R { unstable(getopts::optmulti(a, b, c, d)) }
    pub fn    flag_u(a: S, b: S, c: S)       -> R { unstable(getopts::optflag(a, b, c)) }
    pub fn flagopt_u(a: S, b: S, c: S, d: S) -> R { unstable(getopts::optflagopt(a, b, c, d)) }
    pub fn flagmulti_u(a: S, b: S, c: S)     -> R { unstable(getopts::optflagmulti(a, b, c)) }
}

/// Returns the "short" subset of the rustc command line options,
/// including metadata for each option, such as whether the option is
/// part of the stable long-term interface for rustc.
pub fn rustc_short_optgroups() -> Vec<RustcOptGroup> {
    vec![
        opt::flag("h", "help", "Display this message"),
        opt::multi("", "cfg", "Configure the compilation environment", "SPEC"),
        opt::multi("L", "",   "Add a directory to the library search path",
                   "[KIND=]PATH"),
        opt::multi("l", "",   "Link the generated crate(s) to the specified native
                             library NAME. The optional KIND can be one of,
                             static, dylib, or framework. If omitted, dylib is
                             assumed.", "[KIND=]NAME"),
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
                 "[crate-name|file-names|sysroot]"),
        opt::flagmulti("g",  "",  "Equivalent to -C debuginfo=2"),
        opt::flagmulti("O", "", "Equivalent to -C opt-level=2"),
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
        opt::multi("", "cap-lints", "Set the most restrictive lint level. \
                                     More restrictive lints are capped at this \
                                     level", "LEVEL"),
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
        opt::opt("", "sysroot", "Override the system root", "PATH"),
        opt::multi("Z", "", "Set internal debugging options", "FLAG"),
        opt::opt("", "color", "Configure coloring of output:
            auto   = colorize, if output goes to a tty (default);
            always = always colorize output;
            never  = never colorize output", "auto|always|never"),

        opt::flagopt_u("", "pretty",
                   "Pretty-print the input instead of compiling;
                   valid types are: `normal` (un-annotated source),
                   `expanded` (crates expanded), or
                   `expanded,identified` (fully parenthesized, AST nodes with IDs).",
                 "TYPE"),
        opt::flagopt_u("", "unpretty",
                     "Present the input source, unstable (and less-pretty) variants;
                      valid types are any of the types for `--pretty`, as well as:
                      `flowgraph=<nodeid>` (graphviz formatted flowgraph for node),
                      `everybody_loops` (all function bodies replaced with `loop {}`),
                      `hir` (the HIR), `hir,identified`, or
                      `hir,typed` (HIR with types for each node).",
                     "TYPE"),
        opt::opt_u("", "show-span", "Show spans for compiler debugging", "expr|pat|ty"),
    ]);
    opts
}

// Convert strings provided as --cfg [cfgspec] into a crate_cfg
pub fn parse_cfgspecs(cfgspecs: Vec<String> ) -> ast::CrateConfig {
    cfgspecs.into_iter().map(|s| {
        parse::parse_meta_from_source_str("cfgspec".to_string(),
                                          s.to_string(),
                                          Vec::new(),
                                          &parse::ParseSess::new())
    }).collect::<ast::CrateConfig>()
}

pub fn build_session_options(matches: &getopts::Matches) -> Options {
    let color = match matches.opt_str("color").as_ref().map(|s| &s[..]) {
        Some("auto")   => Auto,
        Some("always") => Always,
        Some("never")  => Never,

        None => Auto,

        Some(arg) => {
            early_error(Auto, &format!("argument for --color must be auto, always \
                                        or never (instead was `{}`)",
                                       arg))
        }
    };

    let unparsed_crate_types = matches.opt_strs("crate-type");
    let crate_types = parse_crate_types_from_list(unparsed_crate_types)
        .unwrap_or_else(|e| early_error(color, &e[..]));

    let mut lint_opts = vec!();
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
            early_error(color, &format!("unknown lint level: `{}`", cap))
        })
    });

    let debugging_opts = build_debugging_options(matches, color);

    let parse_only = debugging_opts.parse_only;
    let no_trans = debugging_opts.no_trans;
    let treat_err_as_bug = debugging_opts.treat_err_as_bug;
    let no_analysis = debugging_opts.no_analysis;

    if debugging_opts.debug_llvm {
        unsafe { llvm::LLVMSetDebug(1); }
    }

    let mut output_types = HashMap::new();
    if !debugging_opts.parse_only && !no_trans {
        for list in matches.opt_strs("emit") {
            for output_type in list.split(',') {
                let mut parts = output_type.splitn(2, '=');
                let output_type = match parts.next().unwrap() {
                    "asm" => OutputType::Assembly,
                    "llvm-ir" => OutputType::LlvmAssembly,
                    "llvm-bc" => OutputType::Bitcode,
                    "obj" => OutputType::Object,
                    "link" => OutputType::Exe,
                    "dep-info" => OutputType::DepInfo,
                    part => {
                        early_error(color, &format!("unknown emission type: `{}`",
                                                    part))
                    }
                };
                let path = parts.next().map(PathBuf::from);
                output_types.insert(output_type, path);
            }
        }
    };
    if output_types.is_empty() {
        output_types.insert(OutputType::Exe, None);
    }

    let mut cg = build_codegen_options(matches, color);

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
                early_warn(color, &format!("--emit={} with -o incompatible with \
                                            -C codegen-units=N for N > 1",
                                           ot.shorthand()));
            }
            early_warn(color, "resetting to default -C codegen-units=1");
            cg.codegen_units = 1;
        }
    }

    let cg = cg;

    let sysroot_opt = matches.opt_str("sysroot").map(|m| PathBuf::from(&m));
    let target = matches.opt_str("target").unwrap_or(
        host_triple().to_string());
    let opt_level = {
        if matches.opt_present("O") {
            if cg.opt_level.is_some() {
                early_error(color, "-O and -C opt-level both provided");
            }
            Default
        } else {
            match cg.opt_level {
                None => No,
                Some(0) => No,
                Some(1) => Less,
                Some(2) => Default,
                Some(3) => Aggressive,
                Some(arg) => {
                    early_error(color, &format!("optimization level needs to be \
                                                 between 0-3 (instead was `{}`)",
                                                arg));
                }
            }
        }
    };
    let debug_assertions = cg.debug_assertions.unwrap_or(opt_level == No);
    let gc = debugging_opts.gc;
    let debuginfo = if matches.opt_present("g") {
        if cg.debuginfo.is_some() {
            early_error(color, "-g and -C debuginfo both provided");
        }
        FullDebugInfo
    } else {
        match cg.debuginfo {
            None | Some(0) => NoDebugInfo,
            Some(1) => LimitedDebugInfo,
            Some(2) => FullDebugInfo,
            Some(arg) => {
                early_error(color, &format!("debug info level needs to be between \
                                             0-2 (instead was `{}`)",
                                            arg));
            }
        }
    };

    let mut search_paths = SearchPaths::new();
    for s in &matches.opt_strs("L") {
        search_paths.add_path(&s[..], color);
    }

    let libs = matches.opt_strs("l").into_iter().map(|s| {
        let mut parts = s.splitn(2, '=');
        let kind = parts.next().unwrap();
        let (name, kind) = match (parts.next(), kind) {
            (None, name) |
            (Some(name), "dylib") => (name, cstore::NativeUnknown),
            (Some(name), "framework") => (name, cstore::NativeFramework),
            (Some(name), "static") => (name, cstore::NativeStatic),
            (_, s) => {
                early_error(color, &format!("unknown library kind `{}`, expected \
                                             one of dylib, framework, or static",
                                            s));
            }
        };
        (name.to_string(), kind)
    }).collect();

    let cfg = parse_cfgspecs(matches.opt_strs("cfg"));
    let test = matches.opt_present("test");

    let prints = matches.opt_strs("print").into_iter().map(|s| {
        match &*s {
            "crate-name" => PrintRequest::CrateName,
            "file-names" => PrintRequest::FileNames,
            "sysroot" => PrintRequest::Sysroot,
            req => {
                early_error(color, &format!("unknown print request `{}`", req))
            }
        }
    }).collect::<Vec<_>>();

    if !cg.remark.is_empty() && debuginfo == NoDebugInfo {
        early_warn(color, "-C remark will not show source locations without \
                           --debuginfo");
    }

    let mut externs = HashMap::new();
    for arg in &matches.opt_strs("extern") {
        let mut parts = arg.splitn(2, '=');
        let name = match parts.next() {
            Some(s) => s,
            None => early_error(color, "--extern value must not be empty"),
        };
        let location = match parts.next() {
            Some(s) => s,
            None => early_error(color, "--extern value must be of the format `foo=bar`"),
        };

        externs.entry(name.to_string()).or_insert(vec![]).push(location.to_string());
    }

    let crate_name = matches.opt_str("crate-name");

    Options {
        crate_types: crate_types,
        gc: gc,
        optimize: opt_level,
        debuginfo: debuginfo,
        lint_opts: lint_opts,
        lint_cap: lint_cap,
        describe_lints: describe_lints,
        output_types: output_types,
        search_paths: search_paths,
        maybe_sysroot: sysroot_opt,
        target_triple: target,
        cfg: cfg,
        test: test,
        parse_only: parse_only,
        no_trans: no_trans,
        treat_err_as_bug: treat_err_as_bug,
        no_analysis: no_analysis,
        debugging_opts: debugging_opts,
        prints: prints,
        cg: cg,
        color: color,
        show_span: None,
        externs: externs,
        crate_name: crate_name,
        alt_std_name: None,
        libs: libs,
        unstable_features: get_unstable_features_setting(),
        debug_assertions: debug_assertions,
    }
}

pub fn get_unstable_features_setting() -> UnstableFeatures {
    // Whether this is a feature-staged build, i.e. on the beta or stable channel
    let disable_unstable_features = option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some();
    // The secret key needed to get through the rustc build itself by
    // subverting the unstable features lints
    let bootstrap_secret_key = option_env!("CFG_BOOTSTRAP_KEY");
    // The matching key to the above, only known by the build system
    let bootstrap_provided_key = env::var("RUSTC_BOOTSTRAP_KEY").ok();
    match (disable_unstable_features, bootstrap_secret_key, bootstrap_provided_key) {
        (_, Some(ref s), Some(ref p)) if s == p => UnstableFeatures::Cheat,
        (true, _, _) => UnstableFeatures::Disallow,
        (false, _, _) => UnstableFeatures::Allow
    }
}

pub fn parse_crate_types_from_list(list_list: Vec<String>) -> Result<Vec<CrateType>, String> {

    let mut crate_types: Vec<CrateType> = Vec::new();
    for unparsed_crate_type in &list_list {
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
            if !crate_types.contains(&new_part) {
                crate_types.push(new_part)
            }
        }
    }

    return Ok(crate_types);
}

impl fmt::Display for CrateType {
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
mod tests {
    use middle::cstore::DummyCrateStore;
    use session::config::{build_configuration, optgroups, build_session_options};
    use session::build_session;

    use std::rc::Rc;
    use getopts::getopts;
    use syntax::attr;
    use syntax::attr::AttrMetaMethods;
    use syntax::diagnostics;

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let matches =
            &match getopts(&["--test".to_string()], &optgroups()) {
              Ok(m) => m,
              Err(f) => panic!("test_switch_implies_cfg_test: {}", f)
            };
        let registry = diagnostics::registry::Registry::new(&[]);
        let sessopts = build_session_options(matches);
        let sess = build_session(sessopts, None, registry, Rc::new(DummyCrateStore));
        let cfg = build_configuration(&sess);
        assert!((attr::contains_name(&cfg[..], "test")));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let matches =
            &match getopts(&["--test".to_string(), "--cfg=test".to_string()],
                           &optgroups()) {
              Ok(m) => m,
              Err(f) => {
                panic!("test_switch_implies_cfg_test_unless_cfg_test: {}", f)
              }
            };
        let registry = diagnostics::registry::Registry::new(&[]);
        let sessopts = build_session_options(matches);
        let sess = build_session(sessopts, None, registry,
                                 Rc::new(DummyCrateStore));
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
            ], &optgroups()).unwrap();
            let registry = diagnostics::registry::Registry::new(&[]);
            let sessopts = build_session_options(&matches);
            let sess = build_session(sessopts, None, registry,
                                     Rc::new(DummyCrateStore));
            assert!(!sess.can_print_warnings);
        }

        {
            let matches = getopts(&[
                "-Awarnings".to_string(),
                "-Dwarnings".to_string()
            ], &optgroups()).unwrap();
            let registry = diagnostics::registry::Registry::new(&[]);
            let sessopts = build_session_options(&matches);
            let sess = build_session(sessopts, None, registry,
                                     Rc::new(DummyCrateStore));
            assert!(sess.can_print_warnings);
        }

        {
            let matches = getopts(&[
                "-Adead_code".to_string()
            ], &optgroups()).unwrap();
            let registry = diagnostics::registry::Registry::new(&[]);
            let sessopts = build_session_options(&matches);
            let sess = build_session(sessopts, None, registry,
                                     Rc::new(DummyCrateStore));
            assert!(sess.can_print_warnings);
        }
    }
}
