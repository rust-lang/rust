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

use driver::{early_error, early_warn};
use driver::driver;
use driver::session::Session;

use back;
use back::write;
use back::target_strs;
use back::{arm, x86, x86_64, mips, mipsel};
use lint;

use syntax::abi;
use syntax::ast;
use syntax::ast::{IntTy, UintTy};
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::diagnostic::{ColorConfig, Auto, Always, Never};
use syntax::parse;
use syntax::parse::token::InternedString;

use std::collections::{HashSet, HashMap};
use getopts::{optopt, optmulti, optflag, optflagopt};
use getopts;
use std::cell::{RefCell};
use std::fmt;

use llvm;

pub struct Config {
    pub os: abi::Os,
    pub arch: abi::Architecture,
    pub target_strs: target_strs::t,
    pub int_type: IntTy,
    pub uint_type: UintTy,
}

#[deriving(Clone, PartialEq)]
pub enum OptLevel {
    No, // -O0
    Less, // -O1
    Default, // -O2
    Aggressive // -O3
}

#[deriving(Clone, PartialEq)]
pub enum DebugInfoLevel {
    NoDebugInfo,
    LimitedDebugInfo,
    FullDebugInfo,
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
    pub output_types: Vec<back::write::OutputType> ,
    // This was mutable for rustpkg, which updates search paths based on the
    // parsed code. It remains mutable in case its replacements wants to use
    // this.
    pub addl_lib_search_paths: RefCell<HashSet<Path>>,
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
    /// Crate id-related things to maybe print. It's (crate_name, crate_file_name).
    pub print_metas: (bool, bool),
    pub cg: CodegenOptions,
    pub color: ColorConfig,
    pub externs: HashMap<String, Vec<String>>,
    pub crate_name: Option<String>,
    /// An optional name to use as the crate for std during std injection,
    /// written `extern crate std = "name"`. Default to "std". Used by
    /// out-of-tree drivers.
    pub alt_std_name: Option<String>
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
        addl_lib_search_paths: RefCell::new(HashSet::new()),
        maybe_sysroot: None,
        target_triple: driver::host_triple().to_string(),
        cfg: Vec::new(),
        test: false,
        parse_only: false,
        no_trans: false,
        no_analysis: false,
        debugging_opts: 0,
        write_dependency_info: (false, None),
        print_metas: (false, false),
        cg: basic_codegen_options(),
        color: Auto,
        externs: HashMap::new(),
        crate_name: None,
        alt_std_name: None,
    }
}

// The type of entry function, so
// users can have their own entry
// functions that don't start a
// scheduler
#[deriving(PartialEq)]
pub enum EntryFnType {
    EntryMain,
    EntryStart,
    EntryNone,
}

#[deriving(PartialEq, PartialOrd, Clone, Ord, Eq, Hash)]
pub enum CrateType {
    CrateTypeExecutable,
    CrateTypeDylib,
    CrateTypeRlib,
    CrateTypeStaticlib,
}

macro_rules! debugging_opts(
    ([ $opt:ident ] $cnt:expr ) => (
        pub static $opt: u64 = 1 << $cnt;
    );
    ([ $opt:ident, $($rest:ident),* ] $cnt:expr ) => (
        pub static $opt: u64 = 1 << $cnt;
        debugging_opts!([ $($rest),* ] $cnt + 1)
    )
)

debugging_opts!(
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
        NO_OPT,
        GC,
        PRINT_LINK_ARGS,
        PRINT_LLVM_PASSES,
        LTO,
        AST_JSON,
        AST_JSON_NOEXPAND,
        LS,
        SAVE_ANALYSIS,
        FLOWGRAPH_PRINT_LOANS,
        FLOWGRAPH_PRINT_MOVES,
        FLOWGRAPH_PRINT_ASSIGNS,
        FLOWGRAPH_PRINT_ALL
    ]
    0
)

pub fn debugging_opts_map() -> Vec<(&'static str, &'static str, u64)> {
    vec!(("verbose", "in general, enable more debug printouts", VERBOSE),
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
     ("no-opt", "do not optimize, even if -O is passed", NO_OPT),
     ("print-link-args", "Print the arguments passed to the linker",
      PRINT_LINK_ARGS),
     ("gc", "Garbage collect shared data (experimental)", GC),
     ("print-llvm-passes",
      "Prints the llvm optimization passes being run",
      PRINT_LLVM_PASSES),
     ("lto", "Perform LLVM link-time optimizations", LTO),
     ("ast-json", "Print the AST as JSON and halt", AST_JSON),
     ("ast-json-noexpand", "Print the pre-expansion AST as JSON and halt", AST_JSON_NOEXPAND),
     ("ls", "List the symbols defined by a library crate", LS),
     ("save-analysis", "Write syntax and type analysis information \
                        in addition to normal output", SAVE_ANALYSIS),
     ("flowgraph-print-loans", "Include loan analysis data in \
                       --pretty flowgraph output", FLOWGRAPH_PRINT_LOANS),
     ("flowgraph-print-moves", "Include move analysis data in \
                       --pretty flowgraph output", FLOWGRAPH_PRINT_MOVES),
     ("flowgraph-print-assigns", "Include assignment analysis data in \
                       --pretty flowgraph output", FLOWGRAPH_PRINT_ASSIGNS),
     ("flowgraph-print-all", "Include all dataflow analysis data in \
                       --pretty flowgraph output", FLOWGRAPH_PRINT_ALL))
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
macro_rules! cgoptions(
    ($($opt:ident : $t:ty = ($init:expr, $parse:ident, $desc:expr)),* ,) =>
(
    #[deriving(Clone)]
    pub struct CodegenOptions { $(pub $opt: $t),* }

    pub fn basic_codegen_options() -> CodegenOptions {
        CodegenOptions { $($opt: $init),* }
    }

    pub type CodegenSetter = fn(&mut CodegenOptions, v: Option<&str>) -> bool;
    pub static CG_OPTIONS: &'static [(&'static str, CodegenSetter,
                                      &'static str)] =
        &[ $( (stringify!($opt), cgsetters::$opt, $desc) ),* ];

    mod cgsetters {
        use super::CodegenOptions;

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

        fn parse_uint(slot: &mut uint, v: Option<&str>) -> bool {
            use std::from_str::FromStr;
            match v.and_then(FromStr::from_str) {
                Some(i) => { *slot = i; true },
                None => false
            }
        }
    }
) )

cgoptions!(
    ar: Option<String> = (None, parse_opt_string,
        "tool to assemble archives with"),
    linker: Option<String> = (None, parse_opt_string,
        "system linker to link outputs with"),
    link_args: Vec<String> = (Vec::new(), parse_list,
        "extra arguments to pass to the linker (space separated)"),
    target_cpu: String = ("generic".to_string(), parse_string,
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
    no_redzone: bool = (false, parse_bool,
        "disable the use of the redzone"),
    relocation_model: String = ("pic".to_string(), parse_string,
         "choose the relocation model to use (llc -relocation-model for details)"),
    code_model: String = ("default".to_string(), parse_string,
         "choose the code model to use (llc -code-model for details)"),
    metadata: Vec<String> = (Vec::new(), parse_list,
         "metadata to mangle symbol names with"),
    extra_filename: String = ("".to_string(), parse_string,
         "extra data to put in each output filename"),
    codegen_units: uint = (1, parse_uint,
        "divide crate into N units to optimize in parallel"),
)

pub fn build_codegen_options(matches: &getopts::Matches) -> CodegenOptions
{
    let mut cg = basic_codegen_options();
    for option in matches.opt_strs("C").move_iter() {
        let mut iter = option.as_slice().splitn(1, '=');
        let key = iter.next().unwrap();
        let value = iter.next();
        let option_to_lookup = key.replace("-", "_");
        let mut found = false;
        for &(candidate, setter, _) in CG_OPTIONS.iter() {
            if option_to_lookup.as_slice() != candidate { continue }
            if !setter(&mut cg, value) {
                match value {
                    Some(..) => {
                        early_error(format!("codegen option `{}` takes no \
                                             value", key).as_slice())
                    }
                    None => {
                        early_error(format!("codegen option `{0}` requires \
                                             a value (-C {0}=<value>)",
                                            key).as_slice())
                    }
                }
            }
            found = true;
            break;
        }
        if !found {
            early_error(format!("unknown codegen option: `{}`",
                                key).as_slice());
        }
    }
    return cg;
}

pub fn default_lib_output() -> CrateType {
    CrateTypeRlib
}

pub fn default_configuration(sess: &Session) -> ast::CrateConfig {
    let tos = match sess.targ_cfg.os {
        abi::OsWindows =>   InternedString::new("windows"),
        abi::OsMacos =>     InternedString::new("macos"),
        abi::OsLinux =>     InternedString::new("linux"),
        abi::OsAndroid =>   InternedString::new("android"),
        abi::OsFreebsd =>   InternedString::new("freebsd"),
        abi::OsDragonfly => InternedString::new("dragonfly"),
        abi::OsiOS =>       InternedString::new("ios"),
    };

    // ARM is bi-endian, however using NDK seems to default
    // to little-endian unless a flag is provided.
    let (end,arch,wordsz) = match sess.targ_cfg.arch {
        abi::X86 =>    ("little", "x86",    "32"),
        abi::X86_64 => ("little", "x86_64", "64"),
        abi::Arm =>    ("little", "arm",    "32"),
        abi::Mips =>   ("big",    "mips",   "32"),
        abi::Mipsel => ("little", "mipsel", "32")
    };

    let fam = match sess.targ_cfg.os {
        abi::OsWindows => InternedString::new("windows"),
        _ => InternedString::new("unix")
    };

    let mk = attr::mk_name_value_item_str;
    return vec!(// Target bindings.
         attr::mk_word_item(fam.clone()),
         mk(InternedString::new("target_os"), tos),
         mk(InternedString::new("target_family"), fam),
         mk(InternedString::new("target_arch"), InternedString::new(arch)),
         mk(InternedString::new("target_endian"), InternedString::new(end)),
         mk(InternedString::new("target_word_size"),
            InternedString::new(wordsz))
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
    user_cfg.move_iter().collect::<Vec<_>>().append(default_cfg.as_slice())
}

pub fn get_os(triple: &str) -> Option<abi::Os> {
    for &(name, os) in os_names.iter() {
        if triple.contains(name) { return Some(os) }
    }
    None
}
static os_names : &'static [(&'static str, abi::Os)] = &[
    ("mingw32",   abi::OsWindows),
    ("win32",     abi::OsWindows),
    ("windows",   abi::OsWindows),
    ("darwin",    abi::OsMacos),
    ("android",   abi::OsAndroid),
    ("linux",     abi::OsLinux),
    ("freebsd",   abi::OsFreebsd),
    ("dragonfly", abi::OsDragonfly),
    ("ios",       abi::OsiOS)];

pub fn get_arch(triple: &str) -> Option<abi::Architecture> {
    for &(arch, abi) in architecture_abis.iter() {
        if triple.contains(arch) { return Some(abi) }
    }
    None
}
static architecture_abis : &'static [(&'static str, abi::Architecture)] = &[
    ("i386",   abi::X86),
    ("i486",   abi::X86),
    ("i586",   abi::X86),
    ("i686",   abi::X86),
    ("i786",   abi::X86),

    ("x86_64", abi::X86_64),

    ("arm",    abi::Arm),
    ("xscale", abi::Arm),
    ("thumb",  abi::Arm),

    ("mipsel", abi::Mipsel),
    ("mips",   abi::Mips)];

pub fn build_target_config(sopts: &Options) -> Config {
    let os = match get_os(sopts.target_triple.as_slice()) {
      Some(os) => os,
      None => early_error("unknown operating system")
    };
    let arch = match get_arch(sopts.target_triple.as_slice()) {
      Some(arch) => arch,
      None => {
          early_error(format!("unknown architecture: {}",
                              sopts.target_triple.as_slice()).as_slice())
      }
    };
    let (int_type, uint_type) = match arch {
      abi::X86 => (ast::TyI32, ast::TyU32),
      abi::X86_64 => (ast::TyI64, ast::TyU64),
      abi::Arm => (ast::TyI32, ast::TyU32),
      abi::Mips => (ast::TyI32, ast::TyU32),
      abi::Mipsel => (ast::TyI32, ast::TyU32)
    };
    let target_triple = sopts.target_triple.clone();
    let target_strs = match arch {
      abi::X86 => x86::get_target_strs(target_triple, os),
      abi::X86_64 => x86_64::get_target_strs(target_triple, os),
      abi::Arm => arm::get_target_strs(target_triple, os),
      abi::Mips => mips::get_target_strs(target_triple, os),
      abi::Mipsel => mipsel::get_target_strs(target_triple, os)
    };
    Config {
        os: os,
        arch: arch,
        target_strs: target_strs,
        int_type: int_type,
        uint_type: uint_type,
    }
}

// rustc command line options
pub fn optgroups() -> Vec<getopts::OptGroup> {
    vec!(
        optflag("h", "help", "Display this message"),
        optmulti("", "cfg", "Configure the compilation environment", "SPEC"),
        optmulti("L", "",   "Add a directory to the library search path", "PATH"),
        optmulti("", "crate-type", "Comma separated list of types of crates
                                    for the compiler to emit",
                 "[bin|lib|rlib|dylib|staticlib]"),
        optmulti("", "emit", "Comma separated list of types of output for the compiler to emit",
                 "[asm|bc|ir|obj|link]"),
        optopt("", "crate-name", "Specify the name of the crate being built",
               "NAME"),
        optflag("", "print-crate-name", "Output the crate name and exit"),
        optflag("", "print-file-name", "Output the file(s) that would be written if compilation \
              continued and exit"),
        optflag("", "crate-file-name", "deprecated in favor of --print-file-name"),
        optflag("g",  "",  "Equivalent to --debuginfo=2"),
        optopt("",  "debuginfo",  "Emit DWARF debug info to the objects created:
             0 = no debug info,
             1 = line-tables only (for stacktraces and breakpoints),
             2 = full debug info with variable and type information (same as -g)", "LEVEL"),
        optflag("", "no-trans", "Run all passes except translation; no output"),
        optflag("", "no-analysis",
              "Parse and expand the source, but run no analysis and produce no output"),
        optflag("O", "", "Equivalent to --opt-level=2"),
        optopt("o", "", "Write output to <filename>", "FILENAME"),
        optopt("", "opt-level", "Optimize with possible levels 0-3", "LEVEL"),
        optopt( "",  "out-dir", "Write output to compiler-chosen filename in <dir>", "DIR"),
        optflag("", "parse-only", "Parse only; do not compile, assemble, or link"),
        optopt("", "explain", "Provide a detailed explanation of an error message", "OPT"),
        optflagopt("", "pretty",
                   "Pretty-print the input instead of compiling;
                   valid types are: `normal` (un-annotated source),
                   `expanded` (crates expanded),
                   `typed` (crates expanded, with type annotations),
                   `expanded,identified` (fully parenthesized, AST nodes with IDs), or
                   `flowgraph=<nodeid>` (graphviz formatted flowgraph for node)",
                 "TYPE"),
        optflagopt("", "dep-info",
                 "Output dependency info to <filename> after compiling, \
                  in a format suitable for use by Makefiles", "FILENAME"),
        optopt("", "sysroot", "Override the system root", "PATH"),
        optflag("", "test", "Build a test harness"),
        optopt("", "target", "Target triple cpu-manufacturer-kernel[-os]
                            to compile for (see chapter 3.4 of http://www.sourceware.org/autobook/
                            for details)", "TRIPLE"),
        optmulti("W", "warn", "Set lint warnings", "OPT"),
        optmulti("A", "allow", "Set lint allowed", "OPT"),
        optmulti("D", "deny", "Set lint denied", "OPT"),
        optmulti("F", "forbid", "Set lint forbidden", "OPT"),
        optmulti("C", "codegen", "Set a codegen option", "OPT[=VALUE]"),
        optmulti("Z", "", "Set internal debugging options", "FLAG"),
        optflagopt("v", "version", "Print version info and exit", "verbose"),
        optopt("", "color", "Configure coloring of output:
            auto   = colorize, if output goes to a tty (default);
            always = always colorize output;
            never  = never colorize output", "auto|always|never"),
        optmulti("", "extern", "Specify where an external rust library is located",
                 "NAME=PATH"),
    )
}


// Convert strings provided as --cfg [cfgspec] into a crate_cfg
fn parse_cfgspecs(cfgspecs: Vec<String> ) -> ast::CrateConfig {
    cfgspecs.move_iter().map(|s| {
        parse::parse_meta_from_source_str("cfgspec".to_string(),
                                          s.to_string(),
                                          Vec::new(),
                                          &parse::new_parse_sess())
    }).collect::<ast::CrateConfig>()
}

pub fn build_session_options(matches: &getopts::Matches) -> Options {

    let unparsed_crate_types = matches.opt_strs("crate-type");
    let crate_types = parse_crate_types_from_list(unparsed_crate_types)
        .unwrap_or_else(|e| early_error(e.as_slice()));

    let parse_only = matches.opt_present("parse-only");
    let no_trans = matches.opt_present("no-trans");
    let no_analysis = matches.opt_present("no-analysis");

    let mut lint_opts = vec!();
    let mut describe_lints = false;

    for &level in [lint::Allow, lint::Warn, lint::Deny, lint::Forbid].iter() {
        for lint_name in matches.opt_strs(level.as_str()).move_iter() {
            if lint_name.as_slice() == "help" {
                describe_lints = true;
            } else {
                lint_opts.push((lint_name.replace("-", "_").into_string(), level));
            }
        }
    }

    let mut debugging_opts = 0;
    let debug_flags = matches.opt_strs("Z");
    let debug_map = debugging_opts_map();
    for debug_flag in debug_flags.iter() {
        let mut this_bit = 0;
        for tuple in debug_map.iter() {
            let (name, bit) = match *tuple { (ref a, _, b) => (a, b) };
            if *name == debug_flag.as_slice() {
                this_bit = bit;
                break;
            }
        }
        if this_bit == 0 {
            early_error(format!("unknown debug flag: {}",
                                *debug_flag).as_slice())
        }
        debugging_opts |= this_bit;
    }

    if debugging_opts & DEBUG_LLVM != 0 {
        unsafe { llvm::LLVMSetDebug(1); }
    }

    let mut output_types = Vec::new();
    if !parse_only && !no_trans {
        let unparsed_output_types = matches.opt_strs("emit");
        for unparsed_output_type in unparsed_output_types.iter() {
            for part in unparsed_output_type.as_slice().split(',') {
                let output_type = match part.as_slice() {
                    "asm"  => write::OutputTypeAssembly,
                    "ir"   => write::OutputTypeLlvmAssembly,
                    "bc"   => write::OutputTypeBitcode,
                    "obj"  => write::OutputTypeObject,
                    "link" => write::OutputTypeExe,
                    _ => {
                        early_error(format!("unknown emission type: `{}`",
                                            part).as_slice())
                    }
                };
                output_types.push(output_type)
            }
        }
    };
    output_types.as_mut_slice().sort();
    output_types.dedup();
    if output_types.len() == 0 {
        output_types.push(write::OutputTypeExe);
    }

    let sysroot_opt = matches.opt_str("sysroot").map(|m| Path::new(m));
    let target = matches.opt_str("target").unwrap_or(
        driver::host_triple().to_string());
    let opt_level = {
        if (debugging_opts & NO_OPT) != 0 {
            No
        } else if matches.opt_present("O") {
            if matches.opt_present("opt-level") {
                early_error("-O and --opt-level both provided");
            }
            Default
        } else if matches.opt_present("opt-level") {
            match matches.opt_str("opt-level").as_ref().map(|s| s.as_slice()) {
                None      |
                Some("0") => No,
                Some("1") => Less,
                Some("2") => Default,
                Some("3") => Aggressive,
                Some(arg) => {
                    early_error(format!("optimization level needs to be \
                                         between 0-3 (instead was `{}`)",
                                        arg).as_slice());
                }
            }
        } else {
            No
        }
    };
    let gc = debugging_opts & GC != 0;
    let debuginfo = if matches.opt_present("g") {
        if matches.opt_present("debuginfo") {
            early_error("-g and --debuginfo both provided");
        }
        FullDebugInfo
    } else if matches.opt_present("debuginfo") {
        match matches.opt_str("debuginfo").as_ref().map(|s| s.as_slice()) {
            Some("0") => NoDebugInfo,
            Some("1") => LimitedDebugInfo,
            None      |
            Some("2") => FullDebugInfo,
            Some(arg) => {
                early_error(format!("optimization level needs to be between \
                                     0-3 (instead was `{}`)",
                                    arg).as_slice());
            }
        }
    } else {
        NoDebugInfo
    };

    let addl_lib_search_paths = matches.opt_strs("L").iter().map(|s| {
        Path::new(s.as_slice())
    }).collect();

    let cfg = parse_cfgspecs(matches.opt_strs("cfg"));
    let test = matches.opt_present("test");
    let write_dependency_info = (matches.opt_present("dep-info"),
                                 matches.opt_str("dep-info")
                                        .map(|p| Path::new(p)));

    let print_metas = (matches.opt_present("print-crate-name"),
                       matches.opt_present("print-file-name") ||
                       matches.opt_present("crate-file-name"));
    if matches.opt_present("crate-file-name") {
        early_warn("the --crate-file-name argument has been renamed to \
                    --print-file-name");
    }
    let cg = build_codegen_options(matches);

    let color = match matches.opt_str("color").as_ref().map(|s| s.as_slice()) {
        Some("auto")   => Auto,
        Some("always") => Always,
        Some("never")  => Never,

        None => Auto,

        Some(arg) => {
            early_error(format!("argument for --color must be auto, always \
                                 or never (instead was `{}`)",
                                arg).as_slice())
        }
    };

    let mut externs = HashMap::new();
    for arg in matches.opt_strs("extern").iter() {
        let mut parts = arg.as_slice().splitn(1, '=');
        let name = match parts.next() {
            Some(s) => s,
            None => early_error("--extern value must not be empty"),
        };
        let location = match parts.next() {
            Some(s) => s,
            None => early_error("--extern value must be of the format `foo=bar`"),
        };
        let locs = externs.find_or_insert(name.to_string(), Vec::new());
        locs.push(location.to_string());
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
        print_metas: print_metas,
        cg: cg,
        color: color,
        externs: externs,
        crate_name: crate_name,
        alt_std_name: None
    }
}

pub fn parse_crate_types_from_list(list_list: Vec<String>) -> Result<Vec<CrateType>, String> {

    let mut crate_types: Vec<CrateType> = Vec::new();
    for unparsed_crate_type in list_list.iter() {
        for part in unparsed_crate_type.as_slice().split(',') {
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

    use driver::config::{build_configuration, optgroups, build_session_options};
    use driver::session::build_session;

    use getopts::getopts;
    use syntax::attr;
    use syntax::attr::AttrMetaMethods;
    use syntax::diagnostics;

    // When the user supplies --test we should implicitly supply --cfg test
    #[test]
    fn test_switch_implies_cfg_test() {
        let matches =
            &match getopts(["--test".to_string()], optgroups().as_slice()) {
              Ok(m) => m,
              Err(f) => fail!("test_switch_implies_cfg_test: {}", f)
            };
        let registry = diagnostics::registry::Registry::new([]);
        let sessopts = build_session_options(matches);
        let sess = build_session(sessopts, None, registry);
        let cfg = build_configuration(&sess);
        assert!((attr::contains_name(cfg.as_slice(), "test")));
    }

    // When the user supplies --test and --cfg test, don't implicitly add
    // another --cfg test
    #[test]
    fn test_switch_implies_cfg_test_unless_cfg_test() {
        let matches =
            &match getopts(["--test".to_string(), "--cfg=test".to_string()],
                           optgroups().as_slice()) {
              Ok(m) => m,
              Err(f) => {
                fail!("test_switch_implies_cfg_test_unless_cfg_test: {}", f)
              }
            };
        let registry = diagnostics::registry::Registry::new([]);
        let sessopts = build_session_options(matches);
        let sess = build_session(sessopts, None, registry);
        let cfg = build_configuration(&sess);
        let mut test_items = cfg.iter().filter(|m| m.name().equiv(&("test")));
        assert!(test_items.next().is_some());
        assert!(test_items.next().is_none());
    }
}
