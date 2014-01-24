// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::link;
use back::target_strs;
use back;
use driver::driver::host_triple;
use metadata::filesearch;
use metadata;
use middle::lint;

use syntax::attr::AttrMetaMethods;
use syntax::ast::NodeId;
use syntax::ast::{IntTy, UintTy};
use syntax::codemap::Span;
use syntax::diagnostic;
use syntax::parse::ParseSess;
use syntax::{ast, codemap};
use syntax::abi;
use syntax::parse::token;
use syntax;

use std::cell::{Cell, RefCell};
use std::hashmap::{HashMap,HashSet};

pub struct Config {
    os: abi::Os,
    arch: abi::Architecture,
    target_strs: target_strs::t,
    int_type: IntTy,
    uint_type: UintTy,
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
        COUNT_TYPE_SIZES,
        META_STATS,
        NO_OPT,
        GC,
        DEBUG_INFO,
        EXTRA_DEBUG_INFO,
        PRINT_LINK_ARGS,
        PRINT_LLVM_PASSES,
        NO_VECTORIZE_LOOPS,
        NO_VECTORIZE_SLP,
        NO_PREPOPULATE_PASSES,
        USE_SOFTFP,
        GEN_CRATE_MAP,
        PREFER_DYNAMIC,
        NO_INTEGRATED_AS,
        LTO
    ]
    0
)

pub fn debugging_opts_map() -> ~[(&'static str, &'static str, u64)] {
    ~[("verbose", "in general, enable more debug printouts", VERBOSE),
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
     ("count-type-sizes", "count the sizes of aggregate types",
      COUNT_TYPE_SIZES),
     ("meta-stats", "gather metadata statistics", META_STATS),
     ("no-opt", "do not optimize, even if -O is passed", NO_OPT),
     ("print-link-args", "Print the arguments passed to the linker",
      PRINT_LINK_ARGS),
     ("gc", "Garbage collect shared data (experimental)", GC),
     ("extra-debug-info", "Extra debugging info (experimental)",
      EXTRA_DEBUG_INFO),
     ("debug-info", "Produce debug info (experimental)", DEBUG_INFO),
     ("print-llvm-passes",
      "Prints the llvm optimization passes being run",
      PRINT_LLVM_PASSES),
     ("no-prepopulate-passes",
      "Don't pre-populate the pass managers with a list of passes, only use \
        the passes from --passes",
      NO_PREPOPULATE_PASSES),
     ("no-vectorize-loops",
      "Don't run the loop vectorization optimization passes",
      NO_VECTORIZE_LOOPS),
     ("no-vectorize-slp", "Don't run LLVM's SLP vectorization passes",
      NO_VECTORIZE_SLP),
     ("soft-float", "Generate software floating point library calls", USE_SOFTFP),
     ("gen-crate-map", "Force generation of a toplevel crate map", GEN_CRATE_MAP),
     ("prefer-dynamic", "Prefer dynamic linking to static linking", PREFER_DYNAMIC),
     ("no-integrated-as",
      "Use external assembler rather than LLVM's integrated one", NO_INTEGRATED_AS),
     ("lto", "Perform LLVM link-time optimizations", LTO),
    ]
}

#[deriving(Clone, Eq)]
pub enum OptLevel {
    No, // -O0
    Less, // -O1
    Default, // -O2
    Aggressive // -O3
}

#[deriving(Clone)]
pub struct Options {
    // The crate config requested for the session, which may be combined
    // with additional crate configurations during the compile process
    outputs: ~[OutputStyle],

    gc: bool,
    optimize: OptLevel,
    custom_passes: ~[~str],
    llvm_args: ~[~str],
    debuginfo: bool,
    extra_debuginfo: bool,
    lint_opts: ~[(lint::Lint, lint::level)],
    save_temps: bool,
    output_type: back::link::OutputType,
    // This is mutable for rustpkg, which updates search paths based on the
    // parsed code.
    addl_lib_search_paths: @RefCell<HashSet<Path>>,
    ar: Option<~str>,
    linker: Option<~str>,
    linker_args: ~[~str],
    maybe_sysroot: Option<@Path>,
    target_triple: ~str,
    target_cpu: ~str,
    target_feature: ~str,
    // User-specified cfg meta items. The compiler itself will add additional
    // items to the crate config, and during parsing the entire crate config
    // will be added to the crate AST node.  This should not be used for
    // anything except building the full crate config prior to parsing.
    cfg: ast::CrateConfig,
    binary: ~str,
    test: bool,
    parse_only: bool,
    no_trans: bool,
    no_analysis: bool,
    no_rpath: bool,
    debugging_opts: u64,
    android_cross_path: Option<~str>,
    /// Whether to write dependency files. It's (enabled, optional filename).
    write_dependency_info: (bool, Option<Path>),
    /// Crate id-related things to maybe print. It's (crate_id, crate_name, crate_file_name).
    print_metas: (bool, bool, bool),
}

// The type of entry function, so
// users can have their own entry
// functions that don't start a
// scheduler
#[deriving(Eq)]
pub enum EntryFnType {
    EntryMain,
    EntryStart,
    EntryNone,
}

#[deriving(Eq, Clone, TotalOrd, TotalEq)]
pub enum OutputStyle {
    OutputExecutable,
    OutputDylib,
    OutputRlib,
    OutputStaticlib,
}

pub struct Session_ {
    targ_cfg: @Config,
    opts: @Options,
    cstore: @metadata::cstore::CStore,
    parse_sess: @ParseSess,
    codemap: @codemap::CodeMap,
    // For a library crate, this is always none
    entry_fn: RefCell<Option<(NodeId, codemap::Span)>>,
    entry_type: Cell<Option<EntryFnType>>,
    span_diagnostic: @diagnostic::SpanHandler,
    macro_registrar_fn: RefCell<Option<ast::DefId>>,
    filesearch: @filesearch::FileSearch,
    building_library: Cell<bool>,
    working_dir: Path,
    lints: RefCell<HashMap<ast::NodeId,
                           ~[(lint::Lint, codemap::Span, ~str)]>>,
    node_id: Cell<ast::NodeId>,
    outputs: @RefCell<~[OutputStyle]>,
}

pub type Session = @Session_;

impl Session_ {
    pub fn span_fatal(&self, sp: Span, msg: &str) -> ! {
        self.span_diagnostic.span_fatal(sp, msg)
    }
    pub fn fatal(&self, msg: &str) -> ! {
        self.span_diagnostic.handler().fatal(msg)
    }
    pub fn span_err(&self, sp: Span, msg: &str) {
        self.span_diagnostic.span_err(sp, msg)
    }
    pub fn err(&self, msg: &str) {
        self.span_diagnostic.handler().err(msg)
    }
    pub fn err_count(&self) -> uint {
        self.span_diagnostic.handler().err_count()
    }
    pub fn has_errors(&self) -> bool {
        self.span_diagnostic.handler().has_errors()
    }
    pub fn abort_if_errors(&self) {
        self.span_diagnostic.handler().abort_if_errors()
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        self.span_diagnostic.span_warn(sp, msg)
    }
    pub fn warn(&self, msg: &str) {
        self.span_diagnostic.handler().warn(msg)
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.span_diagnostic.span_note(sp, msg)
    }
    pub fn span_end_note(&self, sp: Span, msg: &str) {
        self.span_diagnostic.span_end_note(sp, msg)
    }
    pub fn note(&self, msg: &str) {
        self.span_diagnostic.handler().note(msg)
    }
    pub fn span_bug(&self, sp: Span, msg: &str) -> ! {
        self.span_diagnostic.span_bug(sp, msg)
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.span_diagnostic.handler().bug(msg)
    }
    pub fn span_unimpl(&self, sp: Span, msg: &str) -> ! {
        self.span_diagnostic.span_unimpl(sp, msg)
    }
    pub fn unimpl(&self, msg: &str) -> ! {
        self.span_diagnostic.handler().unimpl(msg)
    }
    pub fn add_lint(&self,
                    lint: lint::Lint,
                    id: ast::NodeId,
                    sp: Span,
                    msg: ~str) {
        let mut lints = self.lints.borrow_mut();
        match lints.get().find_mut(&id) {
            Some(arr) => { arr.push((lint, sp, msg)); return; }
            None => {}
        }
        lints.get().insert(id, ~[(lint, sp, msg)]);
    }
    pub fn next_node_id(&self) -> ast::NodeId {
        self.reserve_node_ids(1)
    }
    pub fn reserve_node_ids(&self, count: ast::NodeId) -> ast::NodeId {
        let v = self.node_id.get();

        match v.checked_add(&count) {
            Some(next) => { self.node_id.set(next); }
            None => self.bug("Input too large, ran out of node ids!")
        }

        v
    }
    pub fn diagnostic(&self) -> @diagnostic::SpanHandler {
        self.span_diagnostic
    }
    pub fn debugging_opt(&self, opt: u64) -> bool {
        (self.opts.debugging_opts & opt) != 0
    }
    // This exists to help with refactoring to eliminate impossible
    // cases later on
    pub fn impossible_case(&self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp, format!("Impossible case reached: {}", msg));
    }
    pub fn verbose(&self) -> bool { self.debugging_opt(VERBOSE) }
    pub fn time_passes(&self) -> bool { self.debugging_opt(TIME_PASSES) }
    pub fn count_llvm_insns(&self) -> bool {
        self.debugging_opt(COUNT_LLVM_INSNS)
    }
    pub fn count_type_sizes(&self) -> bool {
        self.debugging_opt(COUNT_TYPE_SIZES)
    }
    pub fn time_llvm_passes(&self) -> bool {
        self.debugging_opt(TIME_LLVM_PASSES)
    }
    pub fn trans_stats(&self) -> bool { self.debugging_opt(TRANS_STATS) }
    pub fn meta_stats(&self) -> bool { self.debugging_opt(META_STATS) }
    pub fn asm_comments(&self) -> bool { self.debugging_opt(ASM_COMMENTS) }
    pub fn no_verify(&self) -> bool { self.debugging_opt(NO_VERIFY) }
    pub fn borrowck_stats(&self) -> bool { self.debugging_opt(BORROWCK_STATS) }
    pub fn print_llvm_passes(&self) -> bool {
        self.debugging_opt(PRINT_LLVM_PASSES)
    }
    pub fn no_prepopulate_passes(&self) -> bool {
        self.debugging_opt(NO_PREPOPULATE_PASSES)
    }
    pub fn no_vectorize_loops(&self) -> bool {
        self.debugging_opt(NO_VECTORIZE_LOOPS)
    }
    pub fn no_vectorize_slp(&self) -> bool {
        self.debugging_opt(NO_VECTORIZE_SLP)
    }
    pub fn gen_crate_map(&self) -> bool {
        self.debugging_opt(GEN_CRATE_MAP)
    }
    pub fn prefer_dynamic(&self) -> bool {
        self.debugging_opt(PREFER_DYNAMIC)
    }
    pub fn no_integrated_as(&self) -> bool {
        self.debugging_opt(NO_INTEGRATED_AS)
    }
    pub fn lto(&self) -> bool {
        self.debugging_opt(LTO)
    }
    pub fn no_landing_pads(&self) -> bool {
        self.debugging_opt(NO_LANDING_PADS)
    }

    // pointless function, now...
    pub fn str_of(&self, id: ast::Ident) -> @str {
        token::ident_to_str(&id)
    }

    // pointless function, now...
    pub fn ident_of(&self, st: &str) -> ast::Ident {
        token::str_to_ident(st)
    }

    // pointless function, now...
    pub fn intr(&self) -> @syntax::parse::token::IdentInterner {
        token::get_ident_interner()
    }
}

/// Some reasonable defaults
pub fn basic_options() -> @Options {
    @Options {
        outputs: ~[],
        gc: false,
        optimize: No,
        custom_passes: ~[],
        llvm_args: ~[],
        debuginfo: false,
        extra_debuginfo: false,
        lint_opts: ~[],
        save_temps: false,
        output_type: link::OutputTypeExe,
        addl_lib_search_paths: @RefCell::new(HashSet::new()),
        ar: None,
        linker: None,
        linker_args: ~[],
        maybe_sysroot: None,
        target_triple: host_triple(),
        target_cpu: ~"generic",
        target_feature: ~"",
        cfg: ~[],
        binary: ~"rustc",
        test: false,
        parse_only: false,
        no_trans: false,
        no_analysis: false,
        no_rpath: false,
        debugging_opts: 0,
        android_cross_path: None,
        write_dependency_info: (false, None),
        print_metas: (false, false, false),
    }
}

// Seems out of place, but it uses session, so I'm putting it here
pub fn expect<T:Clone>(sess: Session, opt: Option<T>, msg: || -> ~str) -> T {
    diagnostic::expect(sess.diagnostic(), opt, msg)
}

pub fn building_library(options: &Options, crate: &ast::Crate) -> bool {
    if options.test { return false }
    for output in options.outputs.iter() {
        match *output {
            OutputExecutable => {}
            OutputStaticlib | OutputDylib | OutputRlib => return true
        }
    }
    match syntax::attr::first_attr_value_str_by_name(crate.attrs, "crate_type") {
        Some(s) => "lib" == s || "rlib" == s || "dylib" == s || "staticlib" == s,
        _ => false
    }
}

pub fn collect_outputs(session: &Session,
                       attrs: &[ast::Attribute]) -> ~[OutputStyle] {
    // If we're generating a test executable, then ignore all other output
    // styles at all other locations
    if session.opts.test {
        return ~[OutputExecutable];
    }
    let mut base = session.opts.outputs.clone();
    let mut iter = attrs.iter().filter_map(|a| {
        if "crate_type" == a.name() {
            match a.value_str() {
                Some(n) if "rlib" == n => Some(OutputRlib),
                Some(n) if "dylib" == n => Some(OutputDylib),
                Some(n) if "lib" == n => Some(OutputDylib),
                Some(n) if "staticlib" == n => Some(OutputStaticlib),
                Some(n) if "bin" == n => Some(OutputExecutable),
                Some(_) => {
                    session.add_lint(lint::UnknownCrateType, ast::CRATE_NODE_ID,
                                     a.span, ~"invalid `crate_type` value");
                    None
                }
                _ => {
                    session.add_lint(lint::UnknownCrateType, ast::CRATE_NODE_ID,
                                    a.span, ~"`crate_type` requires a value");
                    None
                }
            }
        } else {
            None
        }
    });
    base.extend(&mut iter);
    if base.len() == 0 {
        base.push(OutputExecutable);
    }
    base.sort();
    base.dedup();
    return base;
}

pub fn sess_os_to_meta_os(os: abi::Os) -> metadata::loader::Os {
    use metadata::loader;

    match os {
        abi::OsWin32 => loader::OsWin32,
        abi::OsLinux => loader::OsLinux,
        abi::OsAndroid => loader::OsAndroid,
        abi::OsMacos => loader::OsMacos,
        abi::OsFreebsd => loader::OsFreebsd
    }
}
