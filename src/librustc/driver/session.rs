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
use syntax::ast::{int_ty, uint_ty};
use syntax::codemap::Span;
use syntax::diagnostic;
use syntax::parse::ParseSess;
use syntax::{ast, codemap};
use syntax::abi;
use syntax::parse::token;
use syntax;

use std::cell::{Cell, RefCell};
use std::hashmap::{HashMap,HashSet};

pub struct config {
    os: abi::Os,
    arch: abi::Architecture,
    target_strs: target_strs::t,
    int_type: int_ty,
    uint_type: uint_ty,
}

pub static verbose:                 uint = 1 <<  0;
pub static time_passes:             uint = 1 <<  1;
pub static count_llvm_insns:        uint = 1 <<  2;
pub static time_llvm_passes:        uint = 1 <<  3;
pub static trans_stats:             uint = 1 <<  4;
pub static asm_comments:            uint = 1 <<  5;
pub static no_verify:               uint = 1 <<  6;
pub static borrowck_stats:          uint = 1 <<  7;
pub static borrowck_note_pure:      uint = 1 <<  8;
pub static borrowck_note_loan:      uint = 1 <<  9;
pub static no_landing_pads:         uint = 1 << 10;
pub static debug_llvm:              uint = 1 << 11;
pub static count_type_sizes:        uint = 1 << 12;
pub static meta_stats:              uint = 1 << 13;
pub static no_opt:                  uint = 1 << 14;
pub static gc:                      uint = 1 << 15;
pub static debug_info:              uint = 1 << 16;
pub static extra_debug_info:        uint = 1 << 17;
pub static print_link_args:         uint = 1 << 18;
pub static no_debug_borrows:        uint = 1 << 19;
pub static lint_llvm:               uint = 1 << 20;
pub static print_llvm_passes:       uint = 1 << 21;
pub static no_vectorize_loops:      uint = 1 << 22;
pub static no_vectorize_slp:        uint = 1 << 23;
pub static no_prepopulate_passes:   uint = 1 << 24;
pub static use_softfp:              uint = 1 << 25;
pub static gen_crate_map:           uint = 1 << 26;
pub static prefer_dynamic:          uint = 1 << 27;
pub static no_integrated_as:        uint = 1 << 28;
pub static lto:                     uint = 1 << 29;

pub fn debugging_opts_map() -> ~[(&'static str, &'static str, uint)] {
    ~[("verbose", "in general, enable more debug printouts", verbose),
     ("time-passes", "measure time of each rustc pass", time_passes),
     ("count-llvm-insns", "count where LLVM \
                           instrs originate", count_llvm_insns),
     ("time-llvm-passes", "measure time of each LLVM pass",
      time_llvm_passes),
     ("trans-stats", "gather trans statistics", trans_stats),
     ("asm-comments", "generate comments into the assembly (may change behavior)", asm_comments),
     ("no-verify", "skip LLVM verification", no_verify),
     ("borrowck-stats", "gather borrowck statistics",  borrowck_stats),
     ("borrowck-note-pure", "note where purity is req'd",
      borrowck_note_pure),
     ("borrowck-note-loan", "note where loans are req'd",
      borrowck_note_loan),
     ("no-landing-pads", "omit landing pads for unwinding",
      no_landing_pads),
     ("debug-llvm", "enable debug output from LLVM", debug_llvm),
     ("count-type-sizes", "count the sizes of aggregate types",
      count_type_sizes),
     ("meta-stats", "gather metadata statistics", meta_stats),
     ("no-opt", "do not optimize, even if -O is passed", no_opt),
     ("print-link-args", "Print the arguments passed to the linker", print_link_args),
     ("gc", "Garbage collect shared data (experimental)", gc),
     ("extra-debug-info", "Extra debugging info (experimental)",
      extra_debug_info),
     ("debug-info", "Produce debug info (experimental)", debug_info),
     ("no-debug-borrows",
      "do not show where borrow checks fail",
      no_debug_borrows),
     ("lint-llvm",
      "Run the LLVM lint pass on the pre-optimization IR",
      lint_llvm),
     ("print-llvm-passes",
      "Prints the llvm optimization passes being run",
      print_llvm_passes),
     ("no-prepopulate-passes",
      "Don't pre-populate the pass managers with a list of passes, only use \
        the passes from --passes",
      no_prepopulate_passes),
     ("no-vectorize-loops",
      "Don't run the loop vectorization optimization passes",
      no_vectorize_loops),
     ("no-vectorize-slp",
      "Don't run LLVM's SLP vectorization passes",
      no_vectorize_slp),
     ("soft-float", "Generate software floating point library calls", use_softfp),
     ("gen-crate-map", "Force generation of a toplevel crate map", gen_crate_map),
     ("prefer-dynamic", "Prefer dynamic linking to static linking", prefer_dynamic),
     ("no-integrated-as",
      "Use external assembler rather than LLVM's integrated one", no_integrated_as),
     ("lto", "Perform LLVM link-time optimizations", lto),
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
pub struct options {
    // The crate config requested for the session, which may be combined
    // with additional crate configurations during the compile process
    outputs: ~[OutputStyle],

    gc: bool,
    optimize: OptLevel,
    custom_passes: ~[~str],
    llvm_args: ~[~str],
    debuginfo: bool,
    extra_debuginfo: bool,
    lint_opts: ~[(lint::lint, lint::level)],
    save_temps: bool,
    output_type: back::link::output_type,
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
    debugging_opts: uint,
    android_cross_path: Option<~str>,
    /// Whether to write dependency files. It's (enabled, optional filename).
    write_dependency_info: (bool, Option<Path>),
    /// Crate id-related things to maybe print. It's (crate_id, crate_name, crate_file_name).
    print_metas: (bool, bool, bool),
}

pub struct crate_metadata {
    name: ~str,
    data: ~[u8]
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

#[deriving(Eq, Clone)]
pub enum OutputStyle {
    OutputExecutable,
    OutputDylib,
    OutputRlib,
    OutputStaticlib,
}

pub struct Session_ {
    targ_cfg: @config,
    opts: @options,
    cstore: @metadata::cstore::CStore,
    parse_sess: @mut ParseSess,
    codemap: @codemap::CodeMap,
    // For a library crate, this is always none
    entry_fn: RefCell<Option<(NodeId, codemap::Span)>>,
    entry_type: Cell<Option<EntryFnType>>,
    span_diagnostic: @mut diagnostic::SpanHandler,
    filesearch: @filesearch::FileSearch,
    building_library: Cell<bool>,
    working_dir: Path,
    lints: RefCell<HashMap<ast::NodeId,
                           ~[(lint::lint, codemap::Span, ~str)]>>,
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
                    lint: lint::lint,
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
    pub fn diagnostic(&self) -> @mut diagnostic::SpanHandler {
        self.span_diagnostic
    }
    pub fn debugging_opt(&self, opt: uint) -> bool {
        (self.opts.debugging_opts & opt) != 0u
    }
    // This exists to help with refactoring to eliminate impossible
    // cases later on
    pub fn impossible_case(&self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp, format!("Impossible case reached: {}", msg));
    }
    pub fn verbose(&self) -> bool { self.debugging_opt(verbose) }
    pub fn time_passes(&self) -> bool { self.debugging_opt(time_passes) }
    pub fn count_llvm_insns(&self) -> bool {
        self.debugging_opt(count_llvm_insns)
    }
    pub fn count_type_sizes(&self) -> bool {
        self.debugging_opt(count_type_sizes)
    }
    pub fn time_llvm_passes(&self) -> bool {
        self.debugging_opt(time_llvm_passes)
    }
    pub fn trans_stats(&self) -> bool { self.debugging_opt(trans_stats) }
    pub fn meta_stats(&self) -> bool { self.debugging_opt(meta_stats) }
    pub fn asm_comments(&self) -> bool { self.debugging_opt(asm_comments) }
    pub fn no_verify(&self) -> bool { self.debugging_opt(no_verify) }
    pub fn lint_llvm(&self) -> bool { self.debugging_opt(lint_llvm) }
    pub fn borrowck_stats(&self) -> bool { self.debugging_opt(borrowck_stats) }
    pub fn borrowck_note_pure(&self) -> bool {
        self.debugging_opt(borrowck_note_pure)
    }
    pub fn borrowck_note_loan(&self) -> bool {
        self.debugging_opt(borrowck_note_loan)
    }
    pub fn debug_borrows(&self) -> bool {
        self.opts.optimize == No && !self.debugging_opt(no_debug_borrows)
    }
    pub fn print_llvm_passes(&self) -> bool {
        self.debugging_opt(print_llvm_passes)
    }
    pub fn no_prepopulate_passes(&self) -> bool {
        self.debugging_opt(no_prepopulate_passes)
    }
    pub fn no_vectorize_loops(&self) -> bool {
        self.debugging_opt(no_vectorize_loops)
    }
    pub fn no_vectorize_slp(&self) -> bool {
        self.debugging_opt(no_vectorize_slp)
    }
    pub fn gen_crate_map(&self) -> bool {
        self.debugging_opt(gen_crate_map)
    }
    pub fn prefer_dynamic(&self) -> bool {
        self.debugging_opt(prefer_dynamic)
    }
    pub fn no_integrated_as(&self) -> bool {
        self.debugging_opt(no_integrated_as)
    }
    pub fn lto(&self) -> bool {
        self.debugging_opt(lto)
    }
    pub fn no_landing_pads(&self) -> bool {
        self.debugging_opt(no_landing_pads)
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
    pub fn intr(&self) -> @syntax::parse::token::ident_interner {
        token::get_ident_interner()
    }
}

/// Some reasonable defaults
pub fn basic_options() -> @options {
    @options {
        outputs: ~[],
        gc: false,
        optimize: No,
        custom_passes: ~[],
        llvm_args: ~[],
        debuginfo: false,
        extra_debuginfo: false,
        lint_opts: ~[],
        save_temps: false,
        output_type: link::output_type_exe,
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
        debugging_opts: 0u,
        android_cross_path: None,
        write_dependency_info: (false, None),
        print_metas: (false, false, false),
    }
}

// Seems out of place, but it uses session, so I'm putting it here
pub fn expect<T:Clone>(sess: Session, opt: Option<T>, msg: || -> ~str) -> T {
    diagnostic::expect(sess.diagnostic(), opt, msg)
}

pub fn building_library(options: &options, crate: &ast::Crate) -> bool {
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

pub fn collect_outputs(options: &options,
                       attrs: &[ast::Attribute]) -> ~[OutputStyle] {
    // If we're generating a test executable, then ignore all other output
    // styles at all other locations
    if options.test {
        return ~[OutputExecutable];
    }
    let mut base = options.outputs.clone();
    let mut iter = attrs.iter().filter_map(|a| {
        if "crate_type" == a.name() {
            match a.value_str() {
                Some(n) if "rlib" == n => Some(OutputRlib),
                Some(n) if "dylib" == n => Some(OutputDylib),
                Some(n) if "lib" == n => Some(OutputDylib),
                Some(n) if "staticlib" == n => Some(OutputStaticlib),
                Some(n) if "bin" == n => Some(OutputExecutable),
                _ => None
            }
        } else {
            None
        }
    });
    base.extend(&mut iter);
    if base.len() == 0 {
        base.push(OutputExecutable);
    }
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
