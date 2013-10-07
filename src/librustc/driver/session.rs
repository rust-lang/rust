// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use driver::session;
use metadata::filesearch;
use metadata;
use middle::lint;

use syntax::ast::NodeId;
use syntax::ast::{int_ty, uint_ty};
use syntax::codemap::Span;
use syntax::diagnostic;
use syntax::parse::ParseSess;
use syntax::{ast, codemap};
use syntax::abi;
use syntax::parse::token;
use syntax;

use std::int;
use std::hashmap::HashMap;

#[deriving(Eq)]
pub enum Os { OsWin32, OsMacos, OsLinux, OsAndroid, OsFreebsd, }

#[deriving(Clone)]
pub enum crate_type {
    bin_crate,
    lib_crate,
    unknown_crate,
}

pub struct config {
    os: Os,
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
pub static trace:                   uint = 1 <<  7;
pub static coherence:               uint = 1 <<  8;
pub static borrowck_stats:          uint = 1 <<  9;
pub static borrowck_note_pure:      uint = 1 << 10;
pub static borrowck_note_loan:      uint = 1 << 11;
pub static no_landing_pads:         uint = 1 << 12;
pub static debug_llvm:              uint = 1 << 13;
pub static count_type_sizes:        uint = 1 << 14;
pub static meta_stats:              uint = 1 << 15;
pub static no_opt:                  uint = 1 << 16;
pub static gc:                      uint = 1 << 17;
pub static jit:                     uint = 1 << 18;
pub static debug_info:              uint = 1 << 19;
pub static extra_debug_info:        uint = 1 << 20;
pub static statik:                  uint = 1 << 21;
pub static print_link_args:         uint = 1 << 22;
pub static no_debug_borrows:        uint = 1 << 23;
pub static lint_llvm:               uint = 1 << 24;
pub static once_fns:                uint = 1 << 25;
pub static print_llvm_passes:       uint = 1 << 26;
pub static no_vectorize_loops:      uint = 1 << 27;
pub static no_vectorize_slp:        uint = 1 << 28;
pub static no_prepopulate_passes:   uint = 1 << 29;
pub static use_softfp:              uint = 1 << 30;

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
     ("trace", "emit trace logs", trace),
     ("coherence", "perform coherence checking", coherence),
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
     ("jit", "Execute using JIT (experimental)", jit),
     ("extra-debug-info", "Extra debugging info (experimental)",
      extra_debug_info),
     ("debug-info", "Produce debug info (experimental)", debug_info),
     ("static", "Use or produce static libraries or binaries (experimental)", statik),
     ("no-debug-borrows",
      "do not show where borrow checks fail",
      no_debug_borrows),
     ("lint-llvm",
      "Run the LLVM lint pass on the pre-optimization IR",
      lint_llvm),
     ("once-fns",
      "Allow 'once fn' closures to deinitialize captured variables",
      once_fns),
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
    crate_type: crate_type,
    is_static: bool,
    gc: bool,
    optimize: OptLevel,
    custom_passes: ~[~str],
    llvm_args: ~[~str],
    debuginfo: bool,
    extra_debuginfo: bool,
    lint_opts: ~[(lint::lint, lint::level)],
    save_temps: bool,
    jit: bool,
    output_type: back::link::output_type,
    addl_lib_search_paths: @mut ~[Path], // This is mutable for rustpkg, which
                                         // updates search paths based on the
                                         // parsed code
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
    binary: @str,
    test: bool,
    parse_only: bool,
    no_trans: bool,
    debugging_opts: uint,
    android_cross_path: Option<~str>,
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

pub struct Session_ {
    targ_cfg: @config,
    opts: @options,
    cstore: @mut metadata::cstore::CStore,
    parse_sess: @mut ParseSess,
    codemap: @codemap::CodeMap,
    // For a library crate, this is always none
    entry_fn: @mut Option<(NodeId, codemap::Span)>,
    entry_type: @mut Option<EntryFnType>,
    span_diagnostic: @mut diagnostic::span_handler,
    filesearch: @filesearch::FileSearch,
    building_library: @mut bool,
    working_dir: Path,
    lints: @mut HashMap<ast::NodeId, ~[(lint::lint, codemap::Span, ~str)]>,
    node_id: @mut uint,
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
        match self.lints.find_mut(&id) {
            Some(arr) => { arr.push((lint, sp, msg)); return; }
            None => {}
        }
        self.lints.insert(id, ~[(lint, sp, msg)]);
    }
    pub fn next_node_id(&self) -> ast::NodeId {
        self.reserve_node_ids(1)
    }
    pub fn reserve_node_ids(&self, count: uint) -> ast::NodeId {
        let v = *self.node_id;
        *self.node_id += count;
        if v > (int::max_value as uint) {
            self.bug("Input too large, ran out of node ids!");
        }
        v as int
    }
    pub fn diagnostic(&self) -> @mut diagnostic::span_handler {
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
    pub fn trace(&self) -> bool { self.debugging_opt(trace) }
    pub fn coherence(&self) -> bool { self.debugging_opt(coherence) }
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
    pub fn once_fns(&self) -> bool { self.debugging_opt(once_fns) }
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
        crate_type: session::lib_crate,
        is_static: false,
        gc: false,
        optimize: No,
        custom_passes: ~[],
        llvm_args: ~[],
        debuginfo: false,
        extra_debuginfo: false,
        lint_opts: ~[],
        save_temps: false,
        jit: false,
        output_type: link::output_type_exe,
        addl_lib_search_paths: @mut ~[],
        linker: None,
        linker_args: ~[],
        maybe_sysroot: None,
        target_triple: host_triple(),
        target_cpu: ~"generic",
        target_feature: ~"",
        cfg: ~[],
        binary: @"rustc",
        test: false,
        parse_only: false,
        no_trans: false,
        debugging_opts: 0u,
        android_cross_path: None,
    }
}

// Seems out of place, but it uses session, so I'm putting it here
pub fn expect<T:Clone>(sess: Session, opt: Option<T>, msg: &fn() -> ~str)
                       -> T {
    diagnostic::expect(sess.diagnostic(), opt, msg)
}

pub fn building_library(req_crate_type: crate_type,
                        crate: &ast::Crate,
                        testing: bool) -> bool {
    match req_crate_type {
      bin_crate => false,
      lib_crate => true,
      unknown_crate => {
        if testing {
            false
        } else {
            match syntax::attr::first_attr_value_str_by_name(
                crate.attrs,
                "crate_type") {
              Some(s) => "lib" == s,
              _ => false
            }
        }
      }
    }
}

pub fn sess_os_to_meta_os(os: Os) -> metadata::loader::Os {
    use metadata::loader;

    match os {
        OsWin32 => loader::OsWin32,
        OsLinux => loader::OsLinux,
        OsAndroid => loader::OsAndroid,
        OsMacos => loader::OsMacos,
        OsFreebsd => loader::OsFreebsd
    }
}

#[cfg(test)]
mod test {
    use driver::session::{bin_crate, building_library, lib_crate};
    use driver::session::{unknown_crate};

    use syntax::ast;
    use syntax::attr;
    use syntax::codemap;

    fn make_crate_type_attr(t: @str) -> ast::Attribute {
        attr::mk_attr(attr::mk_name_value_item_str(@"crate_type", t))
    }

    fn make_crate(with_bin: bool, with_lib: bool) -> @ast::Crate {
        let mut attrs = ~[];
        if with_bin {
            attrs.push(make_crate_type_attr(@"bin"));
        }
        if with_lib {
            attrs.push(make_crate_type_attr(@"lib"));
        }
        @ast::Crate {
            module: ast::_mod { view_items: ~[], items: ~[] },
            attrs: attrs,
            config: ~[],
            span: codemap::dummy_sp(),
        }
    }

    #[test]
    fn bin_crate_type_attr_results_in_bin_output() {
        let crate = make_crate(true, false);
        assert!(!building_library(unknown_crate, crate, false));
    }

    #[test]
    fn lib_crate_type_attr_results_in_lib_output() {
        let crate = make_crate(false, true);
        assert!(building_library(unknown_crate, crate, false));
    }

    #[test]
    fn bin_option_overrides_lib_crate_type() {
        let crate = make_crate(false, true);
        assert!(!building_library(bin_crate, crate, false));
    }

    #[test]
    fn lib_option_overrides_bin_crate_type() {
        let crate = make_crate(true, false);
        assert!(building_library(lib_crate, crate, false));
    }

    #[test]
    fn bin_crate_type_is_default() {
        let crate = make_crate(false, false);
        assert!(!building_library(unknown_crate, crate, false));
    }

    #[test]
    fn test_option_overrides_lib_crate_type() {
        let crate = make_crate(false, true);
        assert!(!building_library(unknown_crate, crate, true));
    }

    #[test]
    fn test_option_does_not_override_requested_lib_type() {
        let crate = make_crate(false, false);
        assert!(building_library(lib_crate, crate, true));
    }
}
