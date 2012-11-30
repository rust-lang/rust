// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use back::link;
use back::target_strs;
use back;
use driver;
use driver::session;
use metadata::filesearch;
use metadata;
use middle::lint;

use core::cmp;
use core::option;
use syntax::ast::node_id;
use syntax::ast::{int_ty, uint_ty, float_ty};
use syntax::codemap::span;
use syntax::diagnostic;
use syntax::parse::parse_sess;
use syntax::{ast, codemap};
use syntax;

enum os { os_win32, os_macos, os_linux, os_android, os_freebsd, }

impl os : cmp::Eq {
    pure fn eq(&self, other: &os) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &os) -> bool { !(*self).eq(other) }
}

enum arch { arch_x86, arch_x86_64, arch_arm, }

impl arch : cmp::Eq {
    pure fn eq(&self, other: &arch) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &arch) -> bool { !(*self).eq(other) }
}

enum crate_type { bin_crate, lib_crate, unknown_crate, }

type config =
    {os: os,
     arch: arch,
     target_strs: target_strs::t,
     int_type: int_ty,
     uint_type: uint_ty,
     float_type: float_ty};

const verbose: uint = 1 << 0;
const time_passes: uint = 1 << 1;
const count_llvm_insns: uint = 1 << 2;
const time_llvm_passes: uint = 1 << 3;
const trans_stats: uint = 1 << 4;
const no_asm_comments: uint = 1 << 5;
const no_verify: uint = 1 << 6;
const trace: uint = 1 << 7;
const coherence: uint = 1 << 8;
const borrowck_stats: uint = 1 << 9;
const borrowck_note_pure: uint = 1 << 10;
const borrowck_note_loan: uint = 1 << 11;
const no_landing_pads: uint = 1 << 12;
const debug_llvm: uint = 1 << 13;
const count_type_sizes: uint = 1 << 14;
const meta_stats: uint = 1 << 15;
const no_opt: uint = 1 << 16;
const no_monomorphic_collapse: uint = 1 << 17;

fn debugging_opts_map() -> ~[(~str, ~str, uint)] {
    ~[(~"verbose", ~"in general, enable more debug printouts", verbose),
     (~"time-passes", ~"measure time of each rustc pass", time_passes),
     (~"count-llvm-insns", ~"count where LLVM \
                           instrs originate", count_llvm_insns),
     (~"time-llvm-passes", ~"measure time of each LLVM pass",
      time_llvm_passes),
     (~"trans-stats", ~"gather trans statistics", trans_stats),
     (~"no-asm-comments", ~"omit comments when using -S", no_asm_comments),
     (~"no-verify", ~"skip LLVM verification", no_verify),
     (~"trace", ~"emit trace logs", trace),
     (~"coherence", ~"perform coherence checking", coherence),
     (~"borrowck-stats", ~"gather borrowck statistics",  borrowck_stats),
     (~"borrowck-note-pure", ~"note where purity is req'd",
      borrowck_note_pure),
     (~"borrowck-note-loan", ~"note where loans are req'd",
      borrowck_note_loan),
     (~"no-landing-pads", ~"omit landing pads for unwinding",
      no_landing_pads),
     (~"debug-llvm", ~"enable debug output from LLVM", debug_llvm),
     (~"count-type-sizes", ~"count the sizes of aggregate types",
      count_type_sizes),
     (~"meta-stats", ~"gather metadata statistics", meta_stats),
     (~"no-opt", ~"do not optimize, even if -O is passed", no_opt),
     (~"no-monomorphic-collapse", ~"do not collapse template instantiations",
      no_monomorphic_collapse),
    ]
}

enum OptLevel {
    No, // -O0
    Less, // -O1
    Default, // -O2
    Aggressive // -O3
}

impl OptLevel : cmp::Eq {
    pure fn eq(&self, other: &OptLevel) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &OptLevel) -> bool { !(*self).eq(other) }
}

type options =
    // The crate config requested for the session, which may be combined
    // with additional crate configurations during the compile process
    {crate_type: crate_type,
     static: bool,
     gc: bool,
     optimize: OptLevel,
     debuginfo: bool,
     extra_debuginfo: bool,
     lint_opts: ~[(lint::lint, lint::level)],
     save_temps: bool,
     jit: bool,
     output_type: back::link::output_type,
     addl_lib_search_paths: ~[Path],
     maybe_sysroot: Option<Path>,
     target_triple: ~str,
     // User-specified cfg meta items. The compiler itself will add additional
     // items to the crate config, and during parsing the entire crate config
     // will be added to the crate AST node.  This should not be used for
     // anything except building the full crate config prior to parsing.
     cfg: ast::crate_cfg,
     binary: ~str,
     test: bool,
     parse_only: bool,
     no_trans: bool,
     debugging_opts: uint,
    };

type crate_metadata = {name: ~str, data: ~[u8]};

type Session_ = {targ_cfg: @config,
                 opts: @options,
                 cstore: metadata::cstore::CStore,
                 parse_sess: parse_sess,
                 codemap: @codemap::CodeMap,
                 // For a library crate, this is always none
                 mut main_fn: Option<(node_id, codemap::span)>,
                 span_diagnostic: diagnostic::span_handler,
                 filesearch: filesearch::FileSearch,
                 mut building_library: bool,
                 working_dir: Path,
                 lint_settings: lint::lint_settings};

enum Session {
    Session_(@Session_)
}

impl Session {
    fn span_fatal(sp: span, msg: ~str) -> ! {
        self.span_diagnostic.span_fatal(sp, msg)
    }
    fn fatal(msg: ~str) -> ! {
        self.span_diagnostic.handler().fatal(msg)
    }
    fn span_err(sp: span, msg: ~str) {
        self.span_diagnostic.span_err(sp, msg)
    }
    fn err(msg: ~str) {
        self.span_diagnostic.handler().err(msg)
    }
    fn has_errors() -> bool {
        self.span_diagnostic.handler().has_errors()
    }
    fn abort_if_errors() {
        self.span_diagnostic.handler().abort_if_errors()
    }
    fn span_warn(sp: span, msg: ~str) {
        self.span_diagnostic.span_warn(sp, msg)
    }
    fn warn(msg: ~str) {
        self.span_diagnostic.handler().warn(msg)
    }
    fn span_note(sp: span, msg: ~str) {
        self.span_diagnostic.span_note(sp, msg)
    }
    fn note(msg: ~str) {
        self.span_diagnostic.handler().note(msg)
    }
    fn span_bug(sp: span, msg: ~str) -> ! {
        self.span_diagnostic.span_bug(sp, msg)
    }
    fn bug(msg: ~str) -> ! {
        self.span_diagnostic.handler().bug(msg)
    }
    fn span_unimpl(sp: span, msg: ~str) -> ! {
        self.span_diagnostic.span_unimpl(sp, msg)
    }
    fn unimpl(msg: ~str) -> ! {
        self.span_diagnostic.handler().unimpl(msg)
    }
    fn span_lint_level(level: lint::level, sp: span, +msg: ~str) {
        match level {
          lint::allow => { },
          lint::warn => self.span_warn(sp, msg),
          lint::deny | lint::forbid => {
            self.span_err(sp, msg);
          }
        }
    }
    fn span_lint(lint_mode: lint::lint,
                 expr_id: ast::node_id,
                 item_id: ast::node_id,
                 span: span,
                 +msg: ~str) {
        let level = lint::get_lint_settings_level(
            self.lint_settings, lint_mode, expr_id, item_id);
        self.span_lint_level(level, span, msg);
    }
    fn next_node_id() -> ast::node_id {
        return syntax::parse::next_node_id(self.parse_sess);
    }
    fn diagnostic() -> diagnostic::span_handler {
        self.span_diagnostic
    }
    fn debugging_opt(opt: uint) -> bool {
        (self.opts.debugging_opts & opt) != 0u
    }
    // This exists to help with refactoring to eliminate impossible
    // cases later on
    fn impossible_case(sp: span, msg: &str) -> ! {
        self.span_bug(sp, fmt!("Impossible case reached: %s", msg));
    }
    fn verbose() -> bool { self.debugging_opt(verbose) }
    fn time_passes() -> bool { self.debugging_opt(time_passes) }
    fn count_llvm_insns() -> bool { self.debugging_opt(count_llvm_insns) }
    fn count_type_sizes() -> bool { self.debugging_opt(count_type_sizes) }
    fn time_llvm_passes() -> bool { self.debugging_opt(time_llvm_passes) }
    fn trans_stats() -> bool { self.debugging_opt(trans_stats) }
    fn meta_stats() -> bool { self.debugging_opt(meta_stats) }
    fn no_asm_comments() -> bool { self.debugging_opt(no_asm_comments) }
    fn no_verify() -> bool { self.debugging_opt(no_verify) }
    fn trace() -> bool { self.debugging_opt(trace) }
    fn coherence() -> bool { self.debugging_opt(coherence) }
    fn borrowck_stats() -> bool { self.debugging_opt(borrowck_stats) }
    fn borrowck_note_pure() -> bool { self.debugging_opt(borrowck_note_pure) }
    fn borrowck_note_loan() -> bool { self.debugging_opt(borrowck_note_loan) }
    fn no_monomorphic_collapse() -> bool {
        self.debugging_opt(no_monomorphic_collapse)
    }

    fn str_of(id: ast::ident) -> ~str {
        /*bad*/copy *self.parse_sess.interner.get(id)
    }
    fn ident_of(+st: ~str) -> ast::ident {
        self.parse_sess.interner.intern(@st)
    }
    fn intr() -> @syntax::parse::token::ident_interner {
        self.parse_sess.interner
    }
}

/// Some reasonable defaults
fn basic_options() -> @options {
    @{
        crate_type: session::lib_crate,
        static: false,
        gc: false,
        optimize: No,
        debuginfo: false,
        extra_debuginfo: false,
        lint_opts: ~[],
        save_temps: false,
        jit: false,
        output_type: link::output_type_exe,
        addl_lib_search_paths: ~[],
        maybe_sysroot: None,
        target_triple: driver::host_triple(),
        cfg: ~[],
        binary: ~"rustc",
        test: false,
        parse_only: false,
        no_trans: false,
        debugging_opts: 0u
    }
}

// Seems out of place, but it uses session, so I'm putting it here
fn expect<T: Copy>(sess: Session, opt: Option<T>, msg: fn() -> ~str) -> T {
    diagnostic::expect(sess.diagnostic(), opt, msg)
}

fn building_library(req_crate_type: crate_type, crate: @ast::crate,
                    testing: bool) -> bool {
    match req_crate_type {
      bin_crate => false,
      lib_crate => true,
      unknown_crate => {
        if testing {
            false
        } else {
            match syntax::attr::first_attr_value_str_by_name(
                crate.node.attrs,
                ~"crate_type") {
              option::Some(~"lib") => true,
              _ => false
            }
        }
      }
    }
}

fn sess_os_to_meta_os(os: os) -> metadata::loader::os {
    use metadata::loader;

    match os {
      os_win32 => loader::os_win32,
      os_linux => loader::os_linux,
      os_android => loader::os_android,
      os_macos => loader::os_macos,
      os_freebsd => loader::os_freebsd
    }
}

#[cfg(test)]
mod test {
    #[legacy_exports];

    use core::prelude::*;

    use driver::session::{bin_crate, building_library, lib_crate};
    use driver::session::{unknown_crate};

    use syntax::ast;
    use syntax::ast_util;

    fn make_crate_type_attr(+t: ~str) -> ast::attribute {
        ast_util::respan(ast_util::dummy_sp(), {
            style: ast::attr_outer,
            value: ast_util::respan(ast_util::dummy_sp(),
                ast::meta_name_value(
                    ~"crate_type",
                    ast_util::respan(ast_util::dummy_sp(),
                                     ast::lit_str(@t)))),
            is_sugared_doc: false
        })
    }

    fn make_crate(with_bin: bool, with_lib: bool) -> @ast::crate {
        let mut attrs = ~[];
        if with_bin { attrs += ~[make_crate_type_attr(~"bin")]; }
        if with_lib { attrs += ~[make_crate_type_attr(~"lib")]; }
        @ast_util::respan(ast_util::dummy_sp(), {
            module: {view_items: ~[], items: ~[]},
            attrs: attrs,
            config: ~[]
        })
    }

    #[test]
    fn bin_crate_type_attr_results_in_bin_output() {
        let crate = make_crate(true, false);
        assert !building_library(unknown_crate, crate, false);
    }

    #[test]
    fn lib_crate_type_attr_results_in_lib_output() {
        let crate = make_crate(false, true);
        assert building_library(unknown_crate, crate, false);
    }

    #[test]
    fn bin_option_overrides_lib_crate_type() {
        let crate = make_crate(false, true);
        assert !building_library(bin_crate, crate, false);
    }

    #[test]
    fn lib_option_overrides_bin_crate_type() {
        let crate = make_crate(true, false);
        assert building_library(lib_crate, crate, false);
    }

    #[test]
    fn bin_crate_type_is_default() {
        let crate = make_crate(false, false);
        assert !building_library(unknown_crate, crate, false);
    }

    #[test]
    fn test_option_overrides_lib_crate_type() {
        let crate = make_crate(false, true);
        assert !building_library(unknown_crate, crate, true);
    }

    #[test]
    fn test_option_does_not_override_requested_lib_type() {
        let crate = make_crate(false, false);
        assert building_library(lib_crate, crate, true);
    }
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
