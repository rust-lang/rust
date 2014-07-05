// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::config;
use driver::driver;
use front;
use metadata::cstore::CStore;
use metadata::filesearch;
use lint;
use util::nodemap::NodeMap;

use syntax::ast::NodeId;
use syntax::codemap::Span;
use syntax::diagnostic;
use syntax::parse;
use syntax::parse::token;
use syntax::parse::ParseSess;
use syntax::{ast, codemap};

use std::os;
use std::cell::{Cell, RefCell};


pub struct Session {
    pub targ_cfg: config::Config,
    pub opts: config::Options,
    pub cstore: CStore,
    pub parse_sess: ParseSess,
    // For a library crate, this is always none
    pub entry_fn: RefCell<Option<(NodeId, codemap::Span)>>,
    pub entry_type: Cell<Option<config::EntryFnType>>,
    pub plugin_registrar_fn: Cell<Option<ast::NodeId>>,
    pub default_sysroot: Option<Path>,
    // The name of the root source file of the crate, in the local file system. The path is always
    // expected to be absolute. `None` means that there is no source file.
    pub local_crate_source_file: Option<Path>,
    pub working_dir: Path,
    pub lint_store: RefCell<lint::LintStore>,
    pub lints: RefCell<NodeMap<Vec<(lint::LintId, codemap::Span, String)>>>,
    pub node_id: Cell<ast::NodeId>,
    pub crate_types: RefCell<Vec<config::CrateType>>,
    pub crate_metadata: RefCell<Vec<String>>,
    pub features: front::feature_gate::Features,

    /// The maximum recursion limit for potentially infinitely recursive
    /// operations such as auto-dereference and monomorphization.
    pub recursion_limit: Cell<uint>,
}

impl Session {
    pub fn span_fatal(&self, sp: Span, msg: &str) -> ! {
        self.diagnostic().span_fatal(sp, msg)
    }
    pub fn fatal(&self, msg: &str) -> ! {
        self.diagnostic().handler().fatal(msg)
    }
    pub fn span_err(&self, sp: Span, msg: &str) {
        self.diagnostic().span_err(sp, msg)
    }
    pub fn err(&self, msg: &str) {
        self.diagnostic().handler().err(msg)
    }
    pub fn err_count(&self) -> uint {
        self.diagnostic().handler().err_count()
    }
    pub fn has_errors(&self) -> bool {
        self.diagnostic().handler().has_errors()
    }
    pub fn abort_if_errors(&self) {
        self.diagnostic().handler().abort_if_errors()
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        self.diagnostic().span_warn(sp, msg)
    }
    pub fn warn(&self, msg: &str) {
        self.diagnostic().handler().warn(msg)
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.diagnostic().span_note(sp, msg)
    }
    pub fn span_end_note(&self, sp: Span, msg: &str) {
        self.diagnostic().span_end_note(sp, msg)
    }
    pub fn fileline_note(&self, sp: Span, msg: &str) {
        self.diagnostic().fileline_note(sp, msg)
    }
    pub fn note(&self, msg: &str) {
        self.diagnostic().handler().note(msg)
    }
    pub fn span_bug(&self, sp: Span, msg: &str) -> ! {
        self.diagnostic().span_bug(sp, msg)
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.diagnostic().handler().bug(msg)
    }
    pub fn span_unimpl(&self, sp: Span, msg: &str) -> ! {
        self.diagnostic().span_unimpl(sp, msg)
    }
    pub fn unimpl(&self, msg: &str) -> ! {
        self.diagnostic().handler().unimpl(msg)
    }
    pub fn add_lint(&self,
                    lint: &'static lint::Lint,
                    id: ast::NodeId,
                    sp: Span,
                    msg: String) {
        let lint_id = lint::LintId::of(lint);
        let mut lints = self.lints.borrow_mut();
        match lints.find_mut(&id) {
            Some(arr) => { arr.push((lint_id, sp, msg)); return; }
            None => {}
        }
        lints.insert(id, vec!((lint_id, sp, msg)));
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
    pub fn diagnostic<'a>(&'a self) -> &'a diagnostic::SpanHandler {
        &self.parse_sess.span_diagnostic
    }
    pub fn debugging_opt(&self, opt: u64) -> bool {
        (self.opts.debugging_opts & opt) != 0
    }
    pub fn codemap<'a>(&'a self) -> &'a codemap::CodeMap {
        &self.parse_sess.span_diagnostic.cm
    }
    // This exists to help with refactoring to eliminate impossible
    // cases later on
    pub fn impossible_case(&self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp,
                      format!("impossible case reached: {}", msg).as_slice());
    }
    pub fn verbose(&self) -> bool { self.debugging_opt(config::VERBOSE) }
    pub fn time_passes(&self) -> bool { self.debugging_opt(config::TIME_PASSES) }
    pub fn count_llvm_insns(&self) -> bool {
        self.debugging_opt(config::COUNT_LLVM_INSNS)
    }
    pub fn count_type_sizes(&self) -> bool {
        self.debugging_opt(config::COUNT_TYPE_SIZES)
    }
    pub fn time_llvm_passes(&self) -> bool {
        self.debugging_opt(config::TIME_LLVM_PASSES)
    }
    pub fn trans_stats(&self) -> bool { self.debugging_opt(config::TRANS_STATS) }
    pub fn meta_stats(&self) -> bool { self.debugging_opt(config::META_STATS) }
    pub fn asm_comments(&self) -> bool { self.debugging_opt(config::ASM_COMMENTS) }
    pub fn no_verify(&self) -> bool { self.debugging_opt(config::NO_VERIFY) }
    pub fn borrowck_stats(&self) -> bool { self.debugging_opt(config::BORROWCK_STATS) }
    pub fn print_llvm_passes(&self) -> bool {
        self.debugging_opt(config::PRINT_LLVM_PASSES)
    }
    pub fn lto(&self) -> bool {
        self.debugging_opt(config::LTO)
    }
    pub fn no_landing_pads(&self) -> bool {
        self.debugging_opt(config::NO_LANDING_PADS)
    }
    pub fn show_span(&self) -> bool {
        self.debugging_opt(config::SHOW_SPAN)
    }
    pub fn sysroot<'a>(&'a self) -> &'a Path {
        match self.opts.maybe_sysroot {
            Some (ref sysroot) => sysroot,
            None => self.default_sysroot.as_ref()
                        .expect("missing sysroot and default_sysroot in Session")
        }
    }
    pub fn target_filesearch<'a>(&'a self) -> filesearch::FileSearch<'a> {
        filesearch::FileSearch::new(self.sysroot(),
                                    self.opts.target_triple.as_slice(),
                                    &self.opts.addl_lib_search_paths)
    }
    pub fn host_filesearch<'a>(&'a self) -> filesearch::FileSearch<'a> {
        filesearch::FileSearch::new(
            self.sysroot(),
            driver::host_triple(),
            &self.opts.addl_lib_search_paths)
    }
}

pub fn build_session(sopts: config::Options,
                     local_crate_source_file: Option<Path>)
                     -> Session {
    let codemap = codemap::CodeMap::new();
    let diagnostic_handler =
        diagnostic::default_handler(sopts.color);
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);

    build_session_(sopts, local_crate_source_file, span_diagnostic_handler)
}

pub fn build_session_(sopts: config::Options,
                      local_crate_source_file: Option<Path>,
                      span_diagnostic: diagnostic::SpanHandler)
                      -> Session {
    let target_cfg = config::build_target_config(&sopts);
    let p_s = parse::new_parse_sess_special_handler(span_diagnostic);
    let default_sysroot = match sopts.maybe_sysroot {
        Some(_) => None,
        None => Some(filesearch::get_or_default_sysroot())
    };

    // Make the path absolute, if necessary
    let local_crate_source_file = local_crate_source_file.map(|path|
        if path.is_absolute() {
            path.clone()
        } else {
            os::getcwd().join(path.clone())
        }
    );

    let sess = Session {
        targ_cfg: target_cfg,
        opts: sopts,
        cstore: CStore::new(token::get_ident_interner()),
        parse_sess: p_s,
        // For a library crate, this is always none
        entry_fn: RefCell::new(None),
        entry_type: Cell::new(None),
        plugin_registrar_fn: Cell::new(None),
        default_sysroot: default_sysroot,
        local_crate_source_file: local_crate_source_file,
        working_dir: os::getcwd(),
        lint_store: RefCell::new(lint::LintStore::new()),
        lints: RefCell::new(NodeMap::new()),
        node_id: Cell::new(1),
        crate_types: RefCell::new(Vec::new()),
        crate_metadata: RefCell::new(Vec::new()),
        features: front::feature_gate::Features::new(),
        recursion_limit: Cell::new(64),
    };

    sess.lint_store.borrow_mut().register_builtin(Some(&sess));
    sess
}

// Seems out of place, but it uses session, so I'm putting it here
pub fn expect<T:Clone>(sess: &Session, opt: Option<T>, msg: || -> String)
              -> T {
    diagnostic::expect(sess.diagnostic(), opt, msg)
}
