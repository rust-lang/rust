// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::DepGraph;
use hir::def_id::DefIndex;
use hir::svh::Svh;
use lint;
use middle::cstore::CrateStore;
use middle::dependency_format;
use session::search_paths::PathKind;
use session::config::{DebugInfoLevel, PanicStrategy};
use ty::tls;
use util::nodemap::{NodeMap, FnvHashMap};
use mir::transform as mir_pass;

use syntax::ast::{NodeId, NodeIdAssigner, Name};
use errors::{self, DiagnosticBuilder};
use errors::emitter::{Emitter, BasicEmitter, EmitterWriter};
use syntax::json::JsonEmitter;
use syntax::feature_gate;
use syntax::parse;
use syntax::parse::ParseSess;
use syntax::parse::token;
use syntax::{ast, codemap};
use syntax::feature_gate::AttributeType;
use syntax_pos::{Span, MultiSpan};

use rustc_back::target::Target;
use llvm;

use std::path::{Path, PathBuf};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::CString;
use std::rc::Rc;
use std::fmt;
use libc::c_int;

pub mod config;
pub mod filesearch;
pub mod search_paths;

// Represents the data associated with a compilation
// session for a single crate.
pub struct Session {
    pub dep_graph: DepGraph,
    pub target: config::Config,
    pub host: Target,
    pub opts: config::Options,
    pub cstore: Rc<for<'a> CrateStore<'a>>,
    pub parse_sess: ParseSess,
    // For a library crate, this is always none
    pub entry_fn: RefCell<Option<(NodeId, Span)>>,
    pub entry_type: Cell<Option<config::EntryFnType>>,
    pub plugin_registrar_fn: Cell<Option<ast::NodeId>>,
    pub default_sysroot: Option<PathBuf>,
    // The name of the root source file of the crate, in the local file system.
    // The path is always expected to be absolute. `None` means that there is no
    // source file.
    pub local_crate_source_file: Option<PathBuf>,
    pub working_dir: PathBuf,
    pub lint_store: RefCell<lint::LintStore>,
    pub lints: RefCell<NodeMap<Vec<(lint::LintId, Span, String)>>>,
    pub plugin_llvm_passes: RefCell<Vec<String>>,
    pub mir_passes: RefCell<mir_pass::Passes>,
    pub plugin_attributes: RefCell<Vec<(String, AttributeType)>>,
    pub crate_types: RefCell<Vec<config::CrateType>>,
    pub dependency_formats: RefCell<dependency_format::Dependencies>,
    // The crate_disambiguator is constructed out of all the `-C metadata`
    // arguments passed to the compiler. Its value together with the crate-name
    // forms a unique global identifier for the crate. It is used to allow
    // multiple crates with the same name to coexist. See the
    // trans::back::symbol_names module for more information.
    pub crate_disambiguator: Cell<ast::Name>,
    pub features: RefCell<feature_gate::Features>,

    /// The maximum recursion limit for potentially infinitely recursive
    /// operations such as auto-dereference and monomorphization.
    pub recursion_limit: Cell<usize>,

    /// The metadata::creader module may inject an allocator/panic_runtime
    /// dependency if it didn't already find one, and this tracks what was
    /// injected.
    pub injected_allocator: Cell<Option<ast::CrateNum>>,
    pub injected_panic_runtime: Cell<Option<ast::CrateNum>>,

    /// Names of all bang-style macros and syntax extensions
    /// available in this crate
    pub available_macros: RefCell<HashSet<Name>>,

    /// Map from imported macro spans (which consist of
    /// the localized span for the macro body) to the
    /// macro name and defintion span in the source crate.
    pub imported_macro_spans: RefCell<HashMap<Span, (String, Span)>>,

    next_node_id: Cell<ast::NodeId>,
}

impl Session {
    pub fn struct_span_warn<'a, S: Into<MultiSpan>>(&'a self,
                                                    sp: S,
                                                    msg: &str)
                                                    -> DiagnosticBuilder<'a>  {
        self.diagnostic().struct_span_warn(sp, msg)
    }
    pub fn struct_span_warn_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                              sp: S,
                                                              msg: &str,
                                                              code: &str)
                                                              -> DiagnosticBuilder<'a>  {
        self.diagnostic().struct_span_warn_with_code(sp, msg, code)
    }
    pub fn struct_warn<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a>  {
        self.diagnostic().struct_warn(msg)
    }
    pub fn struct_span_err<'a, S: Into<MultiSpan>>(&'a self,
                                                   sp: S,
                                                   msg: &str)
                                                   -> DiagnosticBuilder<'a>  {
        match split_msg_into_multilines(msg) {
            Some(ref msg) => self.diagnostic().struct_span_err(sp, msg),
            None => self.diagnostic().struct_span_err(sp, msg),
        }
    }
    pub fn struct_span_err_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                             sp: S,
                                                             msg: &str,
                                                             code: &str)
                                                             -> DiagnosticBuilder<'a>  {
        match split_msg_into_multilines(msg) {
            Some(ref msg) => self.diagnostic().struct_span_err_with_code(sp, msg, code),
            None => self.diagnostic().struct_span_err_with_code(sp, msg, code),
        }
    }
    pub fn struct_err<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a>  {
        self.diagnostic().struct_err(msg)
    }
    pub fn struct_span_fatal<'a, S: Into<MultiSpan>>(&'a self,
                                                     sp: S,
                                                     msg: &str)
                                                     -> DiagnosticBuilder<'a>  {
        self.diagnostic().struct_span_fatal(sp, msg)
    }
    pub fn struct_span_fatal_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                               sp: S,
                                                               msg: &str,
                                                               code: &str)
                                                               -> DiagnosticBuilder<'a>  {
        self.diagnostic().struct_span_fatal_with_code(sp, msg, code)
    }
    pub fn struct_fatal<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a>  {
        self.diagnostic().struct_fatal(msg)
    }

    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        panic!(self.diagnostic().span_fatal(sp, msg))
    }
    pub fn span_fatal_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str) -> ! {
        panic!(self.diagnostic().span_fatal_with_code(sp, msg, code))
    }
    pub fn fatal(&self, msg: &str) -> ! {
        panic!(self.diagnostic().fatal(msg))
    }
    pub fn span_err_or_warn<S: Into<MultiSpan>>(&self, is_warning: bool, sp: S, msg: &str) {
        if is_warning {
            self.span_warn(sp, msg);
        } else {
            self.span_err(sp, msg);
        }
    }
    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        match split_msg_into_multilines(msg) {
            Some(msg) => self.diagnostic().span_err(sp, &msg),
            None => self.diagnostic().span_err(sp, msg)
        }
    }
    pub fn span_err_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str) {
        match split_msg_into_multilines(msg) {
            Some(msg) => self.diagnostic().span_err_with_code(sp, &msg, code),
            None => self.diagnostic().span_err_with_code(sp, msg, code)
        }
    }
    pub fn err(&self, msg: &str) {
        self.diagnostic().err(msg)
    }
    pub fn err_count(&self) -> usize {
        self.diagnostic().err_count()
    }
    pub fn has_errors(&self) -> bool {
        self.diagnostic().has_errors()
    }
    pub fn abort_if_errors(&self) {
        self.diagnostic().abort_if_errors();
    }
    pub fn track_errors<F, T>(&self, f: F) -> Result<T, usize>
        where F: FnOnce() -> T
    {
        let old_count = self.err_count();
        let result = f();
        let errors = self.err_count() - old_count;
        if errors == 0 {
            Ok(result)
        } else {
            Err(errors)
        }
    }
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().span_warn(sp, msg)
    }
    pub fn span_warn_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str) {
        self.diagnostic().span_warn_with_code(sp, msg, code)
    }
    pub fn warn(&self, msg: &str) {
        self.diagnostic().warn(msg)
    }
    pub fn opt_span_warn<S: Into<MultiSpan>>(&self, opt_sp: Option<S>, msg: &str) {
        match opt_sp {
            Some(sp) => self.span_warn(sp, msg),
            None => self.warn(msg),
        }
    }
    /// Delay a span_bug() call until abort_if_errors()
    pub fn delay_span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().delay_span_bug(sp, msg)
    }
    pub fn note_without_error(&self, msg: &str) {
        self.diagnostic().note_without_error(msg)
    }
    pub fn span_note_without_error<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().span_note_without_error(sp, msg)
    }
    pub fn span_unimpl<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.diagnostic().span_unimpl(sp, msg)
    }
    pub fn unimpl(&self, msg: &str) -> ! {
        self.diagnostic().unimpl(msg)
    }
    pub fn add_lint(&self,
                    lint: &'static lint::Lint,
                    id: ast::NodeId,
                    sp: Span,
                    msg: String) {
        let lint_id = lint::LintId::of(lint);
        let mut lints = self.lints.borrow_mut();
        match lints.get_mut(&id) {
            Some(arr) => {
                let tuple = (lint_id, sp, msg);
                if !arr.contains(&tuple) {
                    arr.push(tuple);
                }
                return;
            }
            None => {}
        }
        lints.insert(id, vec!((lint_id, sp, msg)));
    }
    pub fn reserve_node_ids(&self, count: ast::NodeId) -> ast::NodeId {
        let id = self.next_node_id.get();

        match id.checked_add(count) {
            Some(next) => self.next_node_id.set(next),
            None => bug!("Input too large, ran out of node ids!")
        }

        id
    }
    pub fn diagnostic<'a>(&'a self) -> &'a errors::Handler {
        &self.parse_sess.span_diagnostic
    }
    pub fn codemap<'a>(&'a self) -> &'a codemap::CodeMap {
        self.parse_sess.codemap()
    }
    pub fn verbose(&self) -> bool { self.opts.debugging_opts.verbose }
    pub fn time_passes(&self) -> bool { self.opts.debugging_opts.time_passes }
    pub fn count_llvm_insns(&self) -> bool {
        self.opts.debugging_opts.count_llvm_insns
    }
    pub fn time_llvm_passes(&self) -> bool {
        self.opts.debugging_opts.time_llvm_passes
    }
    pub fn trans_stats(&self) -> bool { self.opts.debugging_opts.trans_stats }
    pub fn meta_stats(&self) -> bool { self.opts.debugging_opts.meta_stats }
    pub fn asm_comments(&self) -> bool { self.opts.debugging_opts.asm_comments }
    pub fn no_verify(&self) -> bool { self.opts.debugging_opts.no_verify }
    pub fn borrowck_stats(&self) -> bool { self.opts.debugging_opts.borrowck_stats }
    pub fn print_llvm_passes(&self) -> bool {
        self.opts.debugging_opts.print_llvm_passes
    }
    pub fn lto(&self) -> bool {
        self.opts.cg.lto
    }
    pub fn no_landing_pads(&self) -> bool {
        self.opts.debugging_opts.no_landing_pads ||
            self.opts.cg.panic == PanicStrategy::Abort
    }
    pub fn unstable_options(&self) -> bool {
        self.opts.debugging_opts.unstable_options
    }
    pub fn print_enum_sizes(&self) -> bool {
        self.opts.debugging_opts.print_enum_sizes
    }
    pub fn nonzeroing_move_hints(&self) -> bool {
        self.opts.debugging_opts.enable_nonzeroing_move_hints
    }

    pub fn must_not_eliminate_frame_pointers(&self) -> bool {
        self.opts.debuginfo != DebugInfoLevel::NoDebugInfo ||
        !self.target.target.options.eliminate_frame_pointer
    }

    /// Returns the symbol name for the registrar function,
    /// given the crate Svh and the function DefIndex.
    pub fn generate_plugin_registrar_symbol(&self, svh: &Svh, index: DefIndex)
                                            -> String {
        format!("__rustc_plugin_registrar__{}_{}", svh, index.as_usize())
    }

    pub fn sysroot<'a>(&'a self) -> &'a Path {
        match self.opts.maybe_sysroot {
            Some (ref sysroot) => sysroot,
            None => self.default_sysroot.as_ref()
                        .expect("missing sysroot and default_sysroot in Session")
        }
    }
    pub fn target_filesearch(&self, kind: PathKind) -> filesearch::FileSearch {
        filesearch::FileSearch::new(self.sysroot(),
                                    &self.opts.target_triple,
                                    &self.opts.search_paths,
                                    kind)
    }
    pub fn host_filesearch(&self, kind: PathKind) -> filesearch::FileSearch {
        filesearch::FileSearch::new(
            self.sysroot(),
            config::host_triple(),
            &self.opts.search_paths,
            kind)
    }
}

impl NodeIdAssigner for Session {
    fn next_node_id(&self) -> NodeId {
        self.reserve_node_ids(1)
    }

    fn peek_node_id(&self) -> NodeId {
        self.next_node_id.get().checked_add(1).unwrap()
    }

    fn diagnostic(&self) -> &errors::Handler {
        self.diagnostic()
    }
}

fn split_msg_into_multilines(msg: &str) -> Option<String> {
    // Conditions for enabling multi-line errors:
    if !msg.contains("mismatched types") &&
        !msg.contains("type mismatch resolving") &&
        !msg.contains("if and else have incompatible types") &&
        !msg.contains("if may be missing an else clause") &&
        !msg.contains("match arms have incompatible types") &&
        !msg.contains("structure constructor specifies a structure of type") &&
        !msg.contains("has an incompatible type for trait") {
            return None
    }
    let first = msg.match_indices("expected").filter(|s| {
        let last = msg[..s.0].chars().rev().next();
        last == Some(' ') || last == Some('(')
    }).map(|(a, b)| (a - 1, a + b.len()));
    let second = msg.match_indices("found").filter(|s| {
        msg[..s.0].chars().rev().next() == Some(' ')
    }).map(|(a, b)| (a - 1, a + b.len()));

    let mut new_msg = String::new();
    let mut head = 0;

    // Insert `\n` before expected and found.
    for (pos1, pos2) in first.zip(second) {
        new_msg = new_msg +
        // A `(` may be preceded by a space and it should be trimmed
                  msg[head..pos1.0].trim_right() + // prefix
                  "\n" +                           // insert before first
                  &msg[pos1.0..pos1.1] +           // insert what first matched
                  &msg[pos1.1..pos2.0] +           // between matches
                  "\n   " +                        // insert before second
        //           123
        // `expected` is 3 char longer than `found`. To align the types,
        // `found` gets 3 spaces prepended.
                  &msg[pos2.0..pos2.1];            // insert what second matched

        head = pos2.1;
    }

    let mut tail = &msg[head..];
    let third = tail.find("(values differ")
                   .or(tail.find("(lifetime"))
                   .or(tail.find("(cyclic type of infinite size"));
    // Insert `\n` before any remaining messages which match.
    if let Some(pos) = third {
        // The end of the message may just be wrapped in `()` without
        // `expected`/`found`.  Push this also to a new line and add the
        // final tail after.
        new_msg = new_msg +
        // `(` is usually preceded by a space and should be trimmed.
                  tail[..pos].trim_right() + // prefix
                  "\n" +                     // insert before paren
                  &tail[pos..];              // append the tail

        tail = "";
    }

    new_msg.push_str(tail);
    return Some(new_msg);
}

pub fn build_session(sopts: config::Options,
                     dep_graph: &DepGraph,
                     local_crate_source_file: Option<PathBuf>,
                     registry: errors::registry::Registry,
                     cstore: Rc<for<'a> CrateStore<'a>>)
                     -> Session {
    build_session_with_codemap(sopts,
                               dep_graph,
                               local_crate_source_file,
                               registry,
                               cstore,
                               Rc::new(codemap::CodeMap::new()))
}

pub fn build_session_with_codemap(sopts: config::Options,
                                  dep_graph: &DepGraph,
                                  local_crate_source_file: Option<PathBuf>,
                                  registry: errors::registry::Registry,
                                  cstore: Rc<for<'a> CrateStore<'a>>,
                                  codemap: Rc<codemap::CodeMap>)
                                  -> Session {
    // FIXME: This is not general enough to make the warning lint completely override
    // normal diagnostic warnings, since the warning lint can also be denied and changed
    // later via the source code.
    let can_print_warnings = sopts.lint_opts
        .iter()
        .filter(|&&(ref key, _)| *key == "warnings")
        .map(|&(_, ref level)| *level != lint::Allow)
        .last()
        .unwrap_or(true);
    let treat_err_as_bug = sopts.treat_err_as_bug;

    let emitter: Box<Emitter> = match sopts.error_format {
        config::ErrorOutputType::HumanReadable(color_config) => {
            Box::new(EmitterWriter::stderr(color_config,
                                           Some(registry),
                                           codemap.clone(),
                                           errors::snippet::FormatMode::EnvironmentSelected))
        }
        config::ErrorOutputType::Json => {
            Box::new(JsonEmitter::stderr(Some(registry), codemap.clone()))
        }
    };

    let diagnostic_handler =
        errors::Handler::with_emitter(can_print_warnings,
                                      treat_err_as_bug,
                                      emitter);

    build_session_(sopts,
                   dep_graph,
                   local_crate_source_file,
                   diagnostic_handler,
                   codemap,
                   cstore)
}

pub fn build_session_(sopts: config::Options,
                      dep_graph: &DepGraph,
                      local_crate_source_file: Option<PathBuf>,
                      span_diagnostic: errors::Handler,
                      codemap: Rc<codemap::CodeMap>,
                      cstore: Rc<for<'a> CrateStore<'a>>)
                      -> Session {
    let host = match Target::search(config::host_triple()) {
        Ok(t) => t,
        Err(e) => {
            panic!(span_diagnostic.fatal(&format!("Error loading host specification: {}", e)));
    }
    };
    let target_cfg = config::build_target_config(&sopts, &span_diagnostic);
    let p_s = parse::ParseSess::with_span_handler(span_diagnostic, codemap);
    let default_sysroot = match sopts.maybe_sysroot {
        Some(_) => None,
        None => Some(filesearch::get_or_default_sysroot())
    };

    // Make the path absolute, if necessary
    let local_crate_source_file = local_crate_source_file.map(|path|
        if path.is_absolute() {
            path.clone()
        } else {
            env::current_dir().unwrap().join(&path)
        }
    );

    let sess = Session {
        dep_graph: dep_graph.clone(),
        target: target_cfg,
        host: host,
        opts: sopts,
        cstore: cstore,
        parse_sess: p_s,
        // For a library crate, this is always none
        entry_fn: RefCell::new(None),
        entry_type: Cell::new(None),
        plugin_registrar_fn: Cell::new(None),
        default_sysroot: default_sysroot,
        local_crate_source_file: local_crate_source_file,
        working_dir: env::current_dir().unwrap(),
        lint_store: RefCell::new(lint::LintStore::new()),
        lints: RefCell::new(NodeMap()),
        plugin_llvm_passes: RefCell::new(Vec::new()),
        mir_passes: RefCell::new(mir_pass::Passes::new()),
        plugin_attributes: RefCell::new(Vec::new()),
        crate_types: RefCell::new(Vec::new()),
        dependency_formats: RefCell::new(FnvHashMap()),
        crate_disambiguator: Cell::new(token::intern("")),
        features: RefCell::new(feature_gate::Features::new()),
        recursion_limit: Cell::new(64),
        next_node_id: Cell::new(1),
        injected_allocator: Cell::new(None),
        injected_panic_runtime: Cell::new(None),
        available_macros: RefCell::new(HashSet::new()),
        imported_macro_spans: RefCell::new(HashMap::new()),
    };

    init_llvm(&sess);

    sess
}

fn init_llvm(sess: &Session) {
    unsafe {
        // Before we touch LLVM, make sure that multithreading is enabled.
        use std::sync::Once;
        static INIT: Once = Once::new();
        static mut POISONED: bool = false;
        INIT.call_once(|| {
            if llvm::LLVMStartMultithreaded() != 1 {
                // use an extra bool to make sure that all future usage of LLVM
                // cannot proceed despite the Once not running more than once.
                POISONED = true;
            }

            configure_llvm(sess);
        });

        if POISONED {
            bug!("couldn't enable multi-threaded LLVM");
        }
    }
}

unsafe fn configure_llvm(sess: &Session) {
    let mut llvm_c_strs = Vec::new();
    let mut llvm_args = Vec::new();

    {
        let mut add = |arg: &str| {
            let s = CString::new(arg).unwrap();
            llvm_args.push(s.as_ptr());
            llvm_c_strs.push(s);
        };
        add("rustc"); // fake program name
        if sess.time_llvm_passes() { add("-time-passes"); }
        if sess.print_llvm_passes() { add("-debug-pass=Structure"); }

        for arg in &sess.opts.cg.llvm_args {
            add(&(*arg));
        }
    }

    llvm::LLVMInitializePasses();

    llvm::initialize_available_targets();

    llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int,
                                 llvm_args.as_ptr());
}

pub fn early_error(output: config::ErrorOutputType, msg: &str) -> ! {
    let mut emitter: Box<Emitter> = match output {
        config::ErrorOutputType::HumanReadable(color_config) => {
            Box::new(BasicEmitter::stderr(color_config))
        }
        config::ErrorOutputType::Json => Box::new(JsonEmitter::basic()),
    };
    emitter.emit(&MultiSpan::new(), msg, None, errors::Level::Fatal);
    panic!(errors::FatalError);
}

pub fn early_warn(output: config::ErrorOutputType, msg: &str) {
    let mut emitter: Box<Emitter> = match output {
        config::ErrorOutputType::HumanReadable(color_config) => {
            Box::new(BasicEmitter::stderr(color_config))
        }
        config::ErrorOutputType::Json => Box::new(JsonEmitter::basic()),
    };
    emitter.emit(&MultiSpan::new(), msg, None, errors::Level::Warning);
}

// Err(0) means compilation was stopped, but no errors were found.
// This would be better as a dedicated enum, but using try! is so convenient.
pub type CompileResult = Result<(), usize>;

pub fn compile_result_from_err_count(err_count: usize) -> CompileResult {
    if err_count == 0 {
        Ok(())
    } else {
        Err(err_count)
    }
}

#[cold]
#[inline(never)]
pub fn bug_fmt(file: &'static str, line: u32, args: fmt::Arguments) -> ! {
    // this wrapper mostly exists so I don't have to write a fully
    // qualified path of None::<Span> inside the bug!() macro defintion
    opt_span_bug_fmt(file, line, None::<Span>, args);
}

#[cold]
#[inline(never)]
pub fn span_bug_fmt<S: Into<MultiSpan>>(file: &'static str,
                                        line: u32,
                                        span: S,
                                        args: fmt::Arguments) -> ! {
    opt_span_bug_fmt(file, line, Some(span), args);
}

fn opt_span_bug_fmt<S: Into<MultiSpan>>(file: &'static str,
                                        line: u32,
                                        span: Option<S>,
                                        args: fmt::Arguments) -> ! {
    tls::with_opt(move |tcx| {
        let msg = format!("{}:{}: {}", file, line, args);
        match (tcx, span) {
            (Some(tcx), Some(span)) => tcx.sess.diagnostic().span_bug(span, &msg),
            (Some(tcx), None) => tcx.sess.diagnostic().bug(&msg),
            (None, _) => panic!(msg)
        }
    });
    unreachable!();
}
