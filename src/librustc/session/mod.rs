// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lint;
use metadata::cstore::CStore;
use metadata::filesearch;
use middle::dependency_format;
use session::search_paths::PathKind;
use util::nodemap::{NodeMap, FnvHashMap};

use syntax::ast::{NodeId, NodeIdAssigner};
use syntax::codemap::Span;
use syntax::diagnostic::{self, Emitter};
use syntax::diagnostics;
use syntax::feature_gate;
use syntax::parse;
use syntax::parse::token;
use syntax::parse::ParseSess;
use syntax::{ast, codemap};
use syntax::feature_gate::AttributeType;

use rustc_back::target::Target;

use std::path::{Path, PathBuf};
use std::cell::{Cell, RefCell};
use std::env;

pub mod config;
pub mod search_paths;

// Represents the data associated with a compilation
// session for a single crate.
pub struct Session {
    pub target: config::Config,
    pub host: Target,
    pub opts: config::Options,
    pub cstore: CStore,
    pub parse_sess: ParseSess,
    // For a library crate, this is always none
    pub entry_fn: RefCell<Option<(NodeId, codemap::Span)>>,
    pub entry_type: Cell<Option<config::EntryFnType>>,
    pub plugin_registrar_fn: Cell<Option<ast::NodeId>>,
    pub default_sysroot: Option<PathBuf>,
    // The name of the root source file of the crate, in the local file system.
    // The path is always expected to be absolute. `None` means that there is no
    // source file.
    pub local_crate_source_file: Option<PathBuf>,
    pub working_dir: PathBuf,
    pub lint_store: RefCell<lint::LintStore>,
    pub lints: RefCell<NodeMap<Vec<(lint::LintId, codemap::Span, String)>>>,
    pub plugin_llvm_passes: RefCell<Vec<String>>,
    pub plugin_attributes: RefCell<Vec<(String, AttributeType)>>,
    pub crate_types: RefCell<Vec<config::CrateType>>,
    pub dependency_formats: RefCell<dependency_format::Dependencies>,
    pub crate_metadata: RefCell<Vec<String>>,
    pub features: RefCell<feature_gate::Features>,

    pub delayed_span_bug: RefCell<Option<(codemap::Span, String)>>,

    /// The maximum recursion limit for potentially infinitely recursive
    /// operations such as auto-dereference and monomorphization.
    pub recursion_limit: Cell<usize>,

    pub can_print_warnings: bool,

    /// The metadata::creader module may inject an allocator dependency if it
    /// didn't already find one, and this tracks what was injected.
    pub injected_allocator: Cell<Option<ast::CrateNum>>,

    next_node_id: Cell<ast::NodeId>,
}

impl Session {
    pub fn span_fatal(&self, sp: Span, msg: &str) -> ! {
        if self.opts.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        panic!(self.diagnostic().span_fatal(sp, msg))
    }
    pub fn span_fatal_with_code(&self, sp: Span, msg: &str, code: &str) -> ! {
        if self.opts.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        panic!(self.diagnostic().span_fatal_with_code(sp, msg, code))
    }
    pub fn fatal(&self, msg: &str) -> ! {
        if self.opts.treat_err_as_bug {
            self.bug(msg);
        }
        self.diagnostic().handler().fatal(msg)
    }
    pub fn span_err_or_warn(&self, is_warning: bool, sp: Span, msg: &str) {
        if is_warning {
            self.span_warn(sp, msg);
        } else {
            self.span_err(sp, msg);
        }
    }
    pub fn span_err(&self, sp: Span, msg: &str) {
        if self.opts.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        match split_msg_into_multilines(msg) {
            Some(msg) => self.diagnostic().span_err(sp, &msg[..]),
            None => self.diagnostic().span_err(sp, msg)
        }
    }
    pub fn note_rfc_1214(&self, span: Span) {
        self.span_note(
            span,
            &format!("this warning results from recent bug fixes and clarifications; \
                      it will become a HARD ERROR in the next release. \
                      See RFC 1214 for details."));
    }
    pub fn span_err_with_code(&self, sp: Span, msg: &str, code: &str) {
        if self.opts.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        match split_msg_into_multilines(msg) {
            Some(msg) => self.diagnostic().span_err_with_code(sp, &msg[..], code),
            None => self.diagnostic().span_err_with_code(sp, msg, code)
        }
    }
    pub fn err(&self, msg: &str) {
        if self.opts.treat_err_as_bug {
            self.bug(msg);
        }
        self.diagnostic().handler().err(msg)
    }
    pub fn err_count(&self) -> usize {
        self.diagnostic().handler().err_count()
    }
    pub fn has_errors(&self) -> bool {
        self.diagnostic().handler().has_errors()
    }
    pub fn abort_if_errors(&self) {
        self.diagnostic().handler().abort_if_errors();

        let delayed_bug = self.delayed_span_bug.borrow();
        match *delayed_bug {
            Some((span, ref errmsg)) => {
                self.diagnostic().span_bug(span, errmsg);
            },
            _ => {}
        }
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        if self.can_print_warnings {
            self.diagnostic().span_warn(sp, msg)
        }
    }
    pub fn span_warn_with_code(&self, sp: Span, msg: &str, code: &str) {
        if self.can_print_warnings {
            self.diagnostic().span_warn_with_code(sp, msg, code)
        }
    }
    pub fn warn(&self, msg: &str) {
        if self.can_print_warnings {
            self.diagnostic().handler().warn(msg)
        }
    }
    pub fn opt_span_warn(&self, opt_sp: Option<Span>, msg: &str) {
        match opt_sp {
            Some(sp) => self.span_warn(sp, msg),
            None => self.warn(msg),
        }
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.diagnostic().span_note(sp, msg)
    }
    pub fn span_end_note(&self, sp: Span, msg: &str) {
        self.diagnostic().span_end_note(sp, msg)
    }

    /// Prints out a message with a suggested edit of the code.
    ///
    /// See `diagnostic::RenderSpan::Suggestion` for more information.
    pub fn span_suggestion(&self, sp: Span, msg: &str, suggestion: String) {
        self.diagnostic().span_suggestion(sp, msg, suggestion)
    }
    pub fn span_help(&self, sp: Span, msg: &str) {
        self.diagnostic().span_help(sp, msg)
    }
    pub fn fileline_note(&self, sp: Span, msg: &str) {
        self.diagnostic().fileline_note(sp, msg)
    }
    pub fn fileline_help(&self, sp: Span, msg: &str) {
        self.diagnostic().fileline_help(sp, msg)
    }
    pub fn note(&self, msg: &str) {
        self.diagnostic().handler().note(msg)
    }
    pub fn help(&self, msg: &str) {
        self.diagnostic().handler().help(msg)
    }
    pub fn opt_span_bug(&self, opt_sp: Option<Span>, msg: &str) -> ! {
        match opt_sp {
            Some(sp) => self.span_bug(sp, msg),
            None => self.bug(msg),
        }
    }
    /// Delay a span_bug() call until abort_if_errors()
    pub fn delay_span_bug(&self, sp: Span, msg: &str) {
        let mut delayed = self.delayed_span_bug.borrow_mut();
        *delayed = Some((sp, msg.to_string()));
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
        match lints.get_mut(&id) {
            Some(arr) => { arr.push((lint_id, sp, msg)); return; }
            None => {}
        }
        lints.insert(id, vec!((lint_id, sp, msg)));
    }
    pub fn reserve_node_ids(&self, count: ast::NodeId) -> ast::NodeId {
        let id = self.next_node_id.get();

        match id.checked_add(count) {
            Some(next) => self.next_node_id.set(next),
            None => self.bug("Input too large, ran out of node ids!")
        }

        id
    }
    pub fn diagnostic<'a>(&'a self) -> &'a diagnostic::SpanHandler {
        &self.parse_sess.span_diagnostic
    }
    pub fn codemap<'a>(&'a self) -> &'a codemap::CodeMap {
        self.parse_sess.codemap()
    }
    // This exists to help with refactoring to eliminate impossible
    // cases later on
    pub fn impossible_case(&self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp,
                      &format!("impossible case reached: {}", msg));
    }
    pub fn verbose(&self) -> bool { self.opts.debugging_opts.verbose }
    pub fn time_passes(&self) -> bool { self.opts.debugging_opts.time_passes }
    pub fn count_llvm_insns(&self) -> bool {
        self.opts.debugging_opts.count_llvm_insns
    }
    pub fn count_type_sizes(&self) -> bool {
        self.opts.debugging_opts.count_type_sizes
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
        self.opts.debugging_opts.no_landing_pads
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
        s.0 > 0 && (msg.char_at_reverse(s.0) == ' ' ||
                    msg.char_at_reverse(s.0) == '(')
    }).map(|(a, b)| (a - 1, a + b.len()));
    let second = msg.match_indices("found").filter(|s| {
        msg.char_at_reverse(s.0) == ' '
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
                     local_crate_source_file: Option<PathBuf>,
                     registry: diagnostics::registry::Registry)
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

    let codemap = codemap::CodeMap::new();
    let diagnostic_handler =
        diagnostic::Handler::new(sopts.emit_cfg, Some(registry), can_print_warnings);
    let span_diagnostic_handler =
        diagnostic::SpanHandler::new(diagnostic_handler, codemap);

    build_session_(sopts, local_crate_source_file, span_diagnostic_handler)
}

pub fn build_session_(sopts: config::Options,
                      local_crate_source_file: Option<PathBuf>,
                      span_diagnostic: diagnostic::SpanHandler)
                      -> Session {
    let host = match Target::search(config::host_triple()) {
        Ok(t) => t,
        Err(e) => {
            span_diagnostic.handler()
                .fatal(&format!("Error loading host specification: {}", e));
    }
    };
    let target_cfg = config::build_target_config(&sopts, &span_diagnostic);
    let p_s = parse::ParseSess::with_span_handler(span_diagnostic);
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

    let can_print_warnings = sopts.lint_opts
        .iter()
        .filter(|&&(ref key, _)| *key == "warnings")
        .map(|&(_, ref level)| *level != lint::Allow)
        .last()
        .unwrap_or(true);

    let sess = Session {
        target: target_cfg,
        host: host,
        opts: sopts,
        cstore: CStore::new(token::get_ident_interner()),
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
        plugin_attributes: RefCell::new(Vec::new()),
        crate_types: RefCell::new(Vec::new()),
        dependency_formats: RefCell::new(FnvHashMap()),
        crate_metadata: RefCell::new(Vec::new()),
        delayed_span_bug: RefCell::new(None),
        features: RefCell::new(feature_gate::Features::new()),
        recursion_limit: Cell::new(64),
        can_print_warnings: can_print_warnings,
        next_node_id: Cell::new(1),
        injected_allocator: Cell::new(None),
    };

    sess
}

// Seems out of place, but it uses session, so I'm putting it here
pub fn expect<T, M>(sess: &Session, opt: Option<T>, msg: M) -> T where
    M: FnOnce() -> String,
{
    diagnostic::expect(sess.diagnostic(), opt, msg)
}

pub fn early_error(cfg: diagnostic::EmitterConfig, msg: &str) -> ! {
    let mut emitter = diagnostic::EmitterWriter::stderr(cfg, None);
    emitter.emit(None, msg, None, diagnostic::Fatal);
    panic!(diagnostic::FatalError);
}

pub fn early_warn(cfg: diagnostic::EmitterConfig, msg: &str) {
    let mut emitter = diagnostic::EmitterWriter::stderr(cfg, None);
    emitter.emit(None, msg, None, diagnostic::Warning);
}
