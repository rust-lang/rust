// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(range_contains)]
#![cfg_attr(unix, feature(libc))]
#![feature(conservative_impl_trait)]
#![feature(i128_type)]

extern crate term;
#[cfg(unix)]
extern crate libc;
extern crate rustc_data_structures;
extern crate serialize as rustc_serialize;
extern crate syntax_pos;
extern crate unicode_width;

pub use emitter::ColorConfig;

use self::Level::*;

use emitter::{Emitter, EmitterWriter};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::stable_hasher::StableHasher;

use std::borrow::Cow;
use std::cell::{RefCell, Cell};
use std::mem;
use std::rc::Rc;
use std::{error, fmt};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;

mod diagnostic;
mod diagnostic_builder;
pub mod emitter;
mod snippet;
pub mod registry;
mod styled_buffer;
mod lock;

use syntax_pos::{BytePos, Loc, FileLinesResult, FileMap, FileName, MultiSpan, Span, NO_EXPANSION};

#[derive(Clone, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub struct CodeSuggestion {
    /// Each substitute can have multiple variants due to multiple
    /// applicable suggestions
    ///
    /// `foo.bar` might be replaced with `a.b` or `x.y` by replacing
    /// `foo` and `bar` on their own:
    ///
    /// ```
    /// vec![
    ///     Substitution { parts: vec![(0..3, "a"), (4..7, "b")] },
    ///     Substitution { parts: vec![(0..3, "x"), (4..7, "y")] },
    /// ]
    /// ```
    ///
    /// or by replacing the entire span:
    ///
    /// ```
    /// vec![
    ///     Substitution { parts: vec![(0..7, "a.b")] },
    ///     Substitution { parts: vec![(0..7, "x.y")] },
    /// ]
    /// ```
    pub substitutions: Vec<Substitution>,
    pub msg: String,
    pub show_code_when_inline: bool,
}

#[derive(Clone, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
/// See the docs on `CodeSuggestion::substitutions`
pub struct Substitution {
    pub parts: Vec<SubstitutionPart>,
}

#[derive(Clone, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub struct SubstitutionPart {
    pub span: Span,
    pub snippet: String,
}

pub trait CodeMapper {
    fn lookup_char_pos(&self, pos: BytePos) -> Loc;
    fn span_to_lines(&self, sp: Span) -> FileLinesResult;
    fn span_to_string(&self, sp: Span) -> String;
    fn span_to_filename(&self, sp: Span) -> FileName;
    fn merge_spans(&self, sp_lhs: Span, sp_rhs: Span) -> Option<Span>;
    fn call_span_if_macro(&self, sp: Span) -> Span;
    fn ensure_filemap_source_present(&self, file_map: Rc<FileMap>) -> bool;
    fn doctest_offset_line(&self, line: usize) -> usize;
}

impl CodeSuggestion {
    /// Returns the assembled code suggestions and whether they should be shown with an underline.
    pub fn splice_lines(&self, cm: &CodeMapper) -> Vec<(String, Vec<SubstitutionPart>)> {
        use syntax_pos::{CharPos, Loc, Pos};

        fn push_trailing(buf: &mut String,
                         line_opt: Option<&Cow<str>>,
                         lo: &Loc,
                         hi_opt: Option<&Loc>) {
            let (lo, hi_opt) = (lo.col.to_usize(), hi_opt.map(|hi| hi.col.to_usize()));
            if let Some(line) = line_opt {
                if let Some(lo) = line.char_indices().map(|(i, _)| i).nth(lo) {
                    let hi_opt = hi_opt.and_then(|hi| line.char_indices().map(|(i, _)| i).nth(hi));
                    buf.push_str(match hi_opt {
                        Some(hi) => &line[lo..hi],
                        None => &line[lo..],
                    });
                }
                if let None = hi_opt {
                    buf.push('\n');
                }
            }
        }

        assert!(!self.substitutions.is_empty());

        self.substitutions.iter().cloned().map(|mut substitution| {
            // Assumption: all spans are in the same file, and all spans
            // are disjoint. Sort in ascending order.
            substitution.parts.sort_by_key(|part| part.span.lo());

            // Find the bounding span.
            let lo = substitution.parts.iter().map(|part| part.span.lo()).min().unwrap();
            let hi = substitution.parts.iter().map(|part| part.span.hi()).min().unwrap();
            let bounding_span = Span::new(lo, hi, NO_EXPANSION);
            let lines = cm.span_to_lines(bounding_span).unwrap();
            assert!(!lines.lines.is_empty());

            // To build up the result, we do this for each span:
            // - push the line segment trailing the previous span
            //   (at the beginning a "phantom" span pointing at the start of the line)
            // - push lines between the previous and current span (if any)
            // - if the previous and current span are not on the same line
            //   push the line segment leading up to the current span
            // - splice in the span substitution
            //
            // Finally push the trailing line segment of the last span
            let fm = &lines.file;
            let mut prev_hi = cm.lookup_char_pos(bounding_span.lo());
            prev_hi.col = CharPos::from_usize(0);

            let mut prev_line = fm.get_line(lines.lines[0].line_index);
            let mut buf = String::new();

            for part in &substitution.parts {
                let cur_lo = cm.lookup_char_pos(part.span.lo());
                if prev_hi.line == cur_lo.line {
                    push_trailing(&mut buf, prev_line.as_ref(), &prev_hi, Some(&cur_lo));
                } else {
                    push_trailing(&mut buf, prev_line.as_ref(), &prev_hi, None);
                    // push lines between the previous and current span (if any)
                    for idx in prev_hi.line..(cur_lo.line - 1) {
                        if let Some(line) = fm.get_line(idx) {
                            buf.push_str(line.as_ref());
                            buf.push('\n');
                        }
                    }
                    if let Some(cur_line) = fm.get_line(cur_lo.line - 1) {
                        buf.push_str(&cur_line[..cur_lo.col.to_usize()]);
                    }
                }
                buf.push_str(&part.snippet);
                prev_hi = cm.lookup_char_pos(part.span.hi());
                prev_line = fm.get_line(prev_hi.line - 1);
            }
            // if the replacement already ends with a newline, don't print the next line
            if !buf.ends_with('\n') {
                push_trailing(&mut buf, prev_line.as_ref(), &prev_hi, None);
            }
            // remove trailing newlines
            while buf.ends_with('\n') {
                buf.pop();
            }
            (buf, substitution.parts)
        }).collect()
    }
}

/// Used as a return value to signify a fatal error occurred. (It is also
/// used as the argument to panic at the moment, but that will eventually
/// not be true.)
#[derive(Copy, Clone, Debug)]
#[must_use]
pub struct FatalError;

impl fmt::Display for FatalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "parser fatal error")
    }
}

impl error::Error for FatalError {
    fn description(&self) -> &str {
        "The parser has encountered a fatal error"
    }
}

/// Signifies that the compiler died with an explicit call to `.bug`
/// or `.span_bug` rather than a failed assertion, etc.
#[derive(Copy, Clone, Debug)]
pub struct ExplicitBug;

impl fmt::Display for ExplicitBug {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "parser internal bug")
    }
}

impl error::Error for ExplicitBug {
    fn description(&self) -> &str {
        "The parser has encountered an internal bug"
    }
}

pub use diagnostic::{Diagnostic, SubDiagnostic, DiagnosticStyledString, DiagnosticId};
pub use diagnostic_builder::DiagnosticBuilder;

/// A handler deals with errors; certain errors
/// (fatal, bug, unimpl) may cause immediate exit,
/// others log errors for later reporting.
pub struct Handler {
    pub flags: HandlerFlags,

    err_count: AtomicUsize,
    emitter: RefCell<Box<Emitter>>,
    continue_after_error: Cell<bool>,
    delayed_span_bug: RefCell<Option<Diagnostic>>,
    tracked_diagnostics: RefCell<Option<Vec<Diagnostic>>>,

    // This set contains a hash of every diagnostic that has been emitted by
    // this handler. These hashes is used to avoid emitting the same error
    // twice.
    emitted_diagnostics: RefCell<FxHashSet<u128>>,
}

#[derive(Default)]
pub struct HandlerFlags {
    pub can_emit_warnings: bool,
    pub treat_err_as_bug: bool,
    pub external_macro_backtrace: bool,
}

impl Handler {
    pub fn with_tty_emitter(color_config: ColorConfig,
                            can_emit_warnings: bool,
                            treat_err_as_bug: bool,
                            cm: Option<Rc<CodeMapper>>)
                            -> Handler {
        Handler::with_tty_emitter_and_flags(
            color_config,
            cm,
            HandlerFlags {
                can_emit_warnings,
                treat_err_as_bug,
                .. Default::default()
            })
    }

    pub fn with_tty_emitter_and_flags(color_config: ColorConfig,
                                      cm: Option<Rc<CodeMapper>>,
                                      flags: HandlerFlags)
                                      -> Handler {
        let emitter = Box::new(EmitterWriter::stderr(color_config, cm, false));
        Handler::with_emitter_and_flags(emitter, flags)
    }

    pub fn with_emitter(can_emit_warnings: bool,
                        treat_err_as_bug: bool,
                        e: Box<Emitter>)
                        -> Handler {
        Handler::with_emitter_and_flags(
            e,
            HandlerFlags {
                can_emit_warnings,
                treat_err_as_bug,
                .. Default::default()
            })
    }

    pub fn with_emitter_and_flags(e: Box<Emitter>, flags: HandlerFlags) -> Handler {
        Handler {
            flags,
            err_count: AtomicUsize::new(0),
            emitter: RefCell::new(e),
            continue_after_error: Cell::new(true),
            delayed_span_bug: RefCell::new(None),
            tracked_diagnostics: RefCell::new(None),
            emitted_diagnostics: RefCell::new(FxHashSet()),
        }
    }

    pub fn set_continue_after_error(&self, continue_after_error: bool) {
        self.continue_after_error.set(continue_after_error);
    }

    /// Resets the diagnostic error count as well as the cached emitted diagnostics.
    ///
    /// NOTE: DO NOT call this function from rustc. It is only meant to be called from external
    /// tools that want to reuse a `Parser` cleaning the previously emitted diagnostics as well as
    /// the overall count of emitted error diagnostics.
    pub fn reset_err_count(&self) {
        self.emitted_diagnostics.replace(FxHashSet());
        self.err_count.store(0, SeqCst);
    }

    pub fn struct_dummy<'a>(&'a self) -> DiagnosticBuilder<'a> {
        DiagnosticBuilder::new(self, Level::Cancelled, "")
    }

    pub fn struct_span_warn<'a, S: Into<MultiSpan>>(&'a self,
                                                    sp: S,
                                                    msg: &str)
                                                    -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Warning, msg);
        result.set_span(sp);
        if !self.flags.can_emit_warnings {
            result.cancel();
        }
        result
    }
    pub fn struct_span_warn_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                              sp: S,
                                                              msg: &str,
                                                              code: DiagnosticId)
                                                              -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Warning, msg);
        result.set_span(sp);
        result.code(code);
        if !self.flags.can_emit_warnings {
            result.cancel();
        }
        result
    }
    pub fn struct_warn<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Warning, msg);
        if !self.flags.can_emit_warnings {
            result.cancel();
        }
        result
    }
    pub fn struct_span_err<'a, S: Into<MultiSpan>>(&'a self,
                                                   sp: S,
                                                   msg: &str)
                                                   -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Error, msg);
        result.set_span(sp);
        result
    }
    pub fn struct_span_err_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                             sp: S,
                                                             msg: &str,
                                                             code: DiagnosticId)
                                                             -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Error, msg);
        result.set_span(sp);
        result.code(code);
        result
    }
    // FIXME: This method should be removed (every error should have an associated error code).
    pub fn struct_err<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        DiagnosticBuilder::new(self, Level::Error, msg)
    }
    pub fn struct_err_with_code<'a>(
        &'a self,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Error, msg);
        result.code(code);
        result
    }
    pub fn struct_span_fatal<'a, S: Into<MultiSpan>>(&'a self,
                                                     sp: S,
                                                     msg: &str)
                                                     -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Fatal, msg);
        result.set_span(sp);
        result
    }
    pub fn struct_span_fatal_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                               sp: S,
                                                               msg: &str,
                                                               code: DiagnosticId)
                                                               -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Fatal, msg);
        result.set_span(sp);
        result.code(code);
        result
    }
    pub fn struct_fatal<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        DiagnosticBuilder::new(self, Level::Fatal, msg)
    }

    pub fn cancel(&self, err: &mut DiagnosticBuilder) {
        err.cancel();
    }

    fn panic_if_treat_err_as_bug(&self) {
        if self.flags.treat_err_as_bug {
            panic!("encountered error with `-Z treat_err_as_bug");
        }
    }

    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> FatalError {
        self.emit(&sp.into(), msg, Fatal);
        FatalError
    }
    pub fn span_fatal_with_code<S: Into<MultiSpan>>(&self,
                                                    sp: S,
                                                    msg: &str,
                                                    code: DiagnosticId)
                                                    -> FatalError {
        self.emit_with_code(&sp.into(), msg, code, Fatal);
        FatalError
    }
    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(&sp.into(), msg, Error);
    }
    pub fn mut_span_err<'a, S: Into<MultiSpan>>(&'a self,
                                                sp: S,
                                                msg: &str)
                                                -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Error, msg);
        result.set_span(sp);
        result
    }
    pub fn span_err_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: DiagnosticId) {
        self.emit_with_code(&sp.into(), msg, code, Error);
    }
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(&sp.into(), msg, Warning);
    }
    pub fn span_warn_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: DiagnosticId) {
        self.emit_with_code(&sp.into(), msg, code, Warning);
    }
    pub fn span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.emit(&sp.into(), msg, Bug);
        panic!(ExplicitBug);
    }
    pub fn delay_span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        if self.flags.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        let mut diagnostic = Diagnostic::new(Level::Bug, msg);
        diagnostic.set_span(sp.into());
        *self.delayed_span_bug.borrow_mut() = Some(diagnostic);
    }
    pub fn span_bug_no_panic<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(&sp.into(), msg, Bug);
    }
    pub fn span_note_without_error<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(&sp.into(), msg, Note);
    }
    pub fn span_note_diag<'a>(&'a self,
                              sp: Span,
                              msg: &str)
                              -> DiagnosticBuilder<'a> {
        let mut db = DiagnosticBuilder::new(self, Note, msg);
        db.set_span(sp);
        db
    }
    pub fn span_unimpl<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.span_bug(sp, &format!("unimplemented {}", msg));
    }
    pub fn fatal(&self, msg: &str) -> FatalError {
        if self.flags.treat_err_as_bug {
            self.bug(msg);
        }
        let mut db = DiagnosticBuilder::new(self, Fatal, msg);
        db.emit();
        FatalError
    }
    pub fn err(&self, msg: &str) {
        if self.flags.treat_err_as_bug {
            self.bug(msg);
        }
        let mut db = DiagnosticBuilder::new(self, Error, msg);
        db.emit();
    }
    pub fn warn(&self, msg: &str) {
        let mut db = DiagnosticBuilder::new(self, Warning, msg);
        db.emit();
    }
    pub fn note_without_error(&self, msg: &str) {
        let mut db = DiagnosticBuilder::new(self, Note, msg);
        db.emit();
    }
    pub fn bug(&self, msg: &str) -> ! {
        let mut db = DiagnosticBuilder::new(self, Bug, msg);
        db.emit();
        panic!(ExplicitBug);
    }
    pub fn unimpl(&self, msg: &str) -> ! {
        self.bug(&format!("unimplemented {}", msg));
    }

    fn bump_err_count(&self) {
        self.panic_if_treat_err_as_bug();
        self.err_count.fetch_add(1, SeqCst);
    }

    pub fn err_count(&self) -> usize {
        self.err_count.load(SeqCst)
    }

    pub fn has_errors(&self) -> bool {
        self.err_count() > 0
    }
    pub fn abort_if_errors(&self) {
        let s;
        match self.err_count() {
            0 => {
                if let Some(bug) = self.delayed_span_bug.borrow_mut().take() {
                    DiagnosticBuilder::new_diagnostic(self, bug).emit();
                }
                return;
            }
            1 => s = "aborting due to previous error".to_string(),
            _ => {
                s = format!("aborting due to {} previous errors", self.err_count());
            }
        }

        panic!(self.fatal(&s));
    }
    pub fn emit(&self, msp: &MultiSpan, msg: &str, lvl: Level) {
        if lvl == Warning && !self.flags.can_emit_warnings {
            return;
        }
        let mut db = DiagnosticBuilder::new(self, lvl, msg);
        db.set_span(msp.clone());
        db.emit();
        if !self.continue_after_error.get() {
            self.abort_if_errors();
        }
    }
    pub fn emit_with_code(&self, msp: &MultiSpan, msg: &str, code: DiagnosticId, lvl: Level) {
        if lvl == Warning && !self.flags.can_emit_warnings {
            return;
        }
        let mut db = DiagnosticBuilder::new_with_code(self, lvl, Some(code), msg);
        db.set_span(msp.clone());
        db.emit();
        if !self.continue_after_error.get() {
            self.abort_if_errors();
        }
    }

    pub fn track_diagnostics<F, R>(&self, f: F) -> (R, Vec<Diagnostic>)
        where F: FnOnce() -> R
    {
        let prev = mem::replace(&mut *self.tracked_diagnostics.borrow_mut(),
                                Some(Vec::new()));
        let ret = f();
        let diagnostics = mem::replace(&mut *self.tracked_diagnostics.borrow_mut(), prev)
            .unwrap();
        (ret, diagnostics)
    }

    fn emit_db(&self, db: &DiagnosticBuilder) {
        let diagnostic = &**db;

        if let Some(ref mut list) = *self.tracked_diagnostics.borrow_mut() {
            list.push(diagnostic.clone());
        }

        let diagnostic_hash = {
            use std::hash::Hash;
            let mut hasher = StableHasher::new();
            diagnostic.hash(&mut hasher);
            hasher.finish()
        };

        // Only emit the diagnostic if we haven't already emitted an equivalent
        // one:
        if self.emitted_diagnostics.borrow_mut().insert(diagnostic_hash) {
            self.emitter.borrow_mut().emit(db);
            if db.is_error() {
                self.bump_err_count();
            }
        }
    }
}


#[derive(Copy, PartialEq, Clone, Hash, Debug, RustcEncodable, RustcDecodable)]
pub enum Level {
    Bug,
    Fatal,
    // An error which while not immediately fatal, should stop the compiler
    // progressing beyond the current phase.
    PhaseFatal,
    Error,
    Warning,
    Note,
    Help,
    Cancelled,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_str().fmt(f)
    }
}

impl Level {
    fn color(self) -> term::color::Color {
        match self {
            Bug | Fatal | PhaseFatal | Error => term::color::BRIGHT_RED,
            Warning => {
                if cfg!(windows) {
                    term::color::BRIGHT_YELLOW
                } else {
                    term::color::YELLOW
                }
            }
            Note => term::color::BRIGHT_GREEN,
            Help => term::color::BRIGHT_CYAN,
            Cancelled => unreachable!(),
        }
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Bug => "error: internal compiler error",
            Fatal | PhaseFatal | Error => "error",
            Warning => "warning",
            Note => "note",
            Help => "help",
            Cancelled => panic!("Shouldn't call on cancelled error"),
        }
    }
}
