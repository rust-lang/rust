// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustc_errors"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(question_mark)]
#![feature(range_contains)]
#![feature(libc)]
#![feature(unicode)]

extern crate serialize;
extern crate term;
#[macro_use] extern crate log;
#[macro_use] extern crate libc;
extern crate rustc_unicode;
extern crate serialize as rustc_serialize; // used by deriving
extern crate syntax_pos;

pub use emitter::ColorConfig;

use self::Level::*;
use self::RenderSpan::*;

use emitter::{Emitter, EmitterWriter};

use std::cell::{RefCell, Cell};
use std::{error, fmt};
use std::rc::Rc;
use std::thread::panicking;

pub mod emitter;
pub mod snippet;
pub mod registry;

use syntax_pos::{BytePos, Loc, FileLinesResult, FileName, MultiSpan, Span, NO_EXPANSION };
use syntax_pos::{MacroBacktrace};

#[derive(Clone)]
pub enum RenderSpan {
    /// A FullSpan renders with both with an initial line for the
    /// message, prefixed by file:linenum, followed by a summary of
    /// the source code covered by the span.
    FullSpan(MultiSpan),

    /// A suggestion renders with both with an initial line for the
    /// message, prefixed by file:linenum, followed by a summary
    /// of hypothetical source code, where each `String` is spliced
    /// into the lines in place of the code covered by each span.
    Suggestion(CodeSuggestion),
}

#[derive(Clone)]
pub struct CodeSuggestion {
    pub msp: MultiSpan,
    pub substitutes: Vec<String>,
}

pub trait CodeMapper {
    fn lookup_char_pos(&self, pos: BytePos) -> Loc;
    fn span_to_lines(&self, sp: Span) -> FileLinesResult;
    fn span_to_string(&self, sp: Span) -> String;
    fn span_to_filename(&self, sp: Span) -> FileName;
    fn macro_backtrace(&self, span: Span) -> Vec<MacroBacktrace>;
}

impl RenderSpan {
    fn span(&self) -> &MultiSpan {
        match *self {
            FullSpan(ref msp) |
            Suggestion(CodeSuggestion { ref msp, .. }) =>
                msp
        }
    }
}

impl CodeSuggestion {
    /// Returns the assembled code suggestion.
    pub fn splice_lines(&self, cm: &CodeMapper) -> String {
        use syntax_pos::{CharPos, Loc, Pos};

        fn push_trailing(buf: &mut String, line_opt: Option<&str>,
                         lo: &Loc, hi_opt: Option<&Loc>) {
            let (lo, hi_opt) = (lo.col.to_usize(), hi_opt.map(|hi|hi.col.to_usize()));
            if let Some(line) = line_opt {
                if line.len() > lo {
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

        let mut primary_spans = self.msp.primary_spans().to_owned();

        assert_eq!(primary_spans.len(), self.substitutes.len());
        if primary_spans.is_empty() {
            return format!("");
        }

        // Assumption: all spans are in the same file, and all spans
        // are disjoint. Sort in ascending order.
        primary_spans.sort_by_key(|sp| sp.lo);

        // Find the bounding span.
        let lo = primary_spans.iter().map(|sp| sp.lo).min().unwrap();
        let hi = primary_spans.iter().map(|sp| sp.hi).min().unwrap();
        let bounding_span = Span { lo: lo, hi: hi, expn_id: NO_EXPANSION };
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
        let mut prev_hi = cm.lookup_char_pos(bounding_span.lo);
        prev_hi.col = CharPos::from_usize(0);

        let mut prev_line = fm.get_line(lines.lines[0].line_index);
        let mut buf = String::new();

        for (sp, substitute) in primary_spans.iter().zip(self.substitutes.iter()) {
            let cur_lo = cm.lookup_char_pos(sp.lo);
            if prev_hi.line == cur_lo.line {
                push_trailing(&mut buf, prev_line, &prev_hi, Some(&cur_lo));
            } else {
                push_trailing(&mut buf, prev_line, &prev_hi, None);
                // push lines between the previous and current span (if any)
                for idx in prev_hi.line..(cur_lo.line - 1) {
                    if let Some(line) = fm.get_line(idx) {
                        buf.push_str(line);
                        buf.push('\n');
                    }
                }
                if let Some(cur_line) = fm.get_line(cur_lo.line - 1) {
                    buf.push_str(&cur_line[.. cur_lo.col.to_usize()]);
                }
            }
            buf.push_str(substitute);
            prev_hi = cm.lookup_char_pos(sp.hi);
            prev_line = fm.get_line(prev_hi.line - 1);
        }
        push_trailing(&mut buf, prev_line, &prev_hi, None);
        // remove trailing newline
        buf.pop();
        buf
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

/// Used for emitting structured error messages and other diagnostic information.
#[must_use]
#[derive(Clone)]
pub struct DiagnosticBuilder<'a> {
    handler: &'a Handler,
    pub level: Level,
    pub message: String,
    pub code: Option<String>,
    pub span: MultiSpan,
    pub children: Vec<SubDiagnostic>,
}

/// For example a note attached to an error.
#[derive(Clone)]
pub struct SubDiagnostic {
    pub level: Level,
    pub message: String,
    pub span: MultiSpan,
    pub render_span: Option<RenderSpan>,
}

impl<'a> DiagnosticBuilder<'a> {
    /// Emit the diagnostic.
    pub fn emit(&mut self) {
        if self.cancelled() {
            return;
        }

        self.handler.emit.borrow_mut().emit_struct(&self);
        self.cancel();
        self.handler.panic_if_treat_err_as_bug();

        // if self.is_fatal() {
        //     panic!(FatalError);
        // }
    }

    /// Cancel the diagnostic (a structured diagnostic must either be emitted or
    /// cancelled or it will panic when dropped).
    /// BEWARE: if this DiagnosticBuilder is an error, then creating it will
    /// bump the error count on the Handler and cancelling it won't undo that.
    /// If you want to decrement the error count you should use `Handler::cancel`.
    pub fn cancel(&mut self) {
        self.level = Level::Cancelled;
    }

    pub fn cancelled(&self) -> bool {
        self.level == Level::Cancelled
    }

    pub fn is_fatal(&self) -> bool {
        self.level == Level::Fatal
    }

    /// Add a span/label to be included in the resulting snippet.
    /// This is pushed onto the `MultiSpan` that was created when the
    /// diagnostic was first built. If you don't call this function at
    /// all, and you just supplied a `Span` to create the diagnostic,
    /// then the snippet will just include that `Span`, which is
    /// called the primary span.
    pub fn span_label(&mut self, span: Span, label: &fmt::Display)
                      -> &mut DiagnosticBuilder<'a> {
        self.span.push_span_label(span, format!("{}", label));
        self
    }

    pub fn note_expected_found(&mut self,
                               label: &fmt::Display,
                               expected: &fmt::Display,
                               found: &fmt::Display)
                               -> &mut DiagnosticBuilder<'a>
    {
        // For now, just attach these as notes
        self.note(&format!("expected {} `{}`", label, expected));
        self.note(&format!("   found {} `{}`", label, found));
        self
    }

    pub fn note(&mut self, msg: &str) -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Note, msg, MultiSpan::new(), None);
        self
    }
    pub fn span_note<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Note, msg, sp.into(), None);
        self
    }
    pub fn warn(&mut self, msg: &str) -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Warning, msg, MultiSpan::new(), None);
        self
    }
    pub fn span_warn<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Warning, msg, sp.into(), None);
        self
    }
    pub fn help(&mut self , msg: &str) -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Help, msg, MultiSpan::new(), None);
        self
    }
    pub fn span_help<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Help, msg, sp.into(), None);
        self
    }
    /// Prints out a message with a suggested edit of the code.
    ///
    /// See `diagnostic::RenderSpan::Suggestion` for more information.
    pub fn span_suggestion<S: Into<MultiSpan>>(&mut self,
                                               sp: S,
                                               msg: &str,
                                               suggestion: String)
                                               -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Help, msg, MultiSpan::new(), Some(Suggestion(CodeSuggestion {
            msp: sp.into(),
            substitutes: vec![suggestion],
        })));
        self
    }

    pub fn set_span<S: Into<MultiSpan>>(&mut self, sp: S) -> &mut Self {
        self.span = sp.into();
        self
    }

    pub fn code(&mut self, s: String) -> &mut Self {
        self.code = Some(s);
        self
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn level(&self) -> Level {
        self.level
    }

    /// Convenience function for internal use, clients should use one of the
    /// struct_* methods on Handler.
    fn new(handler: &'a Handler,
           level: Level,
           message: &str) -> DiagnosticBuilder<'a> {
        DiagnosticBuilder {
            handler: handler,
            level: level,
            message: message.to_owned(),
            code: None,
            span: MultiSpan::new(),
            children: vec![],
        }
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    fn sub(&mut self,
           level: Level,
           message: &str,
           span: MultiSpan,
           render_span: Option<RenderSpan>) {
        let sub = SubDiagnostic {
            level: level,
            message: message.to_owned(),
            span: span,
            render_span: render_span,
        };
        self.children.push(sub);
    }
}

impl<'a> fmt::Debug for DiagnosticBuilder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.message.fmt(f)
    }
}

/// Destructor bomb - a DiagnosticBuilder must be either emitted or cancelled or
/// we emit a bug.
impl<'a> Drop for DiagnosticBuilder<'a> {
    fn drop(&mut self) {
        if !panicking() && !self.cancelled() {
            self.handler.emit.borrow_mut().emit(&MultiSpan::new(),
                                                "Error constructed but not emitted",
                                                None,
                                                Bug);
            panic!();
        }
    }
}

/// A handler deals with errors; certain errors
/// (fatal, bug, unimpl) may cause immediate exit,
/// others log errors for later reporting.
pub struct Handler {
    err_count: Cell<usize>,
    emit: RefCell<Box<Emitter>>,
    pub can_emit_warnings: bool,
    treat_err_as_bug: bool,
    continue_after_error: Cell<bool>,
    delayed_span_bug: RefCell<Option<(MultiSpan, String)>>,
}

impl Handler {
    pub fn with_tty_emitter(color_config: ColorConfig,
                            registry: Option<registry::Registry>,
                            can_emit_warnings: bool,
                            treat_err_as_bug: bool,
                            cm: Rc<CodeMapper>)
                            -> Handler {
        let emitter = Box::new(EmitterWriter::stderr(color_config, registry, cm,
                               snippet::FormatMode::EnvironmentSelected));
        Handler::with_emitter(can_emit_warnings, treat_err_as_bug, emitter)
    }

    pub fn with_emitter(can_emit_warnings: bool,
                        treat_err_as_bug: bool,
                        e: Box<Emitter>) -> Handler {
        Handler {
            err_count: Cell::new(0),
            emit: RefCell::new(e),
            can_emit_warnings: can_emit_warnings,
            treat_err_as_bug: treat_err_as_bug,
            continue_after_error: Cell::new(true),
            delayed_span_bug: RefCell::new(None),
        }
    }

    pub fn set_continue_after_error(&self, continue_after_error: bool) {
        self.continue_after_error.set(continue_after_error);
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
        if !self.can_emit_warnings {
            result.cancel();
        }
        result
    }
    pub fn struct_span_warn_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                              sp: S,
                                                              msg: &str,
                                                              code: &str)
                                                              -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Warning, msg);
        result.set_span(sp);
        result.code(code.to_owned());
        if !self.can_emit_warnings {
            result.cancel();
        }
        result
    }
    pub fn struct_warn<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(self, Level::Warning, msg);
        if !self.can_emit_warnings {
            result.cancel();
        }
        result
    }
    pub fn struct_span_err<'a, S: Into<MultiSpan>>(&'a self,
                                                   sp: S,
                                                   msg: &str)
                                                   -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        let mut result = DiagnosticBuilder::new(self, Level::Error, msg);
        result.set_span(sp);
        result
    }
    pub fn struct_span_err_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                             sp: S,
                                                             msg: &str,
                                                             code: &str)
                                                             -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        let mut result = DiagnosticBuilder::new(self, Level::Error, msg);
        result.set_span(sp);
        result.code(code.to_owned());
        result
    }
    pub fn struct_err<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        DiagnosticBuilder::new(self, Level::Error, msg)
    }
    pub fn struct_span_fatal<'a, S: Into<MultiSpan>>(&'a self,
                                                     sp: S,
                                                     msg: &str)
                                                     -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        let mut result = DiagnosticBuilder::new(self, Level::Fatal, msg);
        result.set_span(sp);
        result
    }
    pub fn struct_span_fatal_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                               sp: S,
                                                               msg: &str,
                                                               code: &str)
                                                               -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        let mut result = DiagnosticBuilder::new(self, Level::Fatal, msg);
        result.set_span(sp);
        result.code(code.to_owned());
        result
    }
    pub fn struct_fatal<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        DiagnosticBuilder::new(self, Level::Fatal, msg)
    }

    pub fn cancel(&mut self, err: &mut DiagnosticBuilder) {
        if err.level == Level::Error || err.level == Level::Fatal {
            assert!(self.has_errors());
            self.err_count.set(self.err_count.get() + 1);
        }
        err.cancel();
    }

    fn panic_if_treat_err_as_bug(&self) {
        if self.treat_err_as_bug {
            panic!("encountered error with `-Z treat_err_as_bug");
        }
    }

    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, msg: &str)
                                          -> FatalError {
        self.emit(&sp.into(), msg, Fatal);
        self.bump_err_count();
        self.panic_if_treat_err_as_bug();
        return FatalError;
    }
    pub fn span_fatal_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str)
                                                    -> FatalError {
        self.emit_with_code(&sp.into(), msg, code, Fatal);
        self.bump_err_count();
        self.panic_if_treat_err_as_bug();
        return FatalError;
    }
    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(&sp.into(), msg, Error);
        self.bump_err_count();
        self.panic_if_treat_err_as_bug();
    }
    pub fn span_err_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str) {
        self.emit_with_code(&sp.into(), msg, code, Error);
        self.bump_err_count();
        self.panic_if_treat_err_as_bug();
    }
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(&sp.into(), msg, Warning);
    }
    pub fn span_warn_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str) {
        self.emit_with_code(&sp.into(), msg, code, Warning);
    }
    pub fn span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.emit(&sp.into(), msg, Bug);
        panic!(ExplicitBug);
    }
    pub fn delay_span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        let mut delayed = self.delayed_span_bug.borrow_mut();
        *delayed = Some((sp.into(), msg.to_string()));
    }
    pub fn span_bug_no_panic<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(&sp.into(), msg, Bug);
        self.bump_err_count();
    }
    pub fn span_note_without_error<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit.borrow_mut().emit(&sp.into(), msg, None, Note);
    }
    pub fn span_unimpl<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.span_bug(sp, &format!("unimplemented {}", msg));
    }
    pub fn fatal(&self, msg: &str) -> FatalError {
        if self.treat_err_as_bug {
            self.bug(msg);
        }
        self.emit.borrow_mut().emit(&MultiSpan::new(), msg, None, Fatal);
        self.bump_err_count();
        FatalError
    }
    pub fn err(&self, msg: &str) {
        if self.treat_err_as_bug {
            self.bug(msg);
        }
        self.emit.borrow_mut().emit(&MultiSpan::new(), msg, None, Error);
        self.bump_err_count();
    }
    pub fn warn(&self, msg: &str) {
        self.emit.borrow_mut().emit(&MultiSpan::new(), msg, None, Warning);
    }
    pub fn note_without_error(&self, msg: &str) {
        self.emit.borrow_mut().emit(&MultiSpan::new(), msg, None, Note);
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.emit.borrow_mut().emit(&MultiSpan::new(), msg, None, Bug);
        panic!(ExplicitBug);
    }
    pub fn unimpl(&self, msg: &str) -> ! {
        self.bug(&format!("unimplemented {}", msg));
    }

    pub fn bump_err_count(&self) {
        self.err_count.set(self.err_count.get() + 1);
    }

    pub fn err_count(&self) -> usize {
        self.err_count.get()
    }

    pub fn has_errors(&self) -> bool {
        self.err_count.get() > 0
    }
    pub fn abort_if_errors(&self) {
        let s;
        match self.err_count.get() {
            0 => {
                let delayed_bug = self.delayed_span_bug.borrow();
                match *delayed_bug {
                    Some((ref span, ref errmsg)) => {
                        self.span_bug(span.clone(), errmsg);
                    },
                    _ => {}
                }

                return;
            }
            1 => s = "aborting due to previous error".to_string(),
            _  => {
                s = format!("aborting due to {} previous errors",
                            self.err_count.get());
            }
        }

        panic!(self.fatal(&s));
    }
    pub fn emit(&self,
                msp: &MultiSpan,
                msg: &str,
                lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().emit(&msp, msg, None, lvl);
        if !self.continue_after_error.get() { self.abort_if_errors(); }
    }
    pub fn emit_with_code(&self,
                          msp: &MultiSpan,
                          msg: &str,
                          code: &str,
                          lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().emit(&msp, msg, Some(code), lvl);
        if !self.continue_after_error.get() { self.abort_if_errors(); }
    }
}


#[derive(Copy, PartialEq, Clone, Debug)]
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
    pub fn color(self) -> term::color::Color {
        match self {
            Bug | Fatal | PhaseFatal | Error => term::color::BRIGHT_RED,
            Warning => term::color::YELLOW,
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

pub fn expect<T, M>(diag: &Handler, opt: Option<T>, msg: M) -> T where
    M: FnOnce() -> String,
{
    match opt {
        Some(t) => t,
        None => diag.bug(&msg()),
    }
}

/// True if we should use the old-skool error format style. This is
/// the default setting until the new errors are deemed stable enough
/// for general use.
///
/// FIXME(#33240)
#[cfg(not(test))]
pub fn check_old_skool() -> bool {
    use std::env;
    env::var("RUST_NEW_ERROR_FORMAT").is_err()
}

/// For unit tests, use the new format.
#[cfg(test)]
pub fn check_old_skool() -> bool {
    false
}
