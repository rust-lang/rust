// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use errors::emitter::ColorConfig;

use self::Level::*;
use self::RenderSpan::*;

use codemap::{self, CodeMap, MultiSpan};
use diagnostics;
use errors::emitter::{Emitter, EmitterWriter};

use std::cell::{RefCell, Cell};
use std::{error, fmt};
use std::rc::Rc;
use term;

pub mod emitter;
pub mod json;

#[derive(Clone)]
pub enum RenderSpan {
    /// A FullSpan renders with both with an initial line for the
    /// message, prefixed by file:linenum, followed by a summary of
    /// the source code covered by the span.
    FullSpan(MultiSpan),

    /// Similar to a FullSpan, but the cited position is the end of
    /// the span, instead of the start. Used, at least, for telling
    /// compiletest/runtest to look at the last line of the span
    /// (since `end_highlight_lines` displays an arrow to the end
    /// of the span).
    EndSpan(MultiSpan),

    /// A suggestion renders with both with an initial line for the
    /// message, prefixed by file:linenum, followed by a summary
    /// of hypothetical source code, where each `String` is spliced
    /// into the lines in place of the code covered by each span.
    Suggestion(CodeSuggestion),

    /// A FileLine renders with just a line for the message prefixed
    /// by file:linenum.
    FileLine(MultiSpan),
}

#[derive(Clone)]
pub struct CodeSuggestion {
    msp: MultiSpan,
    substitutes: Vec<String>,
}

impl RenderSpan {
    fn span(&self) -> &MultiSpan {
        match *self {
            FullSpan(ref msp) |
            Suggestion(CodeSuggestion { ref msp, .. }) |
            EndSpan(ref msp) |
            FileLine(ref msp) =>
                msp
        }
    }
}

impl CodeSuggestion {
    /// Returns the assembled code suggestion.
    pub fn splice_lines(&self, cm: &CodeMap) -> String {
        use codemap::{CharPos, Loc, Pos};

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
        let bounds = self.msp.to_span_bounds();
        let lines = cm.span_to_lines(bounds).unwrap();
        assert!(!lines.lines.is_empty());

        // This isn't strictly necessary, but would in all likelyhood be an error
        assert_eq!(self.msp.spans.len(), self.substitutes.len());

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
        let mut prev_hi = cm.lookup_char_pos(bounds.lo);
        prev_hi.col = CharPos::from_usize(0);

        let mut prev_line = fm.get_line(lines.lines[0].line_index);
        let mut buf = String::new();

        for (sp, substitute) in self.msp.spans.iter().zip(self.substitutes.iter()) {
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
pub struct DiagnosticBuilder<'a> {
    emitter: &'a RefCell<Box<Emitter>>,
    level: Level,
    message: String,
    code: Option<String>,
    span: Option<MultiSpan>,
    children: Vec<SubDiagnostic>,
}

/// For example a note attached to an error.
struct SubDiagnostic {
    level: Level,
    message: String,
    span: Option<MultiSpan>,
    render_span: Option<RenderSpan>,
}

impl<'a> DiagnosticBuilder<'a> {
    /// Emit the diagnostic.
    pub fn emit(&mut self) {
        if self.cancelled() {
            return;
        }

        self.emitter.borrow_mut().emit_struct(&self);
        self.cancel();

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

    pub fn note(&mut self , msg: &str) -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Note, msg, None, None);
        self
    }
    pub fn span_note<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Note, msg, Some(sp.into()), None);
        self
    }
    pub fn warn(&mut self, msg: &str) -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Warning, msg, None, None);
        self
    }
    pub fn span_warn<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Warning, msg, Some(sp.into()), None);
        self
    }
    pub fn help(&mut self , msg: &str) -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Help, msg, None, None);
        self
    }
    pub fn span_help<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Help, msg, Some(sp.into()), None);
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
        self.sub(Level::Help, msg, None, Some(Suggestion(CodeSuggestion {
            msp: sp.into(),
            substitutes: vec![suggestion],
        })));
        self
    }
    pub fn span_end_note<S: Into<MultiSpan>>(&mut self,
                                             sp: S,
                                             msg: &str)
                                             -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Note, msg, None, Some(EndSpan(sp.into())));
        self
    }
    pub fn fileline_warn<S: Into<MultiSpan>>(&mut self,
                                             sp: S,
                                             msg: &str)
                                             -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Warning, msg, None, Some(FileLine(sp.into())));
        self
    }
    pub fn fileline_note<S: Into<MultiSpan>>(&mut self,
                                             sp: S,
                                             msg: &str)
                                             -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Note, msg, None, Some(FileLine(sp.into())));
        self
    }
    pub fn fileline_help<S: Into<MultiSpan>>(&mut self,
                                             sp: S,
                                             msg: &str)
                                             -> &mut DiagnosticBuilder<'a> {
        self.sub(Level::Help, msg, None, Some(FileLine(sp.into())));
        self
    }

    pub fn span<S: Into<MultiSpan>>(&mut self, sp: S) -> &mut Self {
        self.span = Some(sp.into());
        self
    }

    pub fn code(&mut self, s: String) -> &mut Self {
        self.code = Some(s);
        self
    }

    /// Convenience function for internal use, clients should use one of the
    /// struct_* methods on Handler.
    fn new(emitter: &'a RefCell<Box<Emitter>>,
           level: Level,
           message: &str) -> DiagnosticBuilder<'a> {
        DiagnosticBuilder {
            emitter: emitter,
            level: level,
            message: message.to_owned(),
            code: None,
            span: None,
            children: vec![],
        }
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    fn sub(&mut self,
           level: Level,
           message: &str,
           span: Option<MultiSpan>,
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
        if !self.cancelled() {
            self.emitter.borrow_mut().emit(None, "Error constructed but not emitted", None, Bug);
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
                            registry: Option<diagnostics::registry::Registry>,
                            can_emit_warnings: bool,
                            treat_err_as_bug: bool,
                            cm: Rc<codemap::CodeMap>)
                            -> Handler {
        let emitter = Box::new(EmitterWriter::stderr(color_config, registry, cm));
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
        DiagnosticBuilder::new(&self.emit, Level::Cancelled, "")
    }

    pub fn struct_span_warn<'a, S: Into<MultiSpan>>(&'a self,
                                                    sp: S,
                                                    msg: &str)
                                                    -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(&self.emit, Level::Warning, msg);
        result.span(sp);
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
        let mut result = DiagnosticBuilder::new(&self.emit, Level::Warning, msg);
        result.span(sp);
        result.code(code.to_owned());
        if !self.can_emit_warnings {
            result.cancel();
        }
        result
    }
    pub fn struct_warn<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        let mut result = DiagnosticBuilder::new(&self.emit, Level::Warning, msg);
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
        let mut result = DiagnosticBuilder::new(&self.emit, Level::Error, msg);
        result.span(sp);
        result
    }
    pub fn struct_span_err_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                             sp: S,
                                                             msg: &str,
                                                             code: &str)
                                                             -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        let mut result = DiagnosticBuilder::new(&self.emit, Level::Error, msg);
        result.span(sp);
        result.code(code.to_owned());
        result
    }
    pub fn struct_err<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        DiagnosticBuilder::new(&self.emit, Level::Error, msg)
    }
    pub fn struct_span_fatal<'a, S: Into<MultiSpan>>(&'a self,
                                                     sp: S,
                                                     msg: &str)
                                                     -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        let mut result = DiagnosticBuilder::new(&self.emit, Level::Fatal, msg);
        result.span(sp);
        result
    }
    pub fn struct_span_fatal_with_code<'a, S: Into<MultiSpan>>(&'a self,
                                                               sp: S,
                                                               msg: &str,
                                                               code: &str)
                                                               -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        let mut result = DiagnosticBuilder::new(&self.emit, Level::Fatal, msg);
        result.span(sp);
        result.code(code.to_owned());
        result
    }
    pub fn struct_fatal<'a>(&'a self, msg: &str) -> DiagnosticBuilder<'a> {
        self.bump_err_count();
        DiagnosticBuilder::new(&self.emit, Level::Fatal, msg)
    }

    pub fn cancel(&mut self, err: &mut DiagnosticBuilder) {
        if err.level == Level::Error || err.level == Level::Fatal {
            assert!(self.has_errors());
            self.err_count.set(self.err_count.get() + 1);
        }
        err.cancel();
    }

    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> FatalError {
        if self.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        self.emit(Some(&sp.into()), msg, Fatal);
        self.bump_err_count();
        return FatalError;
    }
    pub fn span_fatal_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str)
    -> FatalError {
        if self.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        self.emit_with_code(Some(&sp.into()), msg, code, Fatal);
        self.bump_err_count();
        return FatalError;
    }
    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        if self.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        self.emit(Some(&sp.into()), msg, Error);
        self.bump_err_count();
    }
    pub fn span_err_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str) {
        if self.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        self.emit_with_code(Some(&sp.into()), msg, code, Error);
        self.bump_err_count();
    }
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(Some(&sp.into()), msg, Warning);
    }
    pub fn span_warn_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: &str) {
        self.emit_with_code(Some(&sp.into()), msg, code, Warning);
    }
    pub fn span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.emit(Some(&sp.into()), msg, Bug);
        panic!(ExplicitBug);
    }
    pub fn delay_span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        let mut delayed = self.delayed_span_bug.borrow_mut();
        *delayed = Some((sp.into(), msg.to_string()));
    }
    pub fn span_bug_no_panic<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit(Some(&sp.into()), msg, Bug);
        self.bump_err_count();
    }
    pub fn span_note_without_error<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.emit.borrow_mut().emit(Some(&sp.into()), msg, None, Note);
    }
    pub fn span_unimpl<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.span_bug(sp, &format!("unimplemented {}", msg));
    }
    pub fn fatal(&self, msg: &str) -> FatalError {
        if self.treat_err_as_bug {
            self.bug(msg);
        }
        self.emit.borrow_mut().emit(None, msg, None, Fatal);
        self.bump_err_count();
        FatalError
    }
    pub fn err(&self, msg: &str) {
        if self.treat_err_as_bug {
            self.bug(msg);
        }
        self.emit.borrow_mut().emit(None, msg, None, Error);
        self.bump_err_count();
    }
    pub fn warn(&self, msg: &str) {
        self.emit.borrow_mut().emit(None, msg, None, Warning);
    }
    pub fn note_without_error(&self, msg: &str) {
        self.emit.borrow_mut().emit(None, msg, None, Note);
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.emit.borrow_mut().emit(None, msg, None, Bug);
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
                msp: Option<&MultiSpan>,
                msg: &str,
                lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().emit(msp, msg, None, lvl);
        if !self.continue_after_error.get() { self.abort_if_errors(); }
    }
    pub fn emit_with_code(&self,
                          msp: Option<&MultiSpan>,
                          msg: &str,
                          code: &str,
                          lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().emit(msp, msg, Some(code), lvl);
        if !self.continue_after_error.get() { self.abort_if_errors(); }
    }
    pub fn custom_emit(&self, rsp: RenderSpan, msg: &str, lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().custom_emit(&rsp, msg, lvl);
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
    fn color(self) -> term::color::Color {
        match self {
            Bug | Fatal | PhaseFatal | Error => term::color::BRIGHT_RED,
            Warning => term::color::BRIGHT_YELLOW,
            Note => term::color::BRIGHT_GREEN,
            Help => term::color::BRIGHT_CYAN,
            Cancelled => unreachable!(),
        }
    }

    fn to_str(self) -> &'static str {
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
