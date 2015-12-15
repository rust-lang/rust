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

use codemap::{self, Span};
use diagnostics;
use errors::emitter::{Emitter, EmitterWriter};

use std::cell::{RefCell, Cell};
use std::{error, fmt};
use std::io::prelude::*;
use std::rc::Rc;
use term;

pub mod emitter;

#[derive(Clone)]
pub enum RenderSpan {
    /// A FullSpan renders with both with an initial line for the
    /// message, prefixed by file:linenum, followed by a summary of
    /// the source code covered by the span.
    FullSpan(Span),

    /// Similar to a FullSpan, but the cited position is the end of
    /// the span, instead of the start. Used, at least, for telling
    /// compiletest/runtest to look at the last line of the span
    /// (since `end_highlight_lines` displays an arrow to the end
    /// of the span).
    EndSpan(Span),

    /// A suggestion renders with both with an initial line for the
    /// message, prefixed by file:linenum, followed by a summary
    /// of hypothetical source code, where the `String` is spliced
    /// into the lines in place of the code covered by the span.
    Suggestion(Span, String),

    /// A FileLine renders with just a line for the message prefixed
    /// by file:linenum.
    FileLine(Span),
}

impl RenderSpan {
    fn span(&self) -> Span {
        match *self {
            FullSpan(s) |
            Suggestion(s, _) |
            EndSpan(s) |
            FileLine(s) =>
                s
        }
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

/// A handler deals with errors; certain errors
/// (fatal, bug, unimpl) may cause immediate exit,
/// others log errors for later reporting.
pub struct Handler {
    err_count: Cell<usize>,
    emit: RefCell<Box<Emitter>>,
    pub can_emit_warnings: bool,
    treat_err_as_bug: bool,
    delayed_span_bug: RefCell<Option<(codemap::Span, String)>>,
}

impl Handler {
    pub fn new(color_config: ColorConfig,
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
            delayed_span_bug: RefCell::new(None),
        }
    }

    pub fn span_fatal(&self, sp: Span, msg: &str) -> FatalError {
        if self.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        self.emit(Some(sp), msg, Fatal);
        return FatalError;
    }
    pub fn span_fatal_with_code(&self, sp: Span, msg: &str, code: &str) -> FatalError {
        if self.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        self.emit_with_code(Some(sp), msg, code, Fatal);
        return FatalError;
    }
    pub fn span_err(&self, sp: Span, msg: &str) {
        if self.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        self.emit(Some(sp), msg, Error);
        self.bump_err_count();
    }
    pub fn span_err_with_code(&self, sp: Span, msg: &str, code: &str) {
        if self.treat_err_as_bug {
            self.span_bug(sp, msg);
        }
        self.emit_with_code(Some(sp), msg, code, Error);
        self.bump_err_count();
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        self.emit(Some(sp), msg, Warning);
    }
    pub fn span_warn_with_code(&self, sp: Span, msg: &str, code: &str) {
        self.emit_with_code(Some(sp), msg, code, Warning);
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.emit(Some(sp), msg, Note);
    }
    pub fn span_end_note(&self, sp: Span, msg: &str) {
        self.custom_emit(EndSpan(sp), msg, Note);
    }
    pub fn span_help(&self, sp: Span, msg: &str) {
        self.emit(Some(sp), msg, Help);
    }
    /// Prints out a message with a suggested edit of the code.
    ///
    /// See `diagnostic::RenderSpan::Suggestion` for more information.
    pub fn span_suggestion(&self, sp: Span, msg: &str, suggestion: String) {
        self.custom_emit(Suggestion(sp, suggestion), msg, Help);
    }
    pub fn fileline_note(&self, sp: Span, msg: &str) {
        self.custom_emit(FileLine(sp), msg, Note);
    }
    pub fn fileline_help(&self, sp: Span, msg: &str) {
        self.custom_emit(FileLine(sp), msg, Help);
    }
    pub fn span_bug(&self, sp: Span, msg: &str) -> ! {
        self.emit(Some(sp), msg, Bug);
        panic!(ExplicitBug);
    }
    pub fn delay_span_bug(&self, sp: Span, msg: &str) {
        let mut delayed = self.delayed_span_bug.borrow_mut();
        *delayed = Some((sp, msg.to_string()));
    }
    pub fn span_bug_no_panic(&self, sp: Span, msg: &str) {
        self.emit(Some(sp), msg, Bug);
        self.bump_err_count();
    }
    pub fn span_unimpl(&self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp, &format!("unimplemented {}", msg));
    }
    pub fn fatal(&self, msg: &str) -> FatalError {
        if self.treat_err_as_bug {
            self.bug(msg);
        }
        self.emit.borrow_mut().emit(None, msg, None, Fatal);
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
    pub fn note(&self, msg: &str) {
        self.emit.borrow_mut().emit(None, msg, None, Note);
    }
    pub fn help(&self, msg: &str) {
        self.emit.borrow_mut().emit(None, msg, None, Help);
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
                    Some((span, ref errmsg)) => {
                        self.span_bug(span, errmsg);
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

        panic!(self.fatal(&s[..]));
    }

    pub fn emit(&self,
                sp: Option<Span>,
                msg: &str,
                lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().emit(sp, msg, None, lvl);
    }

    pub fn emit_with_code(&self,
                          sp: Option<Span>,
                          msg: &str,
                          code: &str,
                          lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().emit(sp, msg, Some(code), lvl);
    }

    pub fn custom_emit(&self, sp: RenderSpan, msg: &str, lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().custom_emit(sp, msg, lvl);
    }
}


#[derive(Copy, PartialEq, Clone, Debug)]
pub enum Level {
    Bug,
    Fatal,
    Error,
    Warning,
    Note,
    Help,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Display;

        match *self {
            Bug => "error: internal compiler error".fmt(f),
            Fatal | Error => "error".fmt(f),
            Warning => "warning".fmt(f),
            Note => "note".fmt(f),
            Help => "help".fmt(f),
        }
    }
}

impl Level {
    fn color(self) -> term::color::Color {
        match self {
            Bug | Fatal | Error => term::color::BRIGHT_RED,
            Warning => term::color::BRIGHT_YELLOW,
            Note => term::color::BRIGHT_GREEN,
            Help => term::color::BRIGHT_CYAN,
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

#[cfg(test)]
mod test {
    use super::Level;
    use emitter::EmitterWriter;
    use codemap::{mk_sp, CodeMap};
    use std::sync::{Arc, Mutex};
    use std::io::{self, Write};
    use std::str::from_utf8;

    // Diagnostic doesn't align properly in span where line number increases by one digit
    #[test]
    fn test_hilight_suggestion_issue_11715() {
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl Write for Sink {
            fn write(&mut self, data: &[u8]) -> io::Result<usize> {
                Write::write(&mut *self.0.lock().unwrap(), data)
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        let data = Arc::new(Mutex::new(Vec::new()));
        let mut ew = EmitterWriter::new(Box::new(Sink(data.clone())), None);
        let cm = CodeMap::new();
        let content = "abcdefg
        koksi
        line3
        line4
        cinq
        line6
        line7
        line8
        line9
        line10
        e-lä-vän
        tolv
        dreizehn
        ";
        let file = cm.new_filemap_and_lines("dummy.txt", content);
        let start = file.lines.borrow()[7];
        let end = file.lines.borrow()[11];
        let sp = mk_sp(start, end);
        let lvl = Level::Error;
        println!("span_to_lines");
        let lines = cm.span_to_lines(sp);
        println!("highlight_lines");
        ew.highlight_lines(&cm, sp, lvl, lines).unwrap();
        println!("done");
        let vec = data.lock().unwrap().clone();
        let vec: &[u8] = &vec;
        let str = from_utf8(vec).unwrap();
        println!("{}", str);
        assert_eq!(str, "dummy.txt: 8         line8\n\
                         dummy.txt: 9         line9\n\
                         dummy.txt:10         line10\n\
                         dummy.txt:11         e-lä-vän\n\
                         dummy.txt:12         tolv\n");
    }
}
