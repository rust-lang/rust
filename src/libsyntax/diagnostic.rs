// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate libc;

use codemap::{Pos, Span};
use codemap;

use std::cell::{RefCell, Cell};
use std::fmt;
use std::io;
use std::iter::range;
use std::strbuf::StrBuf;
use term;

// maximum number of lines we will print for each error; arbitrary.
static MAX_LINES: uint = 6u;

#[deriving(Clone)]
pub enum RenderSpan {
    /// A FullSpan renders with both with an initial line for the
    /// message, prefixed by file:linenum, followed by a summary of
    /// the source code covered by the span.
    FullSpan(Span),

    /// A FileLine renders with just a line for the message prefixed
    /// by file:linenum.
    FileLine(Span),
}

impl RenderSpan {
    fn span(self) -> Span {
        match self {
            FullSpan(s) | FileLine(s) => s
        }
    }
    fn is_full_span(&self) -> bool {
        match self {
            &FullSpan(..) => true,
            &FileLine(..) => false,
        }
    }
}

pub trait Emitter {
    fn emit(&mut self, cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str, lvl: Level);
    fn custom_emit(&mut self, cm: &codemap::CodeMap,
                   sp: RenderSpan, msg: &str, lvl: Level);
}

/// This structure is used to signify that a task has failed with a fatal error
/// from the diagnostics. You can use this with the `Any` trait to figure out
/// how a rustc task died (if so desired).
pub struct FatalError;

/// Signifies that the compiler died with an explicit call to `.bug`
/// or `.span_bug` rather than a failed assertion, etc.
pub struct ExplicitBug;

// a span-handler is like a handler but also
// accepts span information for source-location
// reporting.
pub struct SpanHandler {
    pub handler: Handler,
    pub cm: codemap::CodeMap,
}

impl SpanHandler {
    pub fn span_fatal(&self, sp: Span, msg: &str) -> ! {
        self.handler.emit(Some((&self.cm, sp)), msg, Fatal);
        fail!(FatalError);
    }
    pub fn span_err(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&self.cm, sp)), msg, Error);
        self.handler.bump_err_count();
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&self.cm, sp)), msg, Warning);
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&self.cm, sp)), msg, Note);
    }
    pub fn span_end_note(&self, sp: Span, msg: &str) {
        self.handler.custom_emit(&self.cm, FullSpan(sp), msg, Note);
    }
    pub fn fileline_note(&self, sp: Span, msg: &str) {
        self.handler.custom_emit(&self.cm, FileLine(sp), msg, Note);
    }
    pub fn span_bug(&self, sp: Span, msg: &str) -> ! {
        self.handler.emit(Some((&self.cm, sp)), msg, Bug);
        fail!(ExplicitBug);
    }
    pub fn span_unimpl(&self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp, "unimplemented ".to_owned() + msg);
    }
    pub fn handler<'a>(&'a self) -> &'a Handler {
        &self.handler
    }
}

// a handler deals with errors; certain errors
// (fatal, bug, unimpl) may cause immediate exit,
// others log errors for later reporting.
pub struct Handler {
    err_count: Cell<uint>,
    emit: RefCell<Box<Emitter:Send>>,
}

impl Handler {
    pub fn fatal(&self, msg: &str) -> ! {
        self.emit.borrow_mut().emit(None, msg, Fatal);
        fail!(FatalError);
    }
    pub fn err(&self, msg: &str) {
        self.emit.borrow_mut().emit(None, msg, Error);
        self.bump_err_count();
    }
    pub fn bump_err_count(&self) {
        self.err_count.set(self.err_count.get() + 1u);
    }
    pub fn err_count(&self) -> uint {
        self.err_count.get()
    }
    pub fn has_errors(&self) -> bool {
        self.err_count.get()> 0u
    }
    pub fn abort_if_errors(&self) {
        let s;
        match self.err_count.get() {
          0u => return,
          1u => s = "aborting due to previous error".to_owned(),
          _  => {
            s = format!("aborting due to {} previous errors",
                     self.err_count.get());
          }
        }
        self.fatal(s);
    }
    pub fn warn(&self, msg: &str) {
        self.emit.borrow_mut().emit(None, msg, Warning);
    }
    pub fn note(&self, msg: &str) {
        self.emit.borrow_mut().emit(None, msg, Note);
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.emit.borrow_mut().emit(None, msg, Bug);
        fail!(ExplicitBug);
    }
    pub fn unimpl(&self, msg: &str) -> ! {
        self.bug("unimplemented ".to_owned() + msg);
    }
    pub fn emit(&self,
                cmsp: Option<(&codemap::CodeMap, Span)>,
                msg: &str,
                lvl: Level) {
        self.emit.borrow_mut().emit(cmsp, msg, lvl);
    }
    pub fn custom_emit(&self, cm: &codemap::CodeMap,
                       sp: RenderSpan, msg: &str, lvl: Level) {
        self.emit.borrow_mut().custom_emit(cm, sp, msg, lvl);
    }
}

pub fn mk_span_handler(handler: Handler, cm: codemap::CodeMap) -> SpanHandler {
    SpanHandler {
        handler: handler,
        cm: cm,
    }
}

pub fn default_handler() -> Handler {
    mk_handler(box EmitterWriter::stderr())
}

pub fn mk_handler(e: Box<Emitter:Send>) -> Handler {
    Handler {
        err_count: Cell::new(0),
        emit: RefCell::new(e),
    }
}

#[deriving(Eq)]
pub enum Level {
    Bug,
    Fatal,
    Error,
    Warning,
    Note,
}

impl fmt::Show for Level {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Show;

        match *self {
            Bug => "error: internal compiler error".fmt(f),
            Fatal | Error => "error".fmt(f),
            Warning => "warning".fmt(f),
            Note => "note".fmt(f),
        }
    }
}

impl Level {
    fn color(self) -> term::color::Color {
        match self {
            Bug | Fatal | Error => term::color::BRIGHT_RED,
            Warning => term::color::BRIGHT_YELLOW,
            Note => term::color::BRIGHT_GREEN
        }
    }
}

fn print_maybe_styled(w: &mut EmitterWriter,
                      msg: &str,
                      color: term::attr::Attr) -> io::IoResult<()> {
    match w.dst {
        Terminal(ref mut t) => {
            try!(t.attr(color));
            try!(t.write_str(msg));
            try!(t.reset());
            Ok(())
        }
        Raw(ref mut w) => {
            w.write_str(msg)
        }
    }
}

fn print_diagnostic(dst: &mut EmitterWriter,
                    topic: &str, lvl: Level, msg: &str) -> io::IoResult<()> {
    if !topic.is_empty() {
        try!(write!(&mut dst.dst, "{} ", topic));
    }

    try!(print_maybe_styled(dst, format!("{}: ", lvl.to_str()),
                            term::attr::ForegroundColor(lvl.color())));
    try!(print_maybe_styled(dst, format!("{}\n", msg), term::attr::Bold));
    Ok(())
}

pub struct EmitterWriter {
    dst: Destination,
}

enum Destination {
    Terminal(term::Terminal<io::stdio::StdWriter>),
    Raw(Box<Writer:Send>),
}

impl EmitterWriter {
    pub fn stderr() -> EmitterWriter {
        let stderr = io::stderr();
        if stderr.get_ref().isatty() {
            let dst = match term::Terminal::new(stderr.unwrap()) {
                Ok(t) => Terminal(t),
                Err(..) => Raw(box io::stderr()),
            };
            EmitterWriter { dst: dst }
        } else {
            EmitterWriter { dst: Raw(box stderr) }
        }
    }

    pub fn new(dst: Box<Writer:Send>) -> EmitterWriter {
        EmitterWriter { dst: Raw(dst) }
    }
}

impl Writer for Destination {
    fn write(&mut self, bytes: &[u8]) -> io::IoResult<()> {
        match *self {
            Terminal(ref mut t) => t.write(bytes),
            Raw(ref mut w) => w.write(bytes),
        }
    }
}

impl Emitter for EmitterWriter {
    fn emit(&mut self,
            cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str,
            lvl: Level) {
        let error = match cmsp {
            Some((cm, sp)) => emit(self, cm, FullSpan(sp), msg, lvl, false),
            None => print_diagnostic(self, "", lvl, msg),
        };

        match error {
            Ok(()) => {}
            Err(e) => fail!("failed to print diagnostics: {}", e),
        }
    }

    fn custom_emit(&mut self, cm: &codemap::CodeMap,
                   sp: RenderSpan, msg: &str, lvl: Level) {
        match emit(self, cm, sp, msg, lvl, true) {
            Ok(()) => {}
            Err(e) => fail!("failed to print diagnostics: {}", e),
        }
    }
}

fn emit(dst: &mut EmitterWriter, cm: &codemap::CodeMap, rsp: RenderSpan,
        msg: &str, lvl: Level, custom: bool) -> io::IoResult<()> {
    let sp = rsp.span();
    let ss = cm.span_to_str(sp);
    let lines = cm.span_to_lines(sp);
    if custom {
        // we want to tell compiletest/runtest to look at the last line of the
        // span (since `custom_highlight_lines` displays an arrow to the end of
        // the span)
        let span_end = Span { lo: sp.hi, hi: sp.hi, expn_info: sp.expn_info};
        let ses = cm.span_to_str(span_end);
        try!(print_diagnostic(dst, ses, lvl, msg));
        if rsp.is_full_span() {
            try!(custom_highlight_lines(dst, cm, sp, lvl, lines));
        }
    } else {
        try!(print_diagnostic(dst, ss, lvl, msg));
        if rsp.is_full_span() {
            try!(highlight_lines(dst, cm, sp, lvl, lines));
        }
    }
    print_macro_backtrace(dst, cm, sp)
}

fn highlight_lines(err: &mut EmitterWriter,
                   cm: &codemap::CodeMap,
                   sp: Span,
                   lvl: Level,
                   lines: codemap::FileLines) -> io::IoResult<()> {
    let fm = &*lines.file;

    let mut elided = false;
    let mut display_lines = lines.lines.as_slice();
    if display_lines.len() > MAX_LINES {
        display_lines = display_lines.slice(0u, MAX_LINES);
        elided = true;
    }
    // Print the offending lines
    for line in display_lines.iter() {
        try!(write!(&mut err.dst, "{}:{} {}\n", fm.name, *line + 1,
                    fm.get_line(*line as int)));
    }
    if elided {
        let last_line = display_lines[display_lines.len() - 1u];
        let s = format!("{}:{} ", fm.name, last_line + 1u);
        try!(write!(&mut err.dst, "{0:1$}...\n", "", s.len()));
    }

    // FIXME (#3260)
    // If there's one line at fault we can easily point to the problem
    if lines.lines.len() == 1u {
        let lo = cm.lookup_char_pos(sp.lo);
        let mut digits = 0u;
        let mut num = (*lines.lines.get(0) + 1u) / 10u;

        // how many digits must be indent past?
        while num > 0u { num /= 10u; digits += 1u; }

        // indent past |name:## | and the 0-offset column location
        let left = fm.name.len() + digits + lo.col.to_uint() + 3u;
        let mut s = StrBuf::new();
        // Skip is the number of characters we need to skip because they are
        // part of the 'filename:line ' part of the previous line.
        let skip = fm.name.len() + digits + 3u;
        for _ in range(0, skip) {
            s.push_char(' ');
        }
        let orig = fm.get_line(*lines.lines.get(0) as int);
        for pos in range(0u, left-skip) {
            let cur_char = orig[pos] as char;
            // Whenever a tab occurs on the previous line, we insert one on
            // the error-point-squiggly-line as well (instead of a space).
            // That way the squiggly line will usually appear in the correct
            // position.
            match cur_char {
                '\t' => s.push_char('\t'),
                _ => s.push_char(' '),
            };
        }
        try!(write!(&mut err.dst, "{}", s));
        let mut s = StrBuf::from_str("^");
        let hi = cm.lookup_char_pos(sp.hi);
        if hi.col != lo.col {
            // the ^ already takes up one space
            let num_squigglies = hi.col.to_uint()-lo.col.to_uint()-1u;
            for _ in range(0, num_squigglies) {
                s.push_char('~');
            }
        }
        try!(print_maybe_styled(err, s.into_owned() + "\n",
                                term::attr::ForegroundColor(lvl.color())));
    }
    Ok(())
}

// Here are the differences between this and the normal `highlight_lines`:
// `custom_highlight_lines` will always put arrow on the last byte of the
// span (instead of the first byte). Also, when the span is too long (more
// than 6 lines), `custom_highlight_lines` will print the first line, then
// dot dot dot, then last line, whereas `highlight_lines` prints the first
// six lines.
fn custom_highlight_lines(w: &mut EmitterWriter,
                          cm: &codemap::CodeMap,
                          sp: Span,
                          lvl: Level,
                          lines: codemap::FileLines)
                          -> io::IoResult<()> {
    let fm = &*lines.file;

    let lines = lines.lines.as_slice();
    if lines.len() > MAX_LINES {
        try!(write!(&mut w.dst, "{}:{} {}\n", fm.name,
                    lines[0] + 1, fm.get_line(lines[0] as int)));
        try!(write!(&mut w.dst, "...\n"));
        let last_line = lines[lines.len()-1];
        try!(write!(&mut w.dst, "{}:{} {}\n", fm.name,
                    last_line + 1, fm.get_line(last_line as int)));
    } else {
        for line in lines.iter() {
            try!(write!(&mut w.dst, "{}:{} {}\n", fm.name,
                        *line + 1, fm.get_line(*line as int)));
        }
    }
    let last_line_start = format!("{}:{} ", fm.name, lines[lines.len()-1]+1);
    let hi = cm.lookup_char_pos(sp.hi);
    // Span seems to use half-opened interval, so subtract 1
    let skip = last_line_start.len() + hi.col.to_uint() - 1;
    let mut s = StrBuf::new();
    for _ in range(0, skip) {
        s.push_char(' ');
    }
    s.push_char('^');
    s.push_char('\n');
    print_maybe_styled(w,
                       s.into_owned(),
                       term::attr::ForegroundColor(lvl.color()))
}

fn print_macro_backtrace(w: &mut EmitterWriter,
                         cm: &codemap::CodeMap,
                         sp: Span)
                         -> io::IoResult<()> {
    for ei in sp.expn_info.iter() {
        let ss = ei.callee.span.as_ref().map_or("".to_owned(), |span| cm.span_to_str(*span));
        let (pre, post) = match ei.callee.format {
            codemap::MacroAttribute => ("#[", "]"),
            codemap::MacroBang => ("", "!")
        };
        try!(print_diagnostic(w, ss, Note,
                              format!("in expansion of {}{}{}", pre,
                                      ei.callee.name, post)));
        let ss = cm.span_to_str(ei.call_site);
        try!(print_diagnostic(w, ss, Note, "expansion site"));
        try!(print_macro_backtrace(w, cm, ei.call_site));
    }
    Ok(())
}

pub fn expect<T:Clone>(diag: &SpanHandler, opt: Option<T>, msg: || -> ~str) -> T {
    match opt {
       Some(ref t) => (*t).clone(),
       None => diag.handler().bug(msg()),
    }
}
