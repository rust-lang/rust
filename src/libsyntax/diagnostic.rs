// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use codemap::{Pos, Span};
use codemap;

use std::cell::Cell;
use std::io;
use std::io::stdio::StdWriter;
use std::iter::range;
use std::local_data;
use term;

static BUG_REPORT_URL: &'static str =
    "http://static.rust-lang.org/doc/master/complement-bugreport.html";
// maximum number of lines we will print for each error; arbitrary.
static MAX_LINES: uint = 6u;

pub trait Emitter {
    fn emit(&self, cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str, lvl: Level);
    fn custom_emit(&self, cm: &codemap::CodeMap,
                   sp: Span, msg: &str, lvl: Level);
}

/// This structure is used to signify that a task has failed with a fatal error
/// from the diagnostics. You can use this with the `Any` trait to figure out
/// how a rustc task died (if so desired).
pub struct FatalError;

// a span-handler is like a handler but also
// accepts span information for source-location
// reporting.
pub struct SpanHandler {
    handler: @Handler,
    cm: @codemap::CodeMap,
}

impl SpanHandler {
    pub fn span_fatal(&self, sp: Span, msg: &str) -> ! {
        self.handler.emit(Some((&*self.cm, sp)), msg, Fatal);
        fail!(FatalError);
    }
    pub fn span_err(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&*self.cm, sp)), msg, Error);
        self.handler.bump_err_count();
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&*self.cm, sp)), msg, Warning);
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&*self.cm, sp)), msg, Note);
    }
    pub fn span_end_note(&self, sp: Span, msg: &str) {
        self.handler.custom_emit(&*self.cm, sp, msg, Note);
    }
    pub fn span_bug(&self, sp: Span, msg: &str) -> ! {
        self.span_fatal(sp, ice_msg(msg));
    }
    pub fn span_unimpl(&self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp, ~"unimplemented " + msg);
    }
    pub fn handler(&self) -> @Handler {
        self.handler
    }
}

// a handler deals with errors; certain errors
// (fatal, bug, unimpl) may cause immediate exit,
// others log errors for later reporting.
pub struct Handler {
    err_count: Cell<uint>,
    emit: DefaultEmitter,
}

impl Handler {
    pub fn fatal(&self, msg: &str) -> ! {
        self.emit.emit(None, msg, Fatal);
        fail!(FatalError);
    }
    pub fn err(&self, msg: &str) {
        self.emit.emit(None, msg, Error);
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
          1u => s = ~"aborting due to previous error",
          _  => {
            s = format!("aborting due to {} previous errors",
                     self.err_count.get());
          }
        }
        self.fatal(s);
    }
    pub fn warn(&self, msg: &str) {
        self.emit.emit(None, msg, Warning);
    }
    pub fn note(&self, msg: &str) {
        self.emit.emit(None, msg, Note);
    }
    pub fn bug(&self, msg: &str) -> ! {
        self.fatal(ice_msg(msg));
    }
    pub fn unimpl(&self, msg: &str) -> ! {
        self.bug(~"unimplemented " + msg);
    }
    pub fn emit(&self,
                cmsp: Option<(&codemap::CodeMap, Span)>,
                msg: &str,
                lvl: Level) {
        self.emit.emit(cmsp, msg, lvl);
    }
    pub fn custom_emit(&self, cm: &codemap::CodeMap,
                       sp: Span, msg: &str, lvl: Level) {
        self.emit.custom_emit(cm, sp, msg, lvl);
    }
}

pub fn ice_msg(msg: &str) -> ~str {
    format!("internal compiler error: {}\nThis message reflects a bug in the Rust compiler. \
            \nWe would appreciate a bug report: {}", msg, BUG_REPORT_URL)
}

pub fn mk_span_handler(handler: @Handler, cm: @codemap::CodeMap)
                       -> @SpanHandler {
    @SpanHandler {
        handler: handler,
        cm: cm,
    }
}

pub fn mk_handler() -> @Handler {
    @Handler {
        err_count: Cell::new(0),
        emit: DefaultEmitter,
    }
}

#[deriving(Eq)]
pub enum Level {
    Fatal,
    Error,
    Warning,
    Note,
}

impl ToStr for Level {
    fn to_str(&self) -> ~str {
        match *self {
            Fatal | Error => ~"error",
            Warning => ~"warning",
            Note => ~"note"
        }
    }
}

impl Level {
    fn color(self) -> term::color::Color {
        match self {
            Fatal | Error => term::color::BRIGHT_RED,
            Warning => term::color::BRIGHT_YELLOW,
            Note => term::color::BRIGHT_GREEN
        }
    }
}

fn print_maybe_styled(msg: &str, color: term::attr::Attr) -> io::IoResult<()> {
    local_data_key!(tls_terminal: Option<term::Terminal<StdWriter>>)


    fn is_stderr_screen() -> bool {
        use std::libc;
        unsafe { libc::isatty(libc::STDERR_FILENO) != 0 }
    }
    fn write_pretty<T: Writer>(term: &mut term::Terminal<T>, s: &str,
                               c: term::attr::Attr) -> io::IoResult<()> {
        try!(term.attr(c));
        try!(term.write(s.as_bytes()));
        try!(term.reset());
        Ok(())
    }

    if is_stderr_screen() {
        local_data::get_mut(tls_terminal, |term| {
            match term {
                Some(term) => {
                    match *term {
                        Some(ref mut term) => write_pretty(term, msg, color),
                        None => io::stderr().write(msg.as_bytes())
                    }
                }
                None => {
                    let (t, ret) = match term::Terminal::new(io::stderr()) {
                        Ok(mut term) => {
                            let r = write_pretty(&mut term, msg, color);
                            (Some(term), r)
                        }
                        Err(_) => {
                            (None, io::stderr().write(msg.as_bytes()))
                        }
                    };
                    local_data::set(tls_terminal, t);
                    ret
                }
            }
        })
    } else {
        io::stderr().write(msg.as_bytes())
    }
}

fn print_diagnostic(topic: &str, lvl: Level, msg: &str) -> io::IoResult<()> {
    if !topic.is_empty() {
        let mut stderr = io::stderr();
        try!(write!(&mut stderr as &mut io::Writer, "{} ", topic));
    }

    try!(print_maybe_styled(format!("{}: ", lvl.to_str()),
                              term::attr::ForegroundColor(lvl.color())));
    try!(print_maybe_styled(format!("{}\n", msg), term::attr::Bold));
    Ok(())
}

pub struct DefaultEmitter;

impl Emitter for DefaultEmitter {
    fn emit(&self,
            cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str,
            lvl: Level) {
        let error = match cmsp {
            Some((cm, sp)) => emit(cm, sp, msg, lvl, false),
            None => print_diagnostic("", lvl, msg),
        };

        match error {
            Ok(()) => {}
            Err(e) => fail!("failed to print diagnostics: {}", e),
        }
    }

    fn custom_emit(&self, cm: &codemap::CodeMap,
                   sp: Span, msg: &str, lvl: Level) {
        match emit(cm, sp, msg, lvl, true) {
            Ok(()) => {}
            Err(e) => fail!("failed to print diagnostics: {}", e),
        }
    }
}

fn emit(cm: &codemap::CodeMap, sp: Span,
        msg: &str, lvl: Level, custom: bool) -> io::IoResult<()> {
    let ss = cm.span_to_str(sp);
    let lines = cm.span_to_lines(sp);
    if custom {
        // we want to tell compiletest/runtest to look at the last line of the
        // span (since `custom_highlight_lines` displays an arrow to the end of
        // the span)
        let span_end = Span { lo: sp.hi, hi: sp.hi, expn_info: sp.expn_info};
        let ses = cm.span_to_str(span_end);
        try!(print_diagnostic(ses, lvl, msg));
        try!(custom_highlight_lines(cm, sp, lvl, lines));
    } else {
        try!(print_diagnostic(ss, lvl, msg));
        try!(highlight_lines(cm, sp, lvl, lines));
    }
    print_macro_backtrace(cm, sp)
}

fn highlight_lines(cm: &codemap::CodeMap,
                   sp: Span,
                   lvl: Level,
                   lines: &codemap::FileLines) -> io::IoResult<()> {
    let fm = lines.file;
    let mut err = io::stderr();
    let err = &mut err as &mut io::Writer;

    let mut elided = false;
    let mut display_lines = lines.lines.as_slice();
    if display_lines.len() > MAX_LINES {
        display_lines = display_lines.slice(0u, MAX_LINES);
        elided = true;
    }
    // Print the offending lines
    for line in display_lines.iter() {
        try!(write!(err, "{}:{} {}\n", fm.name, *line + 1,
                      fm.get_line(*line as int)));
    }
    if elided {
        let last_line = display_lines[display_lines.len() - 1u];
        let s = format!("{}:{} ", fm.name, last_line + 1u);
        try!(write!(err, "{0:1$}...\n", "", s.len()));
    }

    // FIXME (#3260)
    // If there's one line at fault we can easily point to the problem
    if lines.lines.len() == 1u {
        let lo = cm.lookup_char_pos(sp.lo);
        let mut digits = 0u;
        let mut num = (lines.lines[0] + 1u) / 10u;

        // how many digits must be indent past?
        while num > 0u { num /= 10u; digits += 1u; }

        // indent past |name:## | and the 0-offset column location
        let left = fm.name.len() + digits + lo.col.to_uint() + 3u;
        let mut s = ~"";
        // Skip is the number of characters we need to skip because they are
        // part of the 'filename:line ' part of the previous line.
        let skip = fm.name.len() + digits + 3u;
        for _ in range(0, skip) { s.push_char(' '); }
        let orig = fm.get_line(lines.lines[0] as int);
        for pos in range(0u, left-skip) {
            let curChar = (orig[pos] as char);
            // Whenever a tab occurs on the previous line, we insert one on
            // the error-point-squiggly-line as well (instead of a space).
            // That way the squiggly line will usually appear in the correct
            // position.
            match curChar {
                '\t' => s.push_char('\t'),
                _ => s.push_char(' '),
            };
        }
        try!(write!(err, "{}", s));
        let mut s = ~"^";
        let hi = cm.lookup_char_pos(sp.hi);
        if hi.col != lo.col {
            // the ^ already takes up one space
            let num_squigglies = hi.col.to_uint()-lo.col.to_uint()-1u;
            for _ in range(0, num_squigglies) { s.push_char('~'); }
        }
        try!(print_maybe_styled(s + "\n",
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
fn custom_highlight_lines(cm: &codemap::CodeMap,
                          sp: Span,
                          lvl: Level,
                          lines: &codemap::FileLines) -> io::IoResult<()> {
    let fm = lines.file;
    let mut err = io::stderr();
    let err = &mut err as &mut io::Writer;

    let lines = lines.lines.as_slice();
    if lines.len() > MAX_LINES {
        try!(write!(err, "{}:{} {}\n", fm.name,
                      lines[0] + 1, fm.get_line(lines[0] as int)));
        try!(write!(err, "...\n"));
        let last_line = lines[lines.len()-1];
        try!(write!(err, "{}:{} {}\n", fm.name,
                      last_line + 1, fm.get_line(last_line as int)));
    } else {
        for line in lines.iter() {
            try!(write!(err, "{}:{} {}\n", fm.name,
                          *line + 1, fm.get_line(*line as int)));
        }
    }
    let last_line_start = format!("{}:{} ", fm.name, lines[lines.len()-1]+1);
    let hi = cm.lookup_char_pos(sp.hi);
    // Span seems to use half-opened interval, so subtract 1
    let skip = last_line_start.len() + hi.col.to_uint() - 1;
    let mut s = ~"";
    for _ in range(0, skip) { s.push_char(' '); }
    s.push_char('^');
    print_maybe_styled(s + "\n", term::attr::ForegroundColor(lvl.color()))
}

fn print_macro_backtrace(cm: &codemap::CodeMap, sp: Span) -> io::IoResult<()> {
    for ei in sp.expn_info.iter() {
        let ss = ei.callee.span.as_ref().map_or(~"", |span| cm.span_to_str(*span));
        let (pre, post) = match ei.callee.format {
            codemap::MacroAttribute => ("#[", "]"),
            codemap::MacroBang => ("", "!")
        };
        try!(print_diagnostic(ss, Note,
                                format!("in expansion of {}{}{}", pre,
                                        ei.callee.name, post)));
        let ss = cm.span_to_str(ei.call_site);
        try!(print_diagnostic(ss, Note, "expansion site"));
        try!(print_macro_backtrace(cm, ei.call_site));
    }
    Ok(())
}

pub fn expect<T:Clone>(diag: @SpanHandler, opt: Option<T>, msg: || -> ~str)
              -> T {
    match opt {
       Some(ref t) => (*t).clone(),
       None => diag.handler().bug(msg()),
    }
}
