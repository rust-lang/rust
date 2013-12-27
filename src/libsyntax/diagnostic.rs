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
use std::local_data;
use extra::term;

static BUG_REPORT_URL: &'static str =
    "https://github.com/mozilla/rust/wiki/HOWTO-submit-a-Rust-bug-report";

pub trait Emitter {
    fn emit(&self,
            cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str,
            lvl: level);
}

// a span-handler is like a handler but also
// accepts span information for source-location
// reporting.
pub struct SpanHandler {
    handler: @Handler,
    cm: @codemap::CodeMap,
}

impl SpanHandler {
    pub fn span_fatal(@self, sp: Span, msg: &str) -> ! {
        self.handler.emit(Some((&*self.cm, sp)), msg, fatal);
        fail!();
    }
    pub fn span_err(@self, sp: Span, msg: &str) {
        self.handler.emit(Some((&*self.cm, sp)), msg, error);
        self.handler.bump_err_count();
    }
    pub fn span_warn(@self, sp: Span, msg: &str) {
        self.handler.emit(Some((&*self.cm, sp)), msg, warning);
    }
    pub fn span_note(@self, sp: Span, msg: &str) {
        self.handler.emit(Some((&*self.cm, sp)), msg, note);
    }
    pub fn span_bug(@self, sp: Span, msg: &str) -> ! {
        self.span_fatal(sp, ice_msg(msg));
    }
    pub fn span_unimpl(@self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp, ~"unimplemented " + msg);
    }
    pub fn handler(@self) -> @Handler {
        self.handler
    }
}

// a handler deals with errors; certain errors
// (fatal, bug, unimpl) may cause immediate exit,
// others log errors for later reporting.
pub struct Handler {
    err_count: Cell<uint>,
    emit: @Emitter,
}

impl Handler {
    pub fn fatal(@self, msg: &str) -> ! {
        self.emit.emit(None, msg, fatal);
        fail!();
    }
    pub fn err(@self, msg: &str) {
        self.emit.emit(None, msg, error);
        self.bump_err_count();
    }
    pub fn bump_err_count(@self) {
        self.err_count.set(self.err_count.get() + 1u);
    }
    pub fn err_count(@self) -> uint {
        self.err_count.get()
    }
    pub fn has_errors(@self) -> bool {
        self.err_count.get()> 0u
    }
    pub fn abort_if_errors(@self) {
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
    pub fn warn(@self, msg: &str) {
        self.emit.emit(None, msg, warning);
    }
    pub fn note(@self, msg: &str) {
        self.emit.emit(None, msg, note);
    }
    pub fn bug(@self, msg: &str) -> ! {
        self.fatal(ice_msg(msg));
    }
    pub fn unimpl(@self, msg: &str) -> ! {
        self.bug(~"unimplemented " + msg);
    }
    pub fn emit(@self,
            cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str,
            lvl: level) {
        self.emit.emit(cmsp, msg, lvl);
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

pub fn mk_handler(emitter: Option<@Emitter>) -> @Handler {
    let emit: @Emitter = match emitter {
        Some(e) => e,
        None => @DefaultEmitter as @Emitter
    };

    @Handler {
        err_count: Cell::new(0),
        emit: emit,
    }
}

#[deriving(Eq)]
pub enum level {
    fatal,
    error,
    warning,
    note,
}

fn diagnosticstr(lvl: level) -> ~str {
    match lvl {
        fatal => ~"error",
        error => ~"error",
        warning => ~"warning",
        note => ~"note"
    }
}

fn diagnosticcolor(lvl: level) -> term::color::Color {
    match lvl {
        fatal => term::color::BRIGHT_RED,
        error => term::color::BRIGHT_RED,
        warning => term::color::BRIGHT_YELLOW,
        note => term::color::BRIGHT_GREEN
    }
}

fn print_maybe_styled(msg: &str, color: term::attr::Attr) {
    local_data_key!(tls_terminal: ~Option<term::Terminal<StdWriter>>)

    fn is_stderr_screen() -> bool {
        use std::libc;
        unsafe { libc::isatty(libc::STDERR_FILENO) != 0 }
    }
    fn write_pretty<T: Writer>(term: &mut term::Terminal<T>, s: &str, c: term::attr::Attr) {
        term.attr(c);
        term.write(s.as_bytes());
        term.reset();
    }

    if is_stderr_screen() {
        local_data::get_mut(tls_terminal, |term| {
            match term {
                Some(term) => {
                    match **term {
                        Some(ref mut term) => write_pretty(term, msg, color),
                        None => io::stderr().write(msg.as_bytes())
                    }
                }
                None => {
                    let t = ~match term::Terminal::new(io::stderr()) {
                        Ok(mut term) => {
                            write_pretty(&mut term, msg, color);
                            Some(term)
                        }
                        Err(_) => {
                            io::stderr().write(msg.as_bytes());
                            None
                        }
                    };
                    local_data::set(tls_terminal, t);
                }
            }
        });
    } else {
        io::stderr().write(msg.as_bytes());
    }
}

fn print_diagnostic(topic: &str, lvl: level, msg: &str) {
    let mut stderr = io::stderr();

    if !topic.is_empty() {
        write!(&mut stderr as &mut io::Writer, "{} ", topic);
    }

    print_maybe_styled(format!("{}: ", diagnosticstr(lvl)),
                            term::attr::ForegroundColor(diagnosticcolor(lvl)));
    print_maybe_styled(format!("{}\n", msg), term::attr::Bold);
}

pub struct DefaultEmitter;

impl Emitter for DefaultEmitter {
    fn emit(&self,
            cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str,
            lvl: level) {
        match cmsp {
            Some((cm, sp)) => {
                let sp = cm.adjust_span(sp);
                let ss = cm.span_to_str(sp);
                let lines = cm.span_to_lines(sp);
                print_diagnostic(ss, lvl, msg);
                highlight_lines(cm, sp, lvl, lines);
                print_macro_backtrace(cm, sp);
            }
            None => print_diagnostic("", lvl, msg),
        }
    }
}

fn highlight_lines(cm: &codemap::CodeMap,
                   sp: Span,
                   lvl: level,
                   lines: &codemap::FileLines) {
    let fm = lines.file;
    let mut err = io::stderr();
    let err = &mut err as &mut io::Writer;

    // arbitrarily only print up to six lines of the error
    let max_lines = 6u;
    let mut elided = false;
    let mut display_lines = lines.lines.as_slice();
    if display_lines.len() > max_lines {
        display_lines = display_lines.slice(0u, max_lines);
        elided = true;
    }
    // Print the offending lines
    for line in display_lines.iter() {
        write!(err, "{}:{} {}\n", fm.name, *line + 1, fm.get_line(*line as int));
    }
    if elided {
        let last_line = display_lines[display_lines.len() - 1u];
        let s = format!("{}:{} ", fm.name, last_line + 1u);
        write!(err, "{0:1$}...\n", "", s.len());
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
        skip.times(|| s.push_char(' '));
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
        write!(err, "{}", s);
        let mut s = ~"^";
        let hi = cm.lookup_char_pos(sp.hi);
        if hi.col != lo.col {
            // the ^ already takes up one space
            let num_squigglies = hi.col.to_uint()-lo.col.to_uint()-1u;
            num_squigglies.times(|| s.push_char('~'));
        }
        print_maybe_styled(s + "\n", term::attr::ForegroundColor(diagnosticcolor(lvl)));
    }
}

fn print_macro_backtrace(cm: &codemap::CodeMap, sp: Span) {
    for ei in sp.expn_info.iter() {
        let ss = ei.callee.span.as_ref().map_default(~"", |span| cm.span_to_str(*span));
        let (pre, post) = match ei.callee.format {
            codemap::MacroAttribute => ("#[", "]"),
            codemap::MacroBang => ("", "!")
        };

        print_diagnostic(ss, note,
                         format!("in expansion of {}{}{}", pre, ei.callee.name, post));
        let ss = cm.span_to_str(ei.call_site);
        print_diagnostic(ss, note, "expansion site");
        print_macro_backtrace(cm, ei.call_site);
    }
}

pub fn expect<T:Clone>(diag: @SpanHandler, opt: Option<T>, msg: || -> ~str)
              -> T {
    match opt {
       Some(ref t) => (*t).clone(),
       None => diag.handler().bug(msg()),
    }
}
