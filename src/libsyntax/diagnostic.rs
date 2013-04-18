// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use codemap::{Pos, span};
use codemap;

use core::io::WriterUtil;
use core::io;
use core::str;
use core::vec;

use std::term;

pub type Emitter = @fn(cmsp: Option<(@codemap::CodeMap, span)>,
                       msg: &str,
                       lvl: level);

// a handler deals with errors; certain errors
// (fatal, bug, unimpl) may cause immediate exit,
// others log errors for later reporting.
pub trait handler {
    fn fatal(@mut self, msg: &str) -> !;
    fn err(@mut self, msg: &str);
    fn bump_err_count(@mut self);
    fn has_errors(@mut self) -> bool;
    fn abort_if_errors(@mut self);
    fn abort_error_print(@mut self);
    fn warn(@mut self, msg: &str);
    fn note(@mut self, msg: &str);
    // used to indicate a bug in the compiler:
    fn bug(@mut self, msg: &str) -> !;
    fn unimpl(@mut self, msg: &str) -> !;
    fn emit(@mut self,
            cmsp: Option<(@codemap::CodeMap, span)>,
            msg: &str,
            lvl: level);
    fn emit_print(@mut self, lvl: level);
}

// a span-handler is like a handler but also
// accepts span information for source-location
// reporting.
pub trait span_handler {
    fn span_fatal(@mut self, sp: span, msg: &str) -> !;
    fn span_err(@mut self, sp: span, msg: &str);
    fn span_warn(@mut self, sp: span, msg: &str);
    fn span_note(@mut self, sp: span, msg: &str);
    fn span_bug(@mut self, sp: span, msg: &str) -> !;
    fn span_unimpl(@mut self, sp: span, msg: &str) -> !;
    fn handler(@mut self) -> @handler;
}

struct emitter_arg {
    cmsp: Option<(@codemap::CodeMap, span)>,
    msg: ~str,
    lvl: level,
}

struct HandlerT {
    err_count: uint,
    emitters: ~[emitter_arg],
    emit: Emitter,
}

struct CodemapT {
    handler: @handler,
    cm: @codemap::CodeMap,
}

impl span_handler for CodemapT {
    fn span_fatal(@mut self, sp: span, msg: &str) -> ! {
        self.handler.bump_err_count();
        self.handler.emit(Some((self.cm, sp)), msg, fatal);
        self.handler.emit_print(error);
        self.handler.emit_print(note);
        self.handler.abort_error_print();
        fail!();
    }
    fn span_err(@mut self, sp: span, msg: &str) {
        self.handler.emit(Some((self.cm, sp)), msg, error);
        self.handler.bump_err_count();
    }
    fn span_warn(@mut self, sp: span, msg: &str) {
        emit_one(Some((self.cm, sp)), msg, warning);
    }
    fn span_note(@mut self, sp: span, msg: &str) {
        self.handler.emit(Some((self.cm, sp)), msg, note);
    }
    fn span_bug(@mut self, sp: span, msg: &str) -> ! {
        self.span_fatal(sp, ice_msg(msg));
    }
    fn span_unimpl(@mut self, sp: span, msg: &str) -> ! {
        self.span_bug(sp, ~"unimplemented " + msg);
    }
    fn handler(@mut self) -> @handler {
        self.handler
    }
}

impl handler for HandlerT {
    fn fatal(@mut self, msg: &str) -> ! {
        self.emit_print(error);
        self.emit_print(note);
        emit_one(None, msg, fatal);
        fail!();
    }
    fn err(@mut self, msg: &str) {
        emit_one(None, msg, error);
        self.bump_err_count();
    }
    fn bump_err_count(@mut self) {
        self.err_count += 1u;
    }
    fn has_errors(@mut self) -> bool { self.err_count > 0u }
    fn abort_if_errors(@mut self) {
        let s;
        match self.err_count {
          0u => return,
          1u => s = ~"aborting due to previous error",
          _  => {
            s = fmt!("aborting due to %u previous errors",
                     self.err_count);
          }
        }
        self.fatal(s);
    }
    fn abort_error_print(@mut self) {
        let s;
        match self.err_count {
          0u => return,
          1u => s = ~"aborting due to previous error",
          _  => {
            s = fmt!("aborting due to %u previous errors",
                     self.err_count);
          }
        }
        emit_one(None, s, fatal);
    }
    fn warn(@mut self, msg: &str) {
        emit_one(None, msg, warning);
    }
    fn note(@mut self, msg: &str) {
        self.emit(None, msg, note);
    }
    fn bug(@mut self, msg: &str) -> ! {
        self.bump_err_count();
        self.fatal(ice_msg(msg));
    }
    fn unimpl(@mut self, msg: &str) -> ! {
        self.bug(~"unimplemented " + msg);
    }
    fn emit(@mut self,
            cmsp: Option<(@codemap::CodeMap, span)>,
            msg: &str,
            lvl: level) {
        (self.emit)(cmsp, msg, lvl);
        let emitter = emitter_arg { cmsp: cmsp,
                                    msg: str::from_slice(msg),
                                    lvl: lvl };
        self.emitters.push(emitter);
    }

    fn emit_print(@mut self, lvl: level) {
        let mut old_cmsp = None;
        let mut old_line = 0u;

        let emitters = self.emitters;
        let length = emitters.len();
        for uint::range(0, length) |i| {
            let emitter = copy self.emitters[i];
            let cmsp = emitter.cmsp;
            let msg = emitter.msg;
            if diagnosticstr(lvl) == diagnosticstr(emitter.lvl) {
                match cmsp {
                    Some((cm, sp)) => {
                            let lo = cm.lookup_char_pos_adj(sp.lo);
                            let sp = cm.adjust_span(sp);
                            let ss = cm.span_to_str(sp);
                            if i == 0 || old_line == lo.line {
                                if old_line == lo.line {
                                    highlight_lines_cmp(cmsp, old_cmsp);
                                }
                                print_diagnostic(ss, lvl, msg);
                                highlight_lines_cmp(old_cmsp, cmsp);
                            } else {
                                highlight_lines(old_cmsp);
                                print_macro_backtrace(old_cmsp);
                                print_diagnostic(ss, lvl, msg);
                            }
                            old_line = lo.line;
                            old_cmsp = emitter.cmsp;
                    }
                    None => {
                        print_diagnostic(~"", lvl, msg);
                    }
                }
            }
        }
        highlight_lines(old_cmsp);
        print_macro_backtrace(old_cmsp);
    }
}

pub fn ice_msg(msg: &str) -> ~str {
    fmt!("internal compiler error: %s", msg)
}

pub fn mk_span_handler(handler: @handler, cm: @codemap::CodeMap)
                    -> @span_handler {
    @mut CodemapT { handler: handler, cm: cm } as @span_handler
}

pub fn mk_handler(emitter: Option<Emitter>) -> @handler {
    let emit: Emitter = match emitter {
        Some(e) => e,
        None => {
            let emit: Emitter = |cmsp, msg, t| emit(cmsp, msg, t);
            emit
        }
    };

    @mut HandlerT { err_count: 0,
                    emitters: ~[],
                    emit: emit } as @handler
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

fn diagnosticcolor(lvl: level) -> u8 {
    match lvl {
        fatal => term::color_bright_red,
        error => term::color_bright_red,
        warning => term::color_bright_yellow,
        note => term::color_bright_green
    }
}

fn print_diagnostic(topic: ~str, lvl: level, msg: &str) {
    let use_color = term::color_supported() &&
        io::stderr().get_type() == io::Screen;
    if !topic.is_empty() {
        io::stderr().write_str(fmt!("%s ", topic));
    }
    if use_color {
        term::fg(io::stderr(), diagnosticcolor(lvl));
    }
    io::stderr().write_str(fmt!("%s:", diagnosticstr(lvl)));
    if use_color {
        term::reset(io::stderr());
    }
    io::stderr().write_str(fmt!(" %s\n", msg));
}

pub fn collect(messages: @mut ~[~str])
            -> @fn(Option<(@codemap::CodeMap, span)>, &str, level) {
    let f: @fn(Option<(@codemap::CodeMap, span)>, &str, level) =
        |_o, msg: &str, _l| { messages.push(msg.to_str()); };
    f
}

pub fn emit(_cmsp: Option<(@codemap::CodeMap, span)>, _msg: &str, _lvl: level) {
    // Nothing to do
}

pub fn emit_one(cmsp: Option<(@codemap::CodeMap, span)>,
        msg: &str, lvl: level) {
    match cmsp {
      Some((cm, sp)) => {
        let sp = cm.adjust_span(sp);
        let ss = cm.span_to_str(sp);
        print_diagnostic(ss, lvl, msg);
        highlight_lines(cmsp);
        print_macro_backtrace(cmsp);
      }
      None => {
        print_diagnostic(~"", lvl, msg);
      }
    }
}

fn highlight_lines_internal(cm: @codemap::CodeMap,
                   sp: span,
                   lines: @codemap::FileLines) {
    let fm = lines.file;

    // arbitrarily only print up to six lines of the error
    let max_lines = 6u;
    let mut elided = false;
    let mut display_lines = /* FIXME (#2543) */ copy lines.lines;
    if vec::len(display_lines) > max_lines {
        display_lines = vec::slice(display_lines, 0u, max_lines).to_vec();
        elided = true;
    }
    // Print the offending lines
    for display_lines.each |line| {
        io::stderr().write_str(fmt!("%s:%u ", fm.name, *line + 1u));
        let s = fm.get_line(*line as int) + ~"\n";
        io::stderr().write_str(s);
    }
    if elided {
        let last_line = display_lines[vec::len(display_lines) - 1u];
        let s = fmt!("%s:%u ", fm.name, last_line + 1u);
        let mut indent = str::len(s);
        let mut out = ~"";
        while indent > 0u { out += ~" "; indent -= 1u; }
        out += ~"...\n";
        io::stderr().write_str(out);
    }

    // FIXME (#3260)
    // If there's one line at fault we can easily point to the problem
    if vec::len(lines.lines) == 1u {
        let lo = cm.lookup_char_pos(sp.lo);
        let mut digits = 0u;
        let mut num = (lines.lines[0] + 1u) / 10u;

        // how many digits must be indent past?
        while num > 0u { num /= 10u; digits += 1u; }

        // indent past |name:## | and the 0-offset column location
        let mut left = str::len(fm.name) + digits + lo.col.to_uint() + 3u;
        let mut s = ~"";
        // Skip is the number of characters we need to skip because they are
        // part of the 'filename:line ' part of the previous line.
        let skip = str::len(fm.name) + digits + 3u;
        for skip.times() {
            s += ~" ";
        }
        let orig = fm.get_line(lines.lines[0] as int);
        for uint::range(0u,left-skip) |pos| {
            let curChar = (orig[pos] as char);
            s += match curChar { // Whenever a tab occurs on the previous
                '\t' => "\t",    // line, we insert one on the error-point-
                _ => " "         // -squigly-line as well (instead of a
            };                   // space). This way the squigly-line will
        }                        // usually appear in the correct position.
        s += ~"^";
        let hi = cm.lookup_char_pos(sp.hi);
        if hi.col != lo.col {
            // the ^ already takes up one space
            let num_squiglies = hi.col.to_uint()-lo.col.to_uint()-1u;
            for num_squiglies.times() { s += ~"~"; }
        }
        io::stderr().write_str(s + ~"\n");
    }
}

fn highlight_lines(cmsp: Option<(@codemap::CodeMap, span)>) {
    match cmsp {
        Some((cm, sp)) => {
            let sp = cm.adjust_span(sp);
            let lines = cm.span_to_lines(sp);
            highlight_lines_internal(cm, sp, lines);
        }
        None => ()
    }
}

fn highlight_lines_cmp(old_cmsp: Option<(@codemap::CodeMap, span)>,
                       cmsp: Option<(@codemap::CodeMap, span)> ){
    let mut old_line = ~[];
    let mut new_line = ~[];
    let mut old_lo = 0u;
    let mut new_lo = 0u;
    let mut flag = true;
    match old_cmsp {
        Some((cm, sp)) => {
            let lo = cm.lookup_char_pos(sp.lo);
            let sp = cm.adjust_span(sp);
            let lines = cm.span_to_lines(sp);
            old_line = lines.lines;
            old_lo = lo.col.to_uint();
        }
        None => { flag = false; }
    }

    match cmsp {
        Some((cm, sp)) => {
            let lo = cm.lookup_char_pos(sp.lo);
            let sp = cm.adjust_span(sp);
            let lines = cm.span_to_lines(sp);
            new_line = lines.lines;
            new_lo = lo.col.to_uint();
        }
        None => { flag = false; }
    }

    if flag {
        if old_line == new_line && old_lo > new_lo {
            highlight_lines(cmsp);
        }
    }
}

fn print_macro_backtrace_internal(cm: @codemap::CodeMap, sp: span) {
    for sp.expn_info.each |ei| {
        let ss = ei.callee.span.map_default(@~"", |span| @cm.span_to_str(*span));
        print_diagnostic(*ss, note,
                         fmt!("in expansion of %s!", ei.callee.name));
        let ss = cm.span_to_str(ei.call_site);
        print_diagnostic(ss, note, ~"expansion site");
        print_macro_backtrace_internal(cm, ei.call_site);
    }
}

fn print_macro_backtrace(cmsp: Option<(@codemap::CodeMap, span)>) {
    match cmsp {
        Some((cm, sp)) => {
            print_macro_backtrace_internal(cm, sp);
        }
        None => ()
    }
}

pub fn expect<T:Copy>(diag: @span_handler,
                       opt: Option<T>,
                       msg: &fn() -> ~str) -> T {
    match opt {
       Some(ref t) => (*t),
       None => diag.handler().bug(msg())
    }
}
