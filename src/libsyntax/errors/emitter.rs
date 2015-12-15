// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::Destination::*;

use codemap::{self, COMMAND_LINE_SP, COMMAND_LINE_EXPN, Pos, Span};
use diagnostics;

use errors::{Level, RenderSpan};
use errors::RenderSpan::*;
use errors::Level::*;

use std::{cmp, fmt};
use std::io::prelude::*;
use std::io;
use std::rc::Rc;
use term;


pub trait Emitter {
    fn emit(&mut self, span: Option<Span>, msg: &str, code: Option<&str>, lvl: Level);
    fn custom_emit(&mut self, sp: RenderSpan, msg: &str, lvl: Level);
}

/// maximum number of lines we will print for each error; arbitrary.
const MAX_LINES: usize = 6;

#[derive(Clone, Copy)]
pub enum ColorConfig {
    Auto,
    Always,
    Never,
}

impl ColorConfig {
    fn use_color(&self) -> bool {
        match *self {
            ColorConfig::Always => true,
            ColorConfig::Never  => false,
            ColorConfig::Auto   => stderr_isatty(),
        }        
    }
}

// A basic emitter for when we don't have access to a codemap or registry. Used
// for reporting very early errors, etc.
pub struct BasicEmitter {
    dst: Destination,
}

impl Emitter for BasicEmitter {
    fn emit(&mut self,
            sp: Option<Span>,
            msg: &str,
            code: Option<&str>,
            lvl: Level) {
        assert!(sp.is_none(), "BasicEmitter can't handle spans");
        if let Err(e) = print_diagnostic(&mut self.dst, "", lvl, msg, code) {
            panic!("failed to print diagnostics: {:?}", e);
        }

    }

    fn custom_emit(&mut self, _: RenderSpan, _: &str, _: Level) {
        panic!("BasicEmitter can't handle custom_emit");
    }
}

impl BasicEmitter {
    pub fn stderr(color_config: ColorConfig) -> BasicEmitter {
        if color_config.use_color() {
            let dst = Destination::from_stderr();
            BasicEmitter { dst: dst }
        } else {
            BasicEmitter { dst: Raw(Box::new(io::stderr())) }
        }
    }
}

pub struct EmitterWriter {
    dst: Destination,
    registry: Option<diagnostics::registry::Registry>,
    cm: Rc<codemap::CodeMap>,
}

impl Emitter for EmitterWriter {
    fn emit(&mut self,
            sp: Option<Span>,
            msg: &str,
            code: Option<&str>,
            lvl: Level) {
        let error = match sp {
            Some(COMMAND_LINE_SP) => self.emit_(FileLine(COMMAND_LINE_SP), msg, code, lvl),
            Some(sp) => self.emit_(FullSpan(sp), msg, code, lvl),
            None => print_diagnostic(&mut self.dst, "", lvl, msg, code),
        };

        if let Err(e) = error {
            panic!("failed to print diagnostics: {:?}", e);
        }
    }

    fn custom_emit(&mut self,
                   sp: RenderSpan,
                   msg: &str,
                   lvl: Level) {
        match self.emit_(sp, msg, None, lvl) {
            Ok(()) => {}
            Err(e) => panic!("failed to print diagnostics: {:?}", e),
        }
    }
}

/// Do not use this for messages that end in `\n` – use `println_maybe_styled` instead. See
/// `EmitterWriter::print_maybe_styled` for details.
macro_rules! print_maybe_styled {
    ($dst: expr, $style: expr, $($arg: tt)*) => {
        $dst.print_maybe_styled(format_args!($($arg)*), $style, false)
    }
}

macro_rules! println_maybe_styled {
    ($dst: expr, $style: expr, $($arg: tt)*) => {
        $dst.print_maybe_styled(format_args!($($arg)*), $style, true)
    }
}

impl EmitterWriter {
    pub fn stderr(color_config: ColorConfig,
                  registry: Option<diagnostics::registry::Registry>,
                  code_map: Rc<codemap::CodeMap>)
                  -> EmitterWriter {
        if color_config.use_color() {
            let dst = Destination::from_stderr();
            EmitterWriter { dst: dst, registry: registry, cm: code_map }
        } else {
            EmitterWriter { dst: Raw(Box::new(io::stderr())), registry: registry, cm: code_map }
        }
    }

    pub fn new(dst: Box<Write + Send>,
               registry: Option<diagnostics::registry::Registry>,
               code_map: Rc<codemap::CodeMap>)
               -> EmitterWriter {
        EmitterWriter { dst: Raw(dst), registry: registry, cm: code_map }
    }

    fn emit_(&mut self,
             rsp: RenderSpan,
             msg: &str,
             code: Option<&str>,
             lvl: Level)
             -> io::Result<()> {
        let sp = rsp.span();

        // We cannot check equality directly with COMMAND_LINE_SP
        // since PartialEq is manually implemented to ignore the ExpnId
        let ss = if sp.expn_id == COMMAND_LINE_EXPN {
            "<command line option>".to_string()
        } else if let EndSpan(_) = rsp {
            let span_end = Span { lo: sp.hi, hi: sp.hi, expn_id: sp.expn_id};
            self.cm.span_to_string(span_end)
        } else {
            self.cm.span_to_string(sp)
        };

        try!(print_diagnostic(&mut self.dst, &ss[..], lvl, msg, code));

        match rsp {
            FullSpan(_) => {
                let lines = self.cm.span_to_lines(sp);
                try!(self.highlight_lines(sp, lvl, lines));
                try!(self.print_macro_backtrace(sp));
            }
            EndSpan(_) => {
                let lines = self.cm.span_to_lines(sp);
                try!(self.end_highlight_lines(sp, lvl, lines));
                try!(self.print_macro_backtrace(sp));
            }
            Suggestion(_, ref suggestion) => {
                try!(self.highlight_suggestion(sp, suggestion));
                try!(self.print_macro_backtrace(sp));
            }
            FileLine(..) => {
                // no source text in this case!
            }
        }

        match code {
            Some(code) =>
                match self.registry.as_ref().and_then(|registry| registry.find_description(code)) {
                    Some(_) => {
                        try!(print_diagnostic(&mut self.dst, &ss[..], Help,
                                              &format!("run `rustc --explain {}` to see a \
                                                       detailed explanation", code), None));
                    }
                    None => ()
                },
            None => (),
        }
        Ok(())
    }

    fn highlight_suggestion(&mut self,
                            sp: Span,
                            suggestion: &str)
                            -> io::Result<()>
    {
        let lines = self.cm.span_to_lines(sp).unwrap();
        assert!(!lines.lines.is_empty());

        // To build up the result, we want to take the snippet from the first
        // line that precedes the span, prepend that with the suggestion, and
        // then append the snippet from the last line that trails the span.
        let fm = &lines.file;

        let first_line = &lines.lines[0];
        let prefix = fm.get_line(first_line.line_index)
                       .map(|l| &l[..first_line.start_col.0])
                       .unwrap_or("");

        let last_line = lines.lines.last().unwrap();
        let suffix = fm.get_line(last_line.line_index)
                       .map(|l| &l[last_line.end_col.0..])
                       .unwrap_or("");

        let complete = format!("{}{}{}", prefix, suggestion, suffix);

        // print the suggestion without any line numbers, but leave
        // space for them. This helps with lining up with previous
        // snippets from the actual error being reported.
        let fm = &*lines.file;
        let mut lines = complete.lines();
        for (line, line_index) in lines.by_ref().take(MAX_LINES).zip(first_line.line_index..) {
            let elided_line_num = format!("{}", line_index+1);
            try!(write!(&mut self.dst, "{0}:{1:2$} {3}\n",
                        fm.name, "", elided_line_num.len(), line));
        }

        // if we elided some lines, add an ellipsis
        if lines.next().is_some() {
            let elided_line_num = format!("{}", first_line.line_index + MAX_LINES + 1);
            try!(write!(&mut self.dst, "{0:1$} {0:2$} ...\n",
                        "", fm.name.len(), elided_line_num.len()));
        }

        Ok(())
    }

    fn highlight_lines(&mut self,
                       sp: Span,
                       lvl: Level,
                       lines: codemap::FileLinesResult)
                       -> io::Result<()>
    {
        let lines = match lines {
            Ok(lines) => lines,
            Err(_) => {
                try!(write!(&mut self.dst, "(internal compiler error: unprintable span)\n"));
                return Ok(());
            }
        };

        let fm = &*lines.file;

        let line_strings: Option<Vec<&str>> =
            lines.lines.iter()
                       .map(|info| fm.get_line(info.line_index))
                       .collect();

        let line_strings = match line_strings {
            None => { return Ok(()); }
            Some(line_strings) => line_strings
        };

        // Display only the first MAX_LINES lines.
        let all_lines = lines.lines.len();
        let display_lines = cmp::min(all_lines, MAX_LINES);
        let display_line_infos = &lines.lines[..display_lines];
        let display_line_strings = &line_strings[..display_lines];

        // Calculate the widest number to format evenly and fix #11715
        assert!(display_line_infos.len() > 0);
        let mut max_line_num = display_line_infos[display_line_infos.len() - 1].line_index + 1;
        let mut digits = 0;
        while max_line_num > 0 {
            max_line_num /= 10;
            digits += 1;
        }

        // Print the offending lines
        for (line_info, line) in display_line_infos.iter().zip(display_line_strings) {
            try!(write!(&mut self.dst, "{}:{:>width$} {}\n",
                        fm.name,
                        line_info.line_index + 1,
                        line,
                        width=digits));
        }

        // If we elided something, put an ellipsis.
        if display_lines < all_lines {
            let last_line_index = display_line_infos.last().unwrap().line_index;
            let s = format!("{}:{} ", fm.name, last_line_index + 1);
            try!(write!(&mut self.dst, "{0:1$}...\n", "", s.len()));
        }

        // FIXME (#3260)
        // If there's one line at fault we can easily point to the problem
        if lines.lines.len() == 1 {
            let lo = self.cm.lookup_char_pos(sp.lo);
            let mut digits = 0;
            let mut num = (lines.lines[0].line_index + 1) / 10;

            // how many digits must be indent past?
            while num > 0 { num /= 10; digits += 1; }

            let mut s = String::new();
            // Skip is the number of characters we need to skip because they are
            // part of the 'filename:line ' part of the previous line.
            let skip = fm.name.chars().count() + digits + 3;
            for _ in 0..skip {
                s.push(' ');
            }
            if let Some(orig) = fm.get_line(lines.lines[0].line_index) {
                let mut col = skip;
                let mut lastc = ' ';
                let mut iter = orig.chars().enumerate();
                for (pos, ch) in iter.by_ref() {
                    lastc = ch;
                    if pos >= lo.col.to_usize() { break; }
                    // Whenever a tab occurs on the previous line, we insert one on
                    // the error-point-squiggly-line as well (instead of a space).
                    // That way the squiggly line will usually appear in the correct
                    // position.
                    match ch {
                        '\t' => {
                            col += 8 - col%8;
                            s.push('\t');
                        },
                        _ => {
                            col += 1;
                            s.push(' ');
                        },
                    }
                }

                try!(write!(&mut self.dst, "{}", s));
                let mut s = String::from("^");
                let count = match lastc {
                    // Most terminals have a tab stop every eight columns by default
                    '\t' => 8 - col%8,
                    _ => 1,
                };
                col += count;
                s.extend(::std::iter::repeat('~').take(count));

                let hi = self.cm.lookup_char_pos(sp.hi);
                if hi.col != lo.col {
                    for (pos, ch) in iter {
                        if pos >= hi.col.to_usize() { break; }
                        let count = match ch {
                            '\t' => 8 - col%8,
                            _ => 1,
                        };
                        col += count;
                        s.extend(::std::iter::repeat('~').take(count));
                    }
                }

                if s.len() > 1 {
                    // One extra squiggly is replaced by a "^"
                    s.pop();
                }

                try!(println_maybe_styled!(&mut self.dst, term::Attr::ForegroundColor(lvl.color()),
                                           "{}", s));
            }
        }
        Ok(())
    }

    /// Here are the differences between this and the normal `highlight_lines`:
    /// `end_highlight_lines` will always put arrow on the last byte of the
    /// span (instead of the first byte). Also, when the span is too long (more
    /// than 6 lines), `end_highlight_lines` will print the first line, then
    /// dot dot dot, then last line, whereas `highlight_lines` prints the first
    /// six lines.
    #[allow(deprecated)]
    fn end_highlight_lines(&mut self,
                           sp: Span,
                           lvl: Level,
                           lines: codemap::FileLinesResult)
                          -> io::Result<()> {
        let lines = match lines {
            Ok(lines) => lines,
            Err(_) => {
                try!(write!(&mut self.dst, "(internal compiler error: unprintable span)\n"));
                return Ok(());
            }
        };

        let fm = &*lines.file;

        let lines = &lines.lines[..];
        if lines.len() > MAX_LINES {
            if let Some(line) = fm.get_line(lines[0].line_index) {
                try!(write!(&mut self.dst, "{}:{} {}\n", fm.name,
                            lines[0].line_index + 1, line));
            }
            try!(write!(&mut self.dst, "...\n"));
            let last_line_index = lines[lines.len() - 1].line_index;
            if let Some(last_line) = fm.get_line(last_line_index) {
                try!(write!(&mut self.dst, "{}:{} {}\n", fm.name,
                            last_line_index + 1, last_line));
            }
        } else {
            for line_info in lines {
                if let Some(line) = fm.get_line(line_info.line_index) {
                    try!(write!(&mut self.dst, "{}:{} {}\n", fm.name,
                                line_info.line_index + 1, line));
                }
            }
        }
        let last_line_start = format!("{}:{} ", fm.name, lines[lines.len()-1].line_index + 1);
        let hi = self.cm.lookup_char_pos(sp.hi);
        let skip = last_line_start.chars().count();
        let mut s = String::new();
        for _ in 0..skip {
            s.push(' ');
        }
        if let Some(orig) = fm.get_line(lines[0].line_index) {
            let iter = orig.chars().enumerate();
            for (pos, ch) in iter {
                // Span seems to use half-opened interval, so subtract 1
                if pos >= hi.col.to_usize() - 1 { break; }
                // Whenever a tab occurs on the previous line, we insert one on
                // the error-point-squiggly-line as well (instead of a space).
                // That way the squiggly line will usually appear in the correct
                // position.
                match ch {
                    '\t' => s.push('\t'),
                    _ => s.push(' '),
                }
            }
        }
        s.push('^');
        println_maybe_styled!(&mut self.dst, term::Attr::ForegroundColor(lvl.color()),
                              "{}", s)
    }

    fn print_macro_backtrace(&mut self,
                             sp: Span)
                             -> io::Result<()> {
        let mut last_span = codemap::DUMMY_SP;
        let mut span = sp;

        loop {
            let span_name_span = self.cm.with_expn_info(span.expn_id, |expn_info| {
                expn_info.map(|ei| {
                    let (pre, post) = match ei.callee.format {
                        codemap::MacroAttribute(..) => ("#[", "]"),
                        codemap::MacroBang(..) => ("", "!"),
                    };
                    let macro_decl_name = format!("in this expansion of {}{}{}",
                                                  pre,
                                                  ei.callee.name(),
                                                  post);
                    let def_site_span = ei.callee.span;
                    (ei.call_site, macro_decl_name, def_site_span)
                })
            });
            let (macro_decl_name, def_site_span) = match span_name_span {
                None => break,
                Some((sp, macro_decl_name, def_site_span)) => {
                    span = sp;
                    (macro_decl_name, def_site_span)
                }
            };

            // Don't print recursive invocations
            if span != last_span {
                let mut diag_string = macro_decl_name;
                if let Some(def_site_span) = def_site_span {
                    diag_string.push_str(&format!(" (defined in {})",
                                                  self.cm.span_to_filename(def_site_span)));
                }

                let snippet = self.cm.span_to_string(span);
                try!(print_diagnostic(&mut self.dst, &snippet, Note, &diag_string, None));
            }
            last_span = span;
        }

        Ok(())
    }
}

fn print_diagnostic(dst: &mut Destination,
                    topic: &str,
                    lvl: Level,
                    msg: &str,
                    code: Option<&str>)
                    -> io::Result<()> {
    if !topic.is_empty() {
        try!(write!(dst, "{} ", topic));
    }

    try!(print_maybe_styled!(dst, term::Attr::ForegroundColor(lvl.color()),
                             "{}: ", lvl.to_string()));
    try!(print_maybe_styled!(dst, term::Attr::Bold, "{}", msg));

    match code {
        Some(code) => {
            let style = term::Attr::ForegroundColor(term::color::BRIGHT_MAGENTA);
            try!(print_maybe_styled!(dst, style, " [{}]", code.clone()));
        }
        None => ()
    }
    try!(write!(dst, "\n"));
    Ok(())
}

#[cfg(unix)]
fn stderr_isatty() -> bool {
    use libc;
    unsafe { libc::isatty(libc::STDERR_FILENO) != 0 }
}
#[cfg(windows)]
fn stderr_isatty() -> bool {
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut u8;
    const STD_ERROR_HANDLE: DWORD = -12i32 as DWORD;
    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE,
                          lpMode: *mut DWORD) -> BOOL;
    }
    unsafe {
        let handle = GetStdHandle(STD_ERROR_HANDLE);
        let mut out = 0;
        GetConsoleMode(handle, &mut out) != 0
    }
}

enum Destination {
    Terminal(Box<term::StderrTerminal>),
    Raw(Box<Write + Send>),
}

impl Destination {
    fn from_stderr() -> Destination {
        match term::stderr() {
            Some(t) => Terminal(t),
            None    => Raw(Box::new(io::stderr())),
        }
    }

    fn print_maybe_styled(&mut self,
                          args: fmt::Arguments,
                          color: term::Attr,
                          print_newline_at_end: bool)
                          -> io::Result<()> {
        match *self {
            Terminal(ref mut t) => {
                try!(t.attr(color));
                // If `msg` ends in a newline, we need to reset the color before
                // the newline. We're making the assumption that we end up writing
                // to a `LineBufferedWriter`, which means that emitting the reset
                // after the newline ends up buffering the reset until we print
                // another line or exit. Buffering the reset is a problem if we're
                // sharing the terminal with any other programs (e.g. other rustc
                // instances via `make -jN`).
                //
                // Note that if `msg` contains any internal newlines, this will
                // result in the `LineBufferedWriter` flushing twice instead of
                // once, which still leaves the opportunity for interleaved output
                // to be miscolored. We assume this is rare enough that we don't
                // have to worry about it.
                try!(t.write_fmt(args));
                try!(t.reset());
                if print_newline_at_end {
                    t.write_all(b"\n")
                } else {
                    Ok(())
                }
            }
            Raw(ref mut w) => {
                try!(w.write_fmt(args));
                if print_newline_at_end {
                    w.write_all(b"\n")
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl Write for Destination {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match *self {
            Terminal(ref mut t) => t.write(bytes),
            Raw(ref mut w) => w.write(bytes),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match *self {
            Terminal(ref mut t) => t.flush(),
            Raw(ref mut w) => w.flush(),
        }
    }
}

