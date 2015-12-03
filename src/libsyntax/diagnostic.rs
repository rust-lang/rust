// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Level::*;
pub use self::RenderSpan::*;
pub use self::ColorConfig::*;
use self::Destination::*;

use codemap::{self, COMMAND_LINE_SP, COMMAND_LINE_EXPN, Pos, Span};
use diagnostics;

use std::cell::{RefCell, Cell};
use std::{cmp, error, fmt};
use std::io::prelude::*;
use std::io;
use term;

/// maximum number of lines we will print for each error; arbitrary.
const MAX_LINES: usize = 6;

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

#[derive(Clone, Copy)]
pub enum ColorConfig {
    Auto,
    Always,
    Never
}

pub trait Emitter {
    fn emit(&mut self, cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str, code: Option<&str>, lvl: Level);
    fn custom_emit(&mut self, cm: &codemap::CodeMap,
                   sp: RenderSpan, msg: &str, lvl: Level);
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

/// A span-handler is like a handler but also
/// accepts span information for source-location
/// reporting.
pub struct SpanHandler {
    pub handler: Handler,
    pub cm: codemap::CodeMap,
}

impl SpanHandler {
    pub fn new(handler: Handler, cm: codemap::CodeMap) -> SpanHandler {
        SpanHandler {
            handler: handler,
            cm: cm,
        }
    }
    pub fn span_fatal(&self, sp: Span, msg: &str) -> FatalError {
        self.handler.emit(Some((&self.cm, sp)), msg, Fatal);
        return FatalError;
    }
    pub fn span_fatal_with_code(&self, sp: Span, msg: &str, code: &str) -> FatalError {
        self.handler.emit_with_code(Some((&self.cm, sp)), msg, code, Fatal);
        return FatalError;
    }
    pub fn span_err(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&self.cm, sp)), msg, Error);
        self.handler.bump_err_count();
    }
    pub fn span_err_with_code(&self, sp: Span, msg: &str, code: &str) {
        self.handler.emit_with_code(Some((&self.cm, sp)), msg, code, Error);
        self.handler.bump_err_count();
    }
    pub fn span_warn(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&self.cm, sp)), msg, Warning);
    }
    pub fn span_warn_with_code(&self, sp: Span, msg: &str, code: &str) {
        self.handler.emit_with_code(Some((&self.cm, sp)), msg, code, Warning);
    }
    pub fn span_note(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&self.cm, sp)), msg, Note);
    }
    pub fn span_end_note(&self, sp: Span, msg: &str) {
        self.handler.custom_emit(&self.cm, EndSpan(sp), msg, Note);
    }
    pub fn span_help(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&self.cm, sp)), msg, Help);
    }
    /// Prints out a message with a suggested edit of the code.
    ///
    /// See `diagnostic::RenderSpan::Suggestion` for more information.
    pub fn span_suggestion(&self, sp: Span, msg: &str, suggestion: String) {
        self.handler.custom_emit(&self.cm, Suggestion(sp, suggestion), msg, Help);
    }
    pub fn fileline_note(&self, sp: Span, msg: &str) {
        self.handler.custom_emit(&self.cm, FileLine(sp), msg, Note);
    }
    pub fn fileline_help(&self, sp: Span, msg: &str) {
        self.handler.custom_emit(&self.cm, FileLine(sp), msg, Help);
    }
    pub fn span_bug(&self, sp: Span, msg: &str) -> ! {
        self.handler.emit(Some((&self.cm, sp)), msg, Bug);
        panic!(ExplicitBug);
    }
    pub fn span_bug_no_panic(&self, sp: Span, msg: &str) {
        self.handler.emit(Some((&self.cm, sp)), msg, Bug);
        self.handler.bump_err_count();
    }
    pub fn span_unimpl(&self, sp: Span, msg: &str) -> ! {
        self.span_bug(sp, &format!("unimplemented {}", msg));
    }
    pub fn handler<'a>(&'a self) -> &'a Handler {
        &self.handler
    }
}

/// A handler deals with errors; certain errors
/// (fatal, bug, unimpl) may cause immediate exit,
/// others log errors for later reporting.
pub struct Handler {
    err_count: Cell<usize>,
    emit: RefCell<Box<Emitter + Send>>,
    pub can_emit_warnings: bool
}

impl Handler {
    pub fn new(color_config: ColorConfig,
               registry: Option<diagnostics::registry::Registry>,
               can_emit_warnings: bool) -> Handler {
        let emitter = Box::new(EmitterWriter::stderr(color_config, registry));
        Handler::with_emitter(can_emit_warnings, emitter)
    }
    pub fn with_emitter(can_emit_warnings: bool, e: Box<Emitter + Send>) -> Handler {
        Handler {
            err_count: Cell::new(0),
            emit: RefCell::new(e),
            can_emit_warnings: can_emit_warnings
        }
    }
    pub fn fatal(&self, msg: &str) -> FatalError {
        self.emit.borrow_mut().emit(None, msg, None, Fatal);
        FatalError
    }
    pub fn err(&self, msg: &str) {
        self.emit.borrow_mut().emit(None, msg, None, Error);
        self.bump_err_count();
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
            0 => return,
            1 => s = "aborting due to previous error".to_string(),
            _  => {
                s = format!("aborting due to {} previous errors",
                            self.err_count.get());
            }
        }

        panic!(self.fatal(&s[..]));
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
    pub fn emit(&self,
                cmsp: Option<(&codemap::CodeMap, Span)>,
                msg: &str,
                lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().emit(cmsp, msg, None, lvl);
    }
    pub fn emit_with_code(&self,
                          cmsp: Option<(&codemap::CodeMap, Span)>,
                          msg: &str,
                          code: &str,
                          lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().emit(cmsp, msg, Some(code), lvl);
    }
    pub fn custom_emit(&self, cm: &codemap::CodeMap,
                       sp: RenderSpan, msg: &str, lvl: Level) {
        if lvl == Warning && !self.can_emit_warnings { return }
        self.emit.borrow_mut().custom_emit(cm, sp, msg, lvl);
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

pub struct EmitterWriter {
    dst: Destination,
    registry: Option<diagnostics::registry::Registry>
}

enum Destination {
    Terminal(Box<term::StderrTerminal>),
    Raw(Box<Write + Send>),
}

/// Do not use this for messages that end in `\n` – use `println_maybe_styled` instead. See
/// `EmitterWriter::print_maybe_styled` for details.
macro_rules! print_maybe_styled {
    ($writer: expr, $style: expr, $($arg: tt)*) => {
        $writer.print_maybe_styled(format_args!($($arg)*), $style, false)
    }
}

macro_rules! println_maybe_styled {
    ($writer: expr, $style: expr, $($arg: tt)*) => {
        $writer.print_maybe_styled(format_args!($($arg)*), $style, true)
    }
}

impl EmitterWriter {
    pub fn stderr(color_config: ColorConfig,
                  registry: Option<diagnostics::registry::Registry>) -> EmitterWriter {
        let stderr = io::stderr();

        let use_color = match color_config {
            Always => true,
            Never  => false,
            Auto   => stderr_isatty(),
        };

        if use_color {
            let dst = match term::stderr() {
                Some(t) => Terminal(t),
                None    => Raw(Box::new(stderr)),
            };
            EmitterWriter { dst: dst, registry: registry }
        } else {
            EmitterWriter { dst: Raw(Box::new(stderr)), registry: registry }
        }
    }

    pub fn new(dst: Box<Write + Send>,
               registry: Option<diagnostics::registry::Registry>) -> EmitterWriter {
        EmitterWriter { dst: Raw(dst), registry: registry }
    }

    fn print_maybe_styled(&mut self,
                          args: fmt::Arguments,
                          color: term::Attr,
                          print_newline_at_end: bool) -> io::Result<()> {
        match self.dst {
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

    fn print_diagnostic(&mut self, topic: &str, lvl: Level,
                        msg: &str, code: Option<&str>) -> io::Result<()> {
        if !topic.is_empty() {
            try!(write!(&mut self.dst, "{} ", topic));
        }

        try!(print_maybe_styled!(self, term::Attr::ForegroundColor(lvl.color()),
                                 "{}: ", lvl.to_string()));
        try!(print_maybe_styled!(self, term::Attr::Bold, "{}", msg));

        match code {
            Some(code) => {
                let style = term::Attr::ForegroundColor(term::color::BRIGHT_MAGENTA);
                try!(print_maybe_styled!(self, style, " [{}]", code.clone()));
            }
            None => ()
        }
        try!(write!(&mut self.dst, "\n"));
        Ok(())
    }

    fn emit_(&mut self, cm: &codemap::CodeMap, rsp: RenderSpan,
             msg: &str, code: Option<&str>, lvl: Level) -> io::Result<()> {
        let sp = rsp.span();

        // We cannot check equality directly with COMMAND_LINE_SP
        // since PartialEq is manually implemented to ignore the ExpnId
        let ss = if sp.expn_id == COMMAND_LINE_EXPN {
            "<command line option>".to_string()
        } else if let EndSpan(_) = rsp {
            let span_end = Span { lo: sp.hi, hi: sp.hi, expn_id: sp.expn_id};
            cm.span_to_string(span_end)
        } else {
            cm.span_to_string(sp)
        };

        try!(self.print_diagnostic(&ss[..], lvl, msg, code));

        match rsp {
            FullSpan(_) => {
                try!(self.highlight_lines(cm, sp, lvl, cm.span_to_lines(sp)));
                try!(self.print_macro_backtrace(cm, sp));
            }
            EndSpan(_) => {
                try!(self.end_highlight_lines(cm, sp, lvl, cm.span_to_lines(sp)));
                try!(self.print_macro_backtrace(cm, sp));
            }
            Suggestion(_, ref suggestion) => {
                try!(self.highlight_suggestion(cm, sp, suggestion));
                try!(self.print_macro_backtrace(cm, sp));
            }
            FileLine(..) => {
                // no source text in this case!
            }
        }

        match code {
            Some(code) =>
                match self.registry.as_ref().and_then(|registry| registry.find_description(code)) {
                    Some(_) => {
                        try!(self.print_diagnostic(&ss[..], Help,
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
                            cm: &codemap::CodeMap,
                            sp: Span,
                            suggestion: &str)
                            -> io::Result<()>
    {
        let lines = cm.span_to_lines(sp).unwrap();
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
                       cm: &codemap::CodeMap,
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
            let lo = cm.lookup_char_pos(sp.lo);
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

                let hi = cm.lookup_char_pos(sp.hi);
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

                try!(println_maybe_styled!(self, term::Attr::ForegroundColor(lvl.color()),
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
                           cm: &codemap::CodeMap,
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
        let hi = cm.lookup_char_pos(sp.hi);
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
        println_maybe_styled!(self, term::Attr::ForegroundColor(lvl.color()),
                              "{}", s)
    }

    fn print_macro_backtrace(&mut self,
                             cm: &codemap::CodeMap,
                             sp: Span)
                             -> io::Result<()> {
        let mut last_span = codemap::DUMMY_SP;
        let mut sp_opt = Some(sp);

        while let Some(sp) = sp_opt {
            sp_opt = try!(cm.with_expn_info(sp.expn_id, |expn_info| -> io::Result<_> {
                match expn_info {
                    Some(ei) => {
                        let (pre, post) = match ei.callee.format {
                            codemap::MacroAttribute(..) => ("#[", "]"),
                            codemap::MacroBang(..) => ("", "!"),
                        };
                        // Don't print recursive invocations
                        if ei.call_site != last_span {
                            last_span = ei.call_site;

                            let mut diag_string = format!("in this expansion of {}{}{}",
                                                          pre,
                                                          ei.callee.name(),
                                                          post);

                            if let Some(def_site_span) = ei.callee.span {
                                diag_string.push_str(&format!(" (defined in {})",
                                                              cm.span_to_filename(def_site_span)));
                            }

                            try!(self.print_diagnostic(&cm.span_to_string(ei.call_site),
                                                       Note,
                                                       &diag_string,
                                                       None));
                        }
                        Ok(Some(ei.call_site))
                    }
                    None => Ok(None)
                }
            }));
        }

        Ok(())
    }
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

impl Emitter for EmitterWriter {
    fn emit(&mut self,
            cmsp: Option<(&codemap::CodeMap, Span)>,
            msg: &str, code: Option<&str>, lvl: Level) {
        let error = match cmsp {
            Some((cm, COMMAND_LINE_SP)) => self.emit_(cm,
                                                FileLine(COMMAND_LINE_SP),
                                                msg, code, lvl),
            Some((cm, sp)) => self.emit_(cm, FullSpan(sp), msg, code, lvl),
            None => self.print_diagnostic("", lvl, msg, code),
        };

        match error {
            Ok(()) => {}
            Err(e) => panic!("failed to print diagnostics: {:?}", e),
        }
    }

    fn custom_emit(&mut self, cm: &codemap::CodeMap,
                   sp: RenderSpan, msg: &str, lvl: Level) {
        match self.emit_(cm, sp, msg, None, lvl) {
            Ok(()) => {}
            Err(e) => panic!("failed to print diagnostics: {:?}", e),
        }
    }
}

pub fn expect<T, M>(diag: &SpanHandler, opt: Option<T>, msg: M) -> T where
    M: FnOnce() -> String,
{
    match opt {
        Some(t) => t,
        None => diag.handler().bug(&msg()),
    }
}

#[cfg(test)]
mod test {
    use super::{EmitterWriter, Level};
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
