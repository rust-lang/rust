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

use codemap::{self, COMMAND_LINE_SP, DUMMY_SP, Pos, Span, MultiSpan};
use diagnostics;

use errors::{Level, RenderSpan, CodeSuggestion, DiagnosticBuilder};
use errors::RenderSpan::*;
use errors::Level::*;

use std::{cmp, fmt};
use std::io::prelude::*;
use std::io;
use std::rc::Rc;
use term;

pub trait Emitter {
    fn emit(&mut self, span: Option<&MultiSpan>, msg: &str, code: Option<&str>, lvl: Level);
    fn custom_emit(&mut self, sp: &RenderSpan, msg: &str, lvl: Level);

    /// Emit a structured diagnostic.
    fn emit_struct(&mut self, db: &DiagnosticBuilder) {
        self.emit(db.span.as_ref(), &db.message, db.code.as_ref().map(|s| &**s), db.level);
        for child in &db.children {
            match child.render_span {
                Some(ref sp) => self.custom_emit(sp, &child.message, child.level),
                None => self.emit(child.span.as_ref(), &child.message, None, child.level),
            }
        }
    }
}

/// maximum number of lines we will print for each error; arbitrary.
pub const MAX_HIGHLIGHT_LINES: usize = 6;

/// maximum number of lines we will print for each span; arbitrary.
const MAX_SP_LINES: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// A basic emitter for when we don't have access to a codemap or registry. Used
/// for reporting very early errors, etc.
pub struct BasicEmitter {
    dst: Destination,
}

impl Emitter for BasicEmitter {
    fn emit(&mut self,
            msp: Option<&MultiSpan>,
            msg: &str,
            code: Option<&str>,
            lvl: Level) {
        assert!(msp.is_none(), "BasicEmitter can't handle spans");
        if let Err(e) = print_diagnostic(&mut self.dst, "", lvl, msg, code) {
            panic!("failed to print diagnostics: {:?}", e);
        }

    }

    fn custom_emit(&mut self, _: &RenderSpan, _: &str, _: Level) {
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
            msp: Option<&MultiSpan>,
            msg: &str,
            code: Option<&str>,
            lvl: Level) {
        let error = match msp.map(|s|(s.to_span_bounds(), s)) {
            Some((COMMAND_LINE_SP, msp)) => {
                self.emit_(&FileLine(msp.clone()), msg, code, lvl)
            },
            Some((DUMMY_SP, _)) | None => print_diagnostic(&mut self.dst, "", lvl, msg, code),
            Some((_, msp)) => self.emit_(&FullSpan(msp.clone()), msg, code, lvl),
        };

        if let Err(e) = error {
            panic!("failed to print diagnostics: {:?}", e);
        }
    }

    fn custom_emit(&mut self,
                   rsp: &RenderSpan,
                   msg: &str,
                   lvl: Level) {
        if let Err(e) = self.emit_(rsp, msg, None, lvl) {
            panic!("failed to print diagnostics: {:?}", e);
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
             rsp: &RenderSpan,
             msg: &str,
             code: Option<&str>,
             lvl: Level)
             -> io::Result<()> {
        let msp = rsp.span();
        let bounds = msp.to_span_bounds();

        let ss = if bounds == COMMAND_LINE_SP {
            "<command line option>".to_string()
        } else if let EndSpan(_) = *rsp {
            let span_end = Span { lo: bounds.hi, hi: bounds.hi, expn_id: bounds.expn_id};
            self.cm.span_to_string(span_end)
        } else {
            self.cm.span_to_string(bounds)
        };

        print_diagnostic(&mut self.dst, &ss[..], lvl, msg, code)?;

        match *rsp {
            FullSpan(_) => {
                self.highlight_lines(msp, lvl)?;
                self.print_macro_backtrace(bounds)?;
            }
            EndSpan(_) => {
                self.end_highlight_lines(msp, lvl)?;
                self.print_macro_backtrace(bounds)?;
            }
            Suggestion(ref suggestion) => {
                self.highlight_suggestion(suggestion)?;
                self.print_macro_backtrace(bounds)?;
            }
            FileLine(..) => {
                // no source text in this case!
            }
        }

        if let Some(code) = code {
            if let Some(_) = self.registry.as_ref()
                                          .and_then(|registry| registry.find_description(code)) {
                print_diagnostic(&mut self.dst, &ss[..], Help,
                                 &format!("run `rustc --explain {}` to see a \
                                           detailed explanation", code), None)?;
            }
        }
        Ok(())
    }

    fn highlight_suggestion(&mut self, suggestion: &CodeSuggestion) -> io::Result<()>
    {
        let lines = self.cm.span_to_lines(suggestion.msp.to_span_bounds()).unwrap();
        assert!(!lines.lines.is_empty());

        let complete = suggestion.splice_lines(&self.cm);
        let line_count = cmp::min(lines.lines.len(), MAX_HIGHLIGHT_LINES);
        let display_lines = &lines.lines[..line_count];

        let fm = &*lines.file;
        // Calculate the widest number to format evenly
        let max_digits = line_num_max_digits(display_lines.last().unwrap());

        // print the suggestion without any line numbers, but leave
        // space for them. This helps with lining up with previous
        // snippets from the actual error being reported.
        let mut lines = complete.lines();
        for line in lines.by_ref().take(MAX_HIGHLIGHT_LINES) {
            write!(&mut self.dst, "{0}:{1:2$} {3}\n",
                   fm.name, "", max_digits, line)?;
        }

        // if we elided some lines, add an ellipsis
        if let Some(_) = lines.next() {
            write!(&mut self.dst, "{0:1$} {0:2$} ...\n",
                   "", fm.name.len(), max_digits)?;
        }

        Ok(())
    }

    fn highlight_lines(&mut self,
                       msp: &MultiSpan,
                       lvl: Level)
                       -> io::Result<()>
    {
        let lines = match self.cm.span_to_lines(msp.to_span_bounds()) {
            Ok(lines) => lines,
            Err(_) => {
                write!(&mut self.dst, "(internal compiler error: unprintable span)\n")?;
                return Ok(());
            }
        };

        let fm = &*lines.file;
        if let None = fm.src {
            return Ok(());
        }

        let display_line_infos = &lines.lines[..];
        assert!(display_line_infos.len() > 0);

        // Calculate the widest number to format evenly and fix #11715
        let digits = line_num_max_digits(display_line_infos.last().unwrap());
        let first_line_index = display_line_infos.first().unwrap().line_index;

        let skip = fm.name.chars().count() + digits + 2;

        let mut spans = msp.spans.iter().peekable();
        let mut lines = display_line_infos.iter();
        let mut prev_line_index = first_line_index.wrapping_sub(1);

        // Display at most MAX_HIGHLIGHT_LINES lines.
        let mut remaining_err_lines = MAX_HIGHLIGHT_LINES;

        // To emit a overflowed spans code-lines *AFTER* the rendered spans
        let mut overflowed_buf = String::new();
        let mut overflowed = false;

        // FIXME (#8706)
        'l: loop {
            if remaining_err_lines <= 0 {
                break;
            }
            let line = match lines.next() {
                Some(l) => l,
                None => break,
            };

            // Skip is the number of characters we need to skip because they are
            // part of the 'filename:line ' part of the code line.
            let mut s: String = ::std::iter::repeat(' ').take(skip).collect();
            let mut col = skip;
            let mut lastc = ' ';

            let cur_line_str = fm.get_line(line.line_index).unwrap();
            let mut line_chars = cur_line_str.chars().enumerate().peekable();
            let mut line_spans = 0;

            // Assemble spans for this line
            loop {
                // Peek here to preserve the span if it doesn't belong to this line
                let sp = match spans.peek() {
                    Some(sp) => **sp,
                    None => break,
                };
                let lo = self.cm.lookup_char_pos(sp.lo);
                let hi = self.cm.lookup_char_pos(sp.hi);
                let line_num = line.line_index + 1;

                if !(lo.line <= line_num && hi.line >= line_num) {
                    // This line is not contained in the span
                    if overflowed {
                        // Never elide the final line of an overflowed span
                        prev_line_index = line.line_index - 1;
                        overflowed = false;
                        break;
                    }

                    if line_spans == 0 {
                        continue 'l;
                    } else {
                        // This line is finished, now render the spans we've assembled
                        break;
                    }
                }
                spans.next();
                line_spans += 1;

                if lo.line != hi.line {
                    // Assemble extra code lines to be emitted after this lines spans
                    // (substract `2` because the first and last line are rendered normally)
                    let max_lines = cmp::min(remaining_err_lines, MAX_SP_LINES) - 2;
                    prev_line_index = line.line_index;
                    let count = cmp::min((hi.line - lo.line - 1), max_lines);
                    for _ in 0..count {
                        let line = match lines.next() {
                            Some(l) => l,
                            None => break,
                        };
                        let line_str = fm.get_line(line.line_index).unwrap();
                        overflowed_buf.push_str(&format!("{}:{:>width$} {}\n",
                                                       fm.name,
                                                       line.line_index + 1,
                                                       line_str,
                                                       width=digits));
                        remaining_err_lines -= 1;
                        prev_line_index += 1
                    }
                    // Remember that the span overflowed to ensure
                    // that we emit its last line exactly once
                    // (other spans may, or may not, start on it)
                    overflowed = true;
                    break;
                }

                for (pos, ch) in line_chars.by_ref() {
                    lastc = ch;
                    if pos >= lo.col.to_usize() { break; }
                    // Whenever a tab occurs on the code line, we insert one on
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

                s.push('^');
                let col_ptr = col;
                let count = match lastc {
                    // Most terminals have a tab stop every eight columns by default
                    '\t' => 8 - col%8,
                    _ => 1,
                };
                col += count;
                s.extend(::std::iter::repeat('~').take(count));

                let hi = self.cm.lookup_char_pos(sp.hi);
                if hi.col != lo.col {
                    let mut chars = line_chars.by_ref();
                    loop {
                        // We peek here to preserve the value for the next span
                        let (pos, ch) = match chars.peek() {
                            Some(elem) => *elem,
                            None => break,
                        };
                        if pos >= hi.col.to_usize() { break; }
                        let count = match ch {
                            '\t' => 8 - col%8,
                            _ => 1,
                        };
                        col += count;
                        s.extend(::std::iter::repeat('~').take(count));

                        chars.next();
                    }
                }
                if (col - col_ptr) > 0 {
                    // One extra squiggly is replaced by a "^"
                    s.pop();
                }
            }

            // If we elided something put an ellipsis.
            if prev_line_index != line.line_index.wrapping_sub(1) && !overflowed {
                write!(&mut self.dst, "{0:1$}...\n", "", skip)?;
            }

            // Print offending code-line
            remaining_err_lines -= 1;
            write!(&mut self.dst, "{}:{:>width$} {}\n",
                   fm.name,
                   line.line_index + 1,
                   cur_line_str,
                   width=digits)?;

            if s.len() > skip {
                // Render the spans we assembled previously (if any).
                println_maybe_styled!(&mut self.dst, term::Attr::ForegroundColor(lvl.color()),
                                      "{}", s)?;
            }

            if !overflowed_buf.is_empty() {
                // Print code-lines trailing the rendered spans (when a span overflows)
                write!(&mut self.dst, "{}", &overflowed_buf)?;
                overflowed_buf.clear();
            } else {
                prev_line_index = line.line_index;
            }
        }

        // If we elided something, put an ellipsis.
        if lines.next().is_some() {
            write!(&mut self.dst, "{0:1$}...\n", "", skip)?;
        }
        Ok(())
    }

    /// Here are the differences between this and the normal `highlight_lines`:
    /// `end_highlight_lines` will always put arrow on the last byte of each
    /// span (instead of the first byte). Also, when a span is too long (more
    /// than 6 lines), `end_highlight_lines` will print the first line, then
    /// dot dot dot, then last line, whereas `highlight_lines` prints the first
    /// six lines.
    #[allow(deprecated)]
    fn end_highlight_lines(&mut self,
                           msp: &MultiSpan,
                           lvl: Level)
                          -> io::Result<()> {
        let lines = match self.cm.span_to_lines(msp.to_span_bounds()) {
            Ok(lines) => lines,
            Err(_) => {
                write!(&mut self.dst, "(internal compiler error: unprintable span)\n")?;
                return Ok(());
            }
        };

        let fm = &*lines.file;
        if let None = fm.src {
            return Ok(());
        }

        let lines = &lines.lines[..];

        // Calculate the widest number to format evenly
        let first_line = lines.first().unwrap();
        let last_line = lines.last().unwrap();
        let digits = line_num_max_digits(last_line);

        let skip = fm.name.chars().count() + digits + 2;

        let mut spans = msp.spans.iter().peekable();
        let mut lines = lines.iter();
        let mut prev_line_index = first_line.line_index.wrapping_sub(1);

        // Display at most MAX_HIGHLIGHT_LINES lines.
        let mut remaining_err_lines = MAX_HIGHLIGHT_LINES;

        'l: loop {
            if remaining_err_lines <= 0 {
                break;
            }
            let line = match lines.next() {
                Some(line) => line,
                None => break,
            };

            // Skip is the number of characters we need to skip because they are
            // part of the 'filename:line ' part of the previous line.
            let mut s: String = ::std::iter::repeat(' ').take(skip).collect();

            let line_str = fm.get_line(line.line_index).unwrap();
            let mut line_chars = line_str.chars().enumerate();
            let mut line_spans = 0;

            loop {
                // Peek here to preserve the span if it doesn't belong to this line
                let sp = match spans.peek() {
                    Some(sp) => **sp,
                    None => break,
                };
                let lo = self.cm.lookup_char_pos(sp.lo);
                let hi = self.cm.lookup_char_pos(sp.hi);
                let elide_sp = (hi.line - lo.line) >= MAX_SP_LINES;

                let line_num = line.line_index + 1;
                if !(lo.line <= line_num && hi.line >= line_num) {
                    // This line is not contained in the span
                    if line_spans == 0 {
                        continue 'l;
                    } else {
                        // This line is finished, now render the spans we've assembled
                        break
                    }
                } else if hi.line > line_num {
                    if elide_sp && lo.line < line_num {
                        // This line is inbetween the first and last line of the span,
                        // so we may want to elide it.
                        continue 'l;
                    } else {
                        break
                    }
                }
                line_spans += 1;
                spans.next();

                for (pos, ch) in line_chars.by_ref() {
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
                s.push('^');
            }

            if prev_line_index != line.line_index.wrapping_sub(1) {
                // If we elided something, put an ellipsis.
                write!(&mut self.dst, "{0:1$}...\n", "", skip)?;
            }

            // Print offending code-lines
            write!(&mut self.dst, "{}:{:>width$} {}\n", fm.name,
                   line.line_index + 1, line_str, width=digits)?;
            remaining_err_lines -= 1;

            if s.len() > skip {
                // Render the spans we assembled previously (if any)
                println_maybe_styled!(&mut self.dst, term::Attr::ForegroundColor(lvl.color()),
                                      "{}", s)?;
            }
            prev_line_index = line.line_index;
        }
        Ok(())
    }

    fn print_macro_backtrace(&mut self,
                             sp: Span)
                             -> io::Result<()> {
        for trace in self.cm.macro_backtrace(sp) {
            let mut diag_string =
                format!("in this expansion of {}", trace.macro_decl_name);
            if let Some(def_site_span) = trace.def_site_span {
                diag_string.push_str(
                    &format!(" (defined in {})",
                        self.cm.span_to_filename(def_site_span)));
            }
            let snippet = self.cm.span_to_string(sp);
            print_diagnostic(&mut self.dst, &snippet, Note, &diag_string, None)?;
        }
        Ok(())
    }
}

fn line_num_max_digits(line: &codemap::LineInfo) -> usize {
    let mut max_line_num = line.line_index + 1;
    let mut digits = 0;
    while max_line_num > 0 {
        max_line_num /= 10;
        digits += 1;
    }
    digits
}

fn print_diagnostic(dst: &mut Destination,
                    topic: &str,
                    lvl: Level,
                    msg: &str,
                    code: Option<&str>)
                    -> io::Result<()> {
    if !topic.is_empty() {
        write!(dst, "{} ", topic)?;
    }

    print_maybe_styled!(dst, term::Attr::ForegroundColor(lvl.color()),
                        "{}: ", lvl.to_string())?;
    print_maybe_styled!(dst, term::Attr::Bold, "{}", msg)?;

    if let Some(code) = code {
        let style = term::Attr::ForegroundColor(term::color::BRIGHT_MAGENTA);
        print_maybe_styled!(dst, style, " [{}]", code.clone())?;
    }
    write!(dst, "\n")?;
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
                t.attr(color)?;
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
                t.write_fmt(args)?;
                t.reset()?;
                if print_newline_at_end {
                    t.write_all(b"\n")
                } else {
                    Ok(())
                }
            }
            Raw(ref mut w) => {
                w.write_fmt(args)?;
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


#[cfg(test)]
mod test {
    use errors::{Level, CodeSuggestion};
    use super::EmitterWriter;
    use codemap::{mk_sp, CodeMap, Span, MultiSpan, BytePos, NO_EXPANSION};
    use std::sync::{Arc, Mutex};
    use std::io::{self, Write};
    use std::str::from_utf8;
    use std::rc::Rc;

    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            Write::write(&mut *self.0.lock().unwrap(), data)
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }

    /// Given a string like " ^~~~~~~~~~~~ ", produces a span
    /// coverting that range. The idea is that the string has the same
    /// length as the input, and we uncover the byte positions.  Note
    /// that this can span lines and so on.
    fn span_from_selection(input: &str, selection: &str) -> Span {
        assert_eq!(input.len(), selection.len());
        let left_index = selection.find('^').unwrap() as u32;
        let right_index = selection.rfind('~').map(|x|x as u32).unwrap_or(left_index);
        Span { lo: BytePos(left_index), hi: BytePos(right_index + 1), expn_id: NO_EXPANSION }
    }

    // Diagnostic doesn't align properly in span where line number increases by one digit
    #[test]
    fn test_hilight_suggestion_issue_11715() {
        let data = Arc::new(Mutex::new(Vec::new()));
        let cm = Rc::new(CodeMap::new());
        let mut ew = EmitterWriter::new(Box::new(Sink(data.clone())), None, cm.clone());
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
        println!("highlight_lines");
        ew.highlight_lines(&sp.into(), lvl).unwrap();
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

    #[test]
    fn test_single_span_splice() {
        // Test that a `MultiSpan` containing a single span splices a substition correctly
        let cm = CodeMap::new();
        let inputtext = "aaaaa\nbbbbBB\nCCC\nDDDDDddddd\neee\n";
        let selection = "     \n    ^~\n~~~\n~~~~~     \n   \n";
        cm.new_filemap_and_lines("blork.rs", inputtext);
        let sp = span_from_selection(inputtext, selection);
        let msp: MultiSpan = sp.into();

        // check that we are extracting the text we thought we were extracting
        assert_eq!(&cm.span_to_snippet(sp).unwrap(), "BB\nCCC\nDDDDD");

        let substitute = "ZZZZZZ".to_owned();
        let expected = "bbbbZZZZZZddddd";
        let suggest = CodeSuggestion {
            msp: msp,
            substitutes: vec![substitute],
        };
        assert_eq!(suggest.splice_lines(&cm), expected);
    }

    #[test]
    fn test_multiple_span_splice() {
        // Test that a `MultiSpan` containing multiple spans splices substitions on
        // several lines correctly
        let cm = CodeMap::new();
        let inp = "aaaaabbbbBB\nZZ\nZZ\nCCCDDDDDdddddeee";
        let sp1 = "     ^~~~~~\n  \n  \n                ";
        let sp2 = "           \n  \n  \n^~~~~~          ";
        let sp3 = "           \n  \n  \n        ^~~     ";
        let sp4 = "           \n  \n  \n           ^~~~ ";

        let span_eq = |sp, eq| assert_eq!(&cm.span_to_snippet(sp).unwrap(), eq);

        cm.new_filemap_and_lines("blork.rs", inp);
        let sp1 = span_from_selection(inp, sp1);
        let sp2 = span_from_selection(inp, sp2);
        let sp3 = span_from_selection(inp, sp3);
        let sp4 = span_from_selection(inp, sp4);
        span_eq(sp1, "bbbbBB");
        span_eq(sp2, "CCCDDD");
        span_eq(sp3, "ddd");
        span_eq(sp4, "ddee");

        let substitutes: Vec<String> = ["1", "2", "3", "4"].iter().map(|x|x.to_string()).collect();
        let expected = "aaaaa1\nZZ\nZZ\n2DD34e";

        let test = |msp| {
            let suggest = CodeSuggestion {
                msp: msp,
                substitutes: substitutes.clone(),
            };
            let actual = suggest.splice_lines(&cm);
            assert_eq!(actual, expected);
        };
        test(MultiSpan { spans: vec![sp1, sp2, sp3, sp4] });

        // Test ordering and merging by `MultiSpan::push`
        let mut msp = MultiSpan::new();
        msp.push_merge(sp2);
        msp.push_merge(sp1);
        assert_eq!(&msp.spans, &[sp1, sp2]);
        msp.push_merge(sp4);
        assert_eq!(&msp.spans, &[sp1, sp2, sp4]);
        msp.push_merge(sp3);
        assert_eq!(&msp.spans, &[sp1, sp2, sp3, sp4]);
        test(msp);
    }

    #[test]
    fn test_multispan_highlight() {
        let data = Arc::new(Mutex::new(Vec::new()));
        let cm = Rc::new(CodeMap::new());
        let mut diag = EmitterWriter::new(Box::new(Sink(data.clone())), None, cm.clone());

        let inp =       "_____aaaaaa____bbbbbb__cccccdd_";
        let sp1 =       "     ^~~~~~                    ";
        let sp2 =       "               ^~~~~~          ";
        let sp3 =       "                       ^~~~~   ";
        let sp4 =       "                          ^~~~ ";
        let sp34 =      "                       ^~~~~~~ ";
        let sp4_end =   "                            ^~ ";

        let expect_start = "dummy.txt:1 _____aaaaaa____bbbbbb__cccccdd_\n\
                         \x20                ^~~~~~    ^~~~~~  ^~~~~~~\n";
        let expect_end =   "dummy.txt:1 _____aaaaaa____bbbbbb__cccccdd_\n\
                         \x20                     ^         ^      ^ ^\n";

        let span = |sp, expected| {
            let sp = span_from_selection(inp, sp);
            assert_eq!(&cm.span_to_snippet(sp).unwrap(), expected);
            sp
        };
        cm.new_filemap_and_lines("dummy.txt", inp);
        let sp1 = span(sp1, "aaaaaa");
        let sp2 = span(sp2, "bbbbbb");
        let sp3 = span(sp3, "ccccc");
        let sp4 = span(sp4, "ccdd");
        let sp34 = span(sp34, "cccccdd");
        let sp4_end = span(sp4_end, "dd");

        let spans = vec![sp1, sp2, sp3, sp4];

        let test = |expected, highlight: &mut FnMut()| {
            data.lock().unwrap().clear();
            highlight();
            let vec = data.lock().unwrap().clone();
            let actual = from_utf8(&vec[..]).unwrap();
            assert_eq!(actual, expected);
        };

        let msp = MultiSpan { spans: vec![sp1, sp2, sp34] };
        let msp_end = MultiSpan { spans: vec![sp1, sp2, sp3, sp4_end] };
        test(expect_start, &mut || {
            diag.highlight_lines(&msp, Level::Error).unwrap();
        });
        test(expect_end, &mut || {
            diag.end_highlight_lines(&msp_end, Level::Error).unwrap();
        });
        test(expect_start, &mut || {
            for msp in cm.group_spans(spans.clone()) {
                diag.highlight_lines(&msp, Level::Error).unwrap();
            }
        });
        test(expect_end, &mut || {
            for msp in cm.end_group_spans(spans.clone()) {
                diag.end_highlight_lines(&msp, Level::Error).unwrap();
            }
        });
    }

    #[test]
    fn test_huge_multispan_highlight() {
        let data = Arc::new(Mutex::new(Vec::new()));
        let cm = Rc::new(CodeMap::new());
        let mut diag = EmitterWriter::new(Box::new(Sink(data.clone())), None, cm.clone());

        let inp = "aaaaa\n\
                   aaaaa\n\
                   aaaaa\n\
                   bbbbb\n\
                   ccccc\n\
                   xxxxx\n\
                   yyyyy\n\
                   _____\n\
                   ddd__eee_\n\
                   elided\n\
                   __f_gg";
        let file = cm.new_filemap_and_lines("dummy.txt", inp);

        let span = |lo, hi, (off_lo, off_hi)| {
            let lines = file.lines.borrow();
            let (mut lo, mut hi): (BytePos, BytePos) = (lines[lo], lines[hi]);
            lo.0 += off_lo;
            hi.0 += off_hi;
            mk_sp(lo, hi)
        };
        let sp0 = span(4, 6, (0, 5));
        let sp1 = span(0, 6, (0, 5));
        let sp2 = span(8, 8, (0, 3));
        let sp3 = span(8, 8, (5, 8));
        let sp4 = span(10, 10, (2, 3));
        let sp5 = span(10, 10, (4, 6));

        let expect0 = "dummy.txt: 5 ccccc\n\
                       dummy.txt: 6 xxxxx\n\
                       dummy.txt: 7 yyyyy\n\
                    \x20            ...\n\
                       dummy.txt: 9 ddd__eee_\n\
                    \x20            ^~~  ^~~\n\
                    \x20            ...\n\
                       dummy.txt:11 __f_gg\n\
                    \x20              ^ ^~\n";

        let expect = "dummy.txt: 1 aaaaa\n\
                      dummy.txt: 2 aaaaa\n\
                      dummy.txt: 3 aaaaa\n\
                      dummy.txt: 4 bbbbb\n\
                      dummy.txt: 5 ccccc\n\
                      dummy.txt: 6 xxxxx\n\
                   \x20            ...\n";

        let expect_g1 = "dummy.txt:1 aaaaa\n\
                         dummy.txt:2 aaaaa\n\
                         dummy.txt:3 aaaaa\n\
                         dummy.txt:4 bbbbb\n\
                         dummy.txt:5 ccccc\n\
                         dummy.txt:6 xxxxx\n\
                      \x20           ...\n";

        let expect2 = "dummy.txt: 9 ddd__eee_\n\
                    \x20            ^~~  ^~~\n\
                    \x20            ...\n\
                       dummy.txt:11 __f_gg\n\
                    \x20              ^ ^~\n";


        let expect_end = "dummy.txt: 1 aaaaa\n\
                       \x20            ...\n\
                          dummy.txt: 7 yyyyy\n\
                       \x20                ^\n\
                       \x20            ...\n\
                          dummy.txt: 9 ddd__eee_\n\
                       \x20              ^    ^\n\
                       \x20            ...\n\
                          dummy.txt:11 __f_gg\n\
                       \x20              ^  ^\n";

        let expect0_end = "dummy.txt: 5 ccccc\n\
                           dummy.txt: 6 xxxxx\n\
                           dummy.txt: 7 yyyyy\n\
                        \x20                ^\n\
                        \x20            ...\n\
                           dummy.txt: 9 ddd__eee_\n\
                        \x20              ^    ^\n\
                        \x20            ...\n\
                           dummy.txt:11 __f_gg\n\
                        \x20              ^  ^\n";

        let expect_end_g1 = "dummy.txt:1 aaaaa\n\
                          \x20           ...\n\
                             dummy.txt:7 yyyyy\n\
                          \x20               ^\n";

        let expect2_end = "dummy.txt: 9 ddd__eee_\n\
                        \x20              ^    ^\n\
                        \x20            ...\n\
                           dummy.txt:11 __f_gg\n\
                        \x20              ^  ^\n";

        let expect_groups = [expect2, expect_g1];
        let expect_end_groups = [expect2_end, expect_end_g1];
        let spans = vec![sp3, sp1, sp4, sp2, sp5];

        macro_rules! test {
            ($expected: expr, $highlight: expr) => ({
                data.lock().unwrap().clear();
                $highlight();
                let vec = data.lock().unwrap().clone();
                let actual = from_utf8(&vec[..]).unwrap();
                println!("actual:");
                println!("{}", actual);
                println!("expected:");
                println!("{}", $expected);
                assert_eq!(&actual[..], &$expected[..]);
            });
        }

        let msp0 = MultiSpan { spans: vec![sp0, sp2, sp3, sp4, sp5] };
        let msp = MultiSpan { spans: vec![sp1, sp2, sp3, sp4, sp5] };
        let msp2 = MultiSpan { spans: vec![sp2, sp3, sp4, sp5] };

        test!(expect0, || {
            diag.highlight_lines(&msp0, Level::Error).unwrap();
        });
        test!(expect0_end, || {
            diag.end_highlight_lines(&msp0, Level::Error).unwrap();
        });
        test!(expect, || {
            diag.highlight_lines(&msp, Level::Error).unwrap();
        });
        test!(expect_end, || {
            diag.end_highlight_lines(&msp, Level::Error).unwrap();
        });
        test!(expect2, || {
            diag.highlight_lines(&msp2, Level::Error).unwrap();
        });
        test!(expect2_end, || {
            diag.end_highlight_lines(&msp2, Level::Error).unwrap();
        });
        for (msp, expect) in cm.group_spans(spans.clone()).iter().zip(expect_groups.iter()) {
            test!(expect, || {
                diag.highlight_lines(&msp, Level::Error).unwrap();
            });
        }
        for (msp, expect) in cm.group_spans(spans.clone()).iter().zip(expect_end_groups.iter()) {
            test!(expect, || {
                diag.end_highlight_lines(&msp, Level::Error).unwrap();
            });
        }
    }
}
