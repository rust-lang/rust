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

use syntax_pos::{COMMAND_LINE_SP, DUMMY_SP, Span, MultiSpan, LineInfo};
use registry;

use check_old_skool;
use {Level, RenderSpan, CodeSuggestion, DiagnosticBuilder, CodeMapper};
use RenderSpan::*;
use Level::*;
use snippet::{RenderedLineKind, SnippetData, Style, FormatMode};

use std::{cmp, fmt};
use std::io::prelude::*;
use std::io;
use std::rc::Rc;
use term;

/// Emitter trait for emitting errors. Do not implement this directly:
/// implement `CoreEmitter` instead.
pub trait Emitter {
    /// Emit a standalone diagnostic message.
    fn emit(&mut self, span: &MultiSpan, msg: &str, code: Option<&str>, lvl: Level);

    /// Emit a structured diagnostic.
    fn emit_struct(&mut self, db: &DiagnosticBuilder);
}

pub trait CoreEmitter {
    fn emit_message(&mut self,
                    rsp: &RenderSpan,
                    msg: &str,
                    code: Option<&str>,
                    lvl: Level,
                    is_header: bool,
                    show_snippet: bool);
}

impl<T: CoreEmitter> Emitter for T {
    fn emit(&mut self,
            msp: &MultiSpan,
            msg: &str,
            code: Option<&str>,
            lvl: Level) {
        self.emit_message(&FullSpan(msp.clone()),
                          msg,
                          code,
                          lvl,
                          true,
                          true);
    }

    fn emit_struct(&mut self, db: &DiagnosticBuilder) {
        let old_school = check_old_skool();
        let db_span = FullSpan(db.span.clone());
        self.emit_message(&FullSpan(db.span.clone()),
                          &db.message,
                          db.code.as_ref().map(|s| &**s),
                          db.level,
                          true,
                          true);
        for child in &db.children {
            let render_span = child.render_span
                                   .clone()
                                   .unwrap_or_else(
                                       || FullSpan(child.span.clone()));

            if !old_school {
                self.emit_message(&render_span,
                                    &child.message,
                                    None,
                                    child.level,
                                    false,
                                    true);
            } else {
                let (render_span, show_snippet) = match render_span.span().primary_span() {
                    None => (db_span.clone(), false),
                    _ => (render_span, true)
                };
                self.emit_message(&render_span,
                                    &child.message,
                                    None,
                                    child.level,
                                    false,
                                    show_snippet);
            }
        }
    }
}

/// maximum number of lines we will print for each error; arbitrary.
pub const MAX_HIGHLIGHT_LINES: usize = 6;

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

impl CoreEmitter for BasicEmitter {
    fn emit_message(&mut self,
                    _rsp: &RenderSpan,
                    msg: &str,
                    code: Option<&str>,
                    lvl: Level,
                    _is_header: bool,
                    _show_snippet: bool) {
        // we ignore the span as we have no access to a codemap at this point
        if let Err(e) = print_diagnostic(&mut self.dst, "", lvl, msg, code) {
            panic!("failed to print diagnostics: {:?}", e);
        }
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
    registry: Option<registry::Registry>,
    cm: Rc<CodeMapper>,

    /// Is this the first error emitted thus far? If not, we emit a
    /// `\n` before the top-level errors.
    first: bool,

    // For now, allow an old-school mode while we transition
    format_mode: FormatMode
}

impl CoreEmitter for EmitterWriter {
    fn emit_message(&mut self,
                    rsp: &RenderSpan,
                    msg: &str,
                    code: Option<&str>,
                    lvl: Level,
                    is_header: bool,
                    show_snippet: bool) {
        match self.emit_message_(rsp, msg, code, lvl, is_header, show_snippet) {
            Ok(()) => { }
            Err(e) => panic!("failed to emit error: {}", e)
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
                  registry: Option<registry::Registry>,
                  code_map: Rc<CodeMapper>,
                  format_mode: FormatMode)
                  -> EmitterWriter {
        if color_config.use_color() {
            let dst = Destination::from_stderr();
            EmitterWriter { dst: dst,
                            registry: registry,
                            cm: code_map,
                            first: true,
                            format_mode: format_mode.clone() }
        } else {
            EmitterWriter { dst: Raw(Box::new(io::stderr())),
                            registry: registry,
                            cm: code_map,
                            first: true,
                            format_mode: format_mode.clone() }
        }
    }

    pub fn new(dst: Box<Write + Send>,
               registry: Option<registry::Registry>,
               code_map: Rc<CodeMapper>,
               format_mode: FormatMode)
               -> EmitterWriter {
        EmitterWriter { dst: Raw(dst),
                        registry: registry,
                        cm: code_map,
                        first: true,
                        format_mode: format_mode.clone() }
    }

    fn emit_message_(&mut self,
                     rsp: &RenderSpan,
                     msg: &str,
                     code: Option<&str>,
                     lvl: Level,
                     is_header: bool,
                     show_snippet: bool)
                     -> io::Result<()> {
        let old_school = match self.format_mode {
            FormatMode::NewErrorFormat => false,
            FormatMode::OriginalErrorFormat => true,
            FormatMode::EnvironmentSelected => check_old_skool()
        };

        if is_header {
            if self.first {
                self.first = false;
            } else {
                if !old_school {
                    write!(self.dst, "\n")?;
                }
            }
        }

        match code {
            Some(code) if self.registry.as_ref()
                                       .and_then(|registry| registry.find_description(code))
                                       .is_some() => {
                let code_with_explain = String::from("--explain ") + code;
                if old_school {
                    let loc = match rsp.span().primary_span() {
                        Some(COMMAND_LINE_SP) | Some(DUMMY_SP) => "".to_string(),
                        Some(ps) => self.cm.span_to_string(ps),
                        None => "".to_string()
                    };
                    print_diagnostic(&mut self.dst, &loc, lvl, msg, Some(code))?
                }
                else {
                    print_diagnostic(&mut self.dst, "", lvl, msg, Some(&code_with_explain))?
                }
            }
            _ => {
                if old_school {
                    let loc = match rsp.span().primary_span() {
                        Some(COMMAND_LINE_SP) | Some(DUMMY_SP) => "".to_string(),
                        Some(ps) => self.cm.span_to_string(ps),
                        None => "".to_string()
                    };
                    print_diagnostic(&mut self.dst, &loc, lvl, msg, code)?
                }
                else {
                    print_diagnostic(&mut self.dst, "", lvl, msg, code)?
                }
            }
        }

        if !show_snippet {
            return Ok(());
        }

        // Watch out for various nasty special spans; don't try to
        // print any filename or anything for those.
        match rsp.span().primary_span() {
            Some(COMMAND_LINE_SP) | Some(DUMMY_SP) => {
                return Ok(());
            }
            _ => { }
        }

        // Otherwise, print out the snippet etc as needed.
        match *rsp {
            FullSpan(ref msp) => {
                self.highlight_lines(msp, lvl)?;
                if let Some(primary_span) = msp.primary_span() {
                    self.print_macro_backtrace(primary_span)?;
                }
            }
            Suggestion(ref suggestion) => {
                self.highlight_suggestion(suggestion)?;
                if let Some(primary_span) = rsp.span().primary_span() {
                    self.print_macro_backtrace(primary_span)?;
                }
            }
        }
        if old_school {
            match code {
                Some(code) if self.registry.as_ref()
                                        .and_then(|registry| registry.find_description(code))
                                        .is_some() => {
                    let loc = match rsp.span().primary_span() {
                        Some(COMMAND_LINE_SP) | Some(DUMMY_SP) => "".to_string(),
                        Some(ps) => self.cm.span_to_string(ps),
                        None => "".to_string()
                    };
                    let msg = "run `rustc --explain ".to_string() + &code.to_string() +
                        "` to see a detailed explanation";
                    print_diagnostic(&mut self.dst, &loc, Level::Help, &msg,
                        None)?
                }
                _ => ()
            }
        }
        Ok(())
    }

    fn highlight_suggestion(&mut self, suggestion: &CodeSuggestion) -> io::Result<()>
    {
        use std::borrow::Borrow;

        let primary_span = suggestion.msp.primary_span().unwrap();
        let lines = self.cm.span_to_lines(primary_span).unwrap();
        assert!(!lines.lines.is_empty());

        let complete = suggestion.splice_lines(self.cm.borrow());
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

    pub fn highlight_lines(&mut self,
                       msp: &MultiSpan,
                       lvl: Level)
                       -> io::Result<()>
    {
        let old_school = match self.format_mode {
            FormatMode::NewErrorFormat => false,
            FormatMode::OriginalErrorFormat => true,
            FormatMode::EnvironmentSelected => check_old_skool()
        };

        let mut snippet_data = SnippetData::new(self.cm.clone(),
                                                msp.primary_span(),
                                                self.format_mode.clone());
        if old_school {
            let mut output_vec = vec![];

            for span_label in msp.span_labels() {
                let mut snippet_data = SnippetData::new(self.cm.clone(),
                                                        Some(span_label.span),
                                                        self.format_mode.clone());

                snippet_data.push(span_label.span,
                                  span_label.is_primary,
                                  span_label.label);
                if span_label.is_primary {
                    output_vec.insert(0, snippet_data);
                }
                else {
                    output_vec.push(snippet_data);
                }
            }

            for snippet_data in output_vec.iter() {
                let rendered_lines = snippet_data.render_lines();
                for rendered_line in &rendered_lines {
                    for styled_string in &rendered_line.text {
                        self.dst.apply_style(lvl, &rendered_line.kind, styled_string.style)?;
                        write!(&mut self.dst, "{}", styled_string.text)?;
                        self.dst.reset_attrs()?;
                    }
                    write!(&mut self.dst, "\n")?;
                }
            }
        }
        else {
            for span_label in msp.span_labels() {
                snippet_data.push(span_label.span,
                                  span_label.is_primary,
                                  span_label.label);
            }
            let rendered_lines = snippet_data.render_lines();
            for rendered_line in &rendered_lines {
                for styled_string in &rendered_line.text {
                    self.dst.apply_style(lvl, &rendered_line.kind, styled_string.style)?;
                    write!(&mut self.dst, "{}", styled_string.text)?;
                    self.dst.reset_attrs()?;
                }
                write!(&mut self.dst, "\n")?;
            }
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
            let snippet = self.cm.span_to_string(trace.call_site);
            print_diagnostic(&mut self.dst, &snippet, Note, &diag_string, None)?;
        }
        Ok(())
    }
}

fn line_num_max_digits(line: &LineInfo) -> usize {
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
        let old_school = check_old_skool();
        if !old_school {
            write!(dst, "{}: ", topic)?;
        }
        else {
            write!(dst, "{} ", topic)?;
        }
        dst.reset_attrs()?;
    }
    dst.start_attr(term::Attr::Bold)?;
    dst.start_attr(term::Attr::ForegroundColor(lvl.color()))?;
    write!(dst, "{}", lvl.to_string())?;
    dst.reset_attrs()?;
    write!(dst, ": ")?;
    dst.start_attr(term::Attr::Bold)?;
    write!(dst, "{}", msg)?;

    if let Some(code) = code {
        let style = term::Attr::ForegroundColor(term::color::BRIGHT_MAGENTA);
        print_maybe_styled!(dst, style, " [{}]", code.clone())?;
    }

    dst.reset_attrs()?;
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

    fn apply_style(&mut self,
                   lvl: Level,
                   _kind: &RenderedLineKind,
                   style: Style)
                   -> io::Result<()> {
        match style {
            Style::FileNameStyle |
            Style::LineAndColumn => {
            }
            Style::LineNumber => {
                self.start_attr(term::Attr::Bold)?;
                self.start_attr(term::Attr::ForegroundColor(term::color::BRIGHT_BLUE))?;
            }
            Style::Quotation => {
            }
            Style::OldSkoolNote => {
                self.start_attr(term::Attr::Bold)?;
                self.start_attr(term::Attr::ForegroundColor(term::color::BRIGHT_GREEN))?;
            }
            Style::OldSkoolNoteText => {
                self.start_attr(term::Attr::Bold)?;
            }
            Style::UnderlinePrimary | Style::LabelPrimary => {
                self.start_attr(term::Attr::Bold)?;
                self.start_attr(term::Attr::ForegroundColor(lvl.color()))?;
            }
            Style::UnderlineSecondary | Style::LabelSecondary => {
                self.start_attr(term::Attr::Bold)?;
                self.start_attr(term::Attr::ForegroundColor(term::color::BRIGHT_BLUE))?;
            }
            Style::NoStyle => {
            }
        }
        Ok(())
    }

    fn start_attr(&mut self, attr: term::Attr) -> io::Result<()> {
        match *self {
            Terminal(ref mut t) => { t.attr(attr)?; }
            Raw(_) => { }
        }
        Ok(())
    }

    fn reset_attrs(&mut self) -> io::Result<()> {
        match *self {
            Terminal(ref mut t) => { t.reset()?; }
            Raw(_) => { }
        }
        Ok(())
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
