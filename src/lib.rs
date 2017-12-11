// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]

#[macro_use]
extern crate derive_new;
extern crate diff;
#[macro_use]
extern crate log;
extern crate regex;
extern crate rustc_errors as errors;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate syntax;
extern crate term;
extern crate unicode_segmentation;

use std::collections::HashMap;
use std::fmt;
use std::io::{self, stdout, Write};
use std::iter::repeat;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use errors::{DiagnosticBuilder, Handler};
use errors::emitter::{ColorConfig, EmitterWriter};
use syntax::ast;
use syntax::codemap::{CodeMap, FilePathMapping};
use syntax::parse::{self, ParseSess};

use checkstyle::{output_footer, output_header};
use config::Config;
use filemap::FileMap;
use issues::{BadIssueSeeker, Issue};
use utils::use_colored_tty;
use visitor::{FmtVisitor, SnippetProvider};

pub use self::summary::Summary;

#[macro_use]
mod utils;
mod shape;
mod spanned;
pub mod config;
pub mod codemap;
pub mod filemap;
pub mod file_lines;
pub mod visitor;
mod checkstyle;
mod closures;
mod items;
mod missed_spans;
mod lists;
mod types;
mod expr;
mod imports;
mod issues;
mod rewrite;
mod string;
mod comment;
pub mod modules;
pub mod rustfmt_diff;
mod chains;
mod macros;
mod patterns;
mod summary;
mod vertical;

pub enum ErrorKind {
    // Line has exceeded character limit (found, maximum)
    LineOverflow(usize, usize),
    // Line ends in whitespace
    TrailingWhitespace,
    // TO-DO or FIX-ME item without an issue number
    BadIssue(Issue),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            ErrorKind::LineOverflow(found, maximum) => write!(
                fmt,
                "line exceeded maximum width (maximum: {}, found: {})",
                maximum, found
            ),
            ErrorKind::TrailingWhitespace => write!(fmt, "left behind trailing whitespace"),
            ErrorKind::BadIssue(issue) => write!(fmt, "found {}", issue),
        }
    }
}

// Formatting errors that are identified *after* rustfmt has run.
pub struct FormattingError {
    line: usize,
    kind: ErrorKind,
    is_comment: bool,
    line_buffer: String,
}

impl FormattingError {
    fn msg_prefix(&self) -> &str {
        match self.kind {
            ErrorKind::LineOverflow(..) | ErrorKind::TrailingWhitespace => "error:",
            ErrorKind::BadIssue(_) => "WARNING:",
        }
    }

    fn msg_suffix(&self) -> &str {
        match self.kind {
            ErrorKind::LineOverflow(..) if self.is_comment => {
                "use `error_on_line_overflow_comments = false` to suppress \
                 the warning against line comments\n"
            }
            _ => "",
        }
    }

    // (space, target)
    pub fn format_len(&self) -> (usize, usize) {
        match self.kind {
            ErrorKind::LineOverflow(found, max) => (max, found - max),
            ErrorKind::TrailingWhitespace => {
                let trailing_ws_len = self.line_buffer
                    .chars()
                    .rev()
                    .take_while(|c| c.is_whitespace())
                    .count();
                (self.line_buffer.len() - trailing_ws_len, trailing_ws_len)
            }
            _ => unreachable!(),
        }
    }
}

pub struct FormatReport {
    // Maps stringified file paths to their associated formatting errors.
    file_error_map: HashMap<String, Vec<FormattingError>>,
}

impl FormatReport {
    fn new() -> FormatReport {
        FormatReport {
            file_error_map: HashMap::new(),
        }
    }

    pub fn warning_count(&self) -> usize {
        self.file_error_map
            .iter()
            .map(|(_, errors)| errors.len())
            .fold(0, |acc, x| acc + x)
    }

    pub fn has_warnings(&self) -> bool {
        self.warning_count() > 0
    }

    pub fn print_warnings_fancy(
        &self,
        mut t: Box<term::Terminal<Output = io::Stderr>>,
    ) -> Result<(), term::Error> {
        for (file, errors) in &self.file_error_map {
            for error in errors {
                let prefix_space_len = error.line.to_string().len();
                let prefix_spaces: String = repeat(" ").take(1 + prefix_space_len).collect();

                // First line: the overview of error
                t.fg(term::color::RED)?;
                t.attr(term::Attr::Bold)?;
                write!(t, "{} ", error.msg_prefix())?;
                t.reset()?;
                t.attr(term::Attr::Bold)?;
                write!(t, "{}\n", error.kind)?;

                // Second line: file info
                write!(t, "{}--> ", &prefix_spaces[1..])?;
                t.reset()?;
                write!(t, "{}:{}\n", file, error.line)?;

                // Third to fifth lines: show the line which triggered error, if available.
                if !error.line_buffer.is_empty() {
                    let (space_len, target_len) = error.format_len();
                    t.attr(term::Attr::Bold)?;
                    write!(t, "{}|\n{} | ", prefix_spaces, error.line)?;
                    t.reset()?;
                    write!(t, "{}\n", error.line_buffer)?;
                    t.attr(term::Attr::Bold)?;
                    write!(t, "{}| ", prefix_spaces)?;
                    t.fg(term::color::RED)?;
                    write!(t, "{}\n", target_str(space_len, target_len))?;
                    t.reset()?;
                }

                // The last line: show note if available.
                let msg_suffix = error.msg_suffix();
                if !msg_suffix.is_empty() {
                    t.attr(term::Attr::Bold)?;
                    write!(t, "{}= note: ", prefix_spaces)?;
                    t.reset()?;
                    write!(t, "{}\n", error.msg_suffix())?;
                } else {
                    write!(t, "\n")?;
                }
                t.reset()?;
            }
        }

        if !self.file_error_map.is_empty() {
            t.attr(term::Attr::Bold)?;
            write!(t, "warning: ")?;
            t.reset()?;
            write!(
                t,
                "rustfmt may have failed to format. See previous {} errors.\n\n",
                self.warning_count(),
            )?;
        }

        Ok(())
    }
}

fn target_str(space_len: usize, target_len: usize) -> String {
    let empty_line: String = repeat(" ").take(space_len).collect();
    let overflowed: String = repeat("^").take(target_len).collect();
    empty_line + &overflowed
}

impl fmt::Display for FormatReport {
    // Prints all the formatting errors.
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for (file, errors) in &self.file_error_map {
            for error in errors {
                let prefix_space_len = error.line.to_string().len();
                let prefix_spaces: String = repeat(" ").take(1 + prefix_space_len).collect();

                let error_line_buffer = if error.line_buffer.is_empty() {
                    String::from(" ")
                } else {
                    let (space_len, target_len) = error.format_len();
                    format!(
                        "{}|\n{} | {}\n{}| {}",
                        prefix_spaces,
                        error.line,
                        error.line_buffer,
                        prefix_spaces,
                        target_str(space_len, target_len)
                    )
                };

                let error_info = format!("{} {}", error.msg_prefix(), error.kind);
                let file_info = format!("{}--> {}:{}", &prefix_spaces[1..], file, error.line);
                let msg_suffix = error.msg_suffix();
                let note = if msg_suffix.is_empty() {
                    String::new()
                } else {
                    format!("{}note= ", prefix_spaces)
                };

                write!(
                    fmt,
                    "{}\n{}\n{}\n{}{}\n",
                    error_info,
                    file_info,
                    error_line_buffer,
                    note,
                    error.msg_suffix()
                )?;
            }
        }
        if !self.file_error_map.is_empty() {
            write!(
                fmt,
                "warning: rustfmt may have failed to format. See previous {} errors.\n",
                self.warning_count(),
            )?;
        }
        Ok(())
    }
}

// Formatting which depends on the AST.
fn format_ast<F>(
    krate: &ast::Crate,
    parse_session: &mut ParseSess,
    main_file: &Path,
    config: &Config,
    mut after_file: F,
) -> Result<(FileMap, bool), io::Error>
where
    F: FnMut(&str, &mut String, &[(usize, usize)]) -> Result<bool, io::Error>,
{
    let mut result = FileMap::new();
    // diff mode: check if any files are differing
    let mut has_diff = false;

    // We always skip children for the "Plain" write mode, since there is
    // nothing to distinguish the nested module contents.
    let skip_children = config.skip_children() || config.write_mode() == config::WriteMode::Plain;
    for (path, module) in modules::list_files(krate, parse_session.codemap())? {
        if skip_children && path.as_path() != main_file {
            continue;
        }
        let path_str = path.to_str().unwrap();
        if config.verbose() {
            println!("Formatting {}", path_str);
        }
        let filemap = parse_session
            .codemap()
            .lookup_char_pos(module.inner.lo())
            .file;
        let big_snippet = filemap.src.as_ref().unwrap();
        let snippet_provider = SnippetProvider::new(filemap.start_pos, big_snippet);
        let mut visitor = FmtVisitor::from_codemap(parse_session, config, &snippet_provider);
        // Format inner attributes if available.
        if !krate.attrs.is_empty() && path == main_file {
            visitor.skip_empty_lines(filemap.end_pos);
            if visitor.visit_attrs(&krate.attrs, ast::AttrStyle::Inner) {
                visitor.push_rewrite(module.inner, None);
            } else {
                visitor.format_separate_mod(module, &*filemap);
            }
        } else {
            visitor.last_pos = filemap.start_pos;
            visitor.skip_empty_lines(filemap.end_pos);
            visitor.format_separate_mod(module, &*filemap);
        };

        assert_eq!(
            visitor.line_number,
            ::utils::count_newlines(&format!("{}", visitor.buffer))
        );

        has_diff |= match after_file(path_str, &mut visitor.buffer, &visitor.skipped_range) {
            Ok(result) => result,
            Err(e) => {
                // Create a new error with path_str to help users see which files failed
                let err_msg = path_str.to_string() + &": ".to_string() + &e.to_string();
                return Err(io::Error::new(e.kind(), err_msg));
            }
        };

        result.push((path_str.to_owned(), visitor.buffer));
    }

    Ok((result, has_diff))
}

/// Returns true if the line with the given line number was skipped by `#[rustfmt_skip]`.
fn is_skipped_line(line_number: usize, skipped_range: &[(usize, usize)]) -> bool {
    skipped_range
        .iter()
        .any(|&(lo, hi)| lo <= line_number && line_number <= hi)
}

// Formatting done on a char by char or line by line basis.
// FIXME(#209) warn on bad license
// FIXME(#20) other stuff for parity with make tidy
fn format_lines(
    text: &mut String,
    name: &str,
    skipped_range: &[(usize, usize)],
    config: &Config,
    report: &mut FormatReport,
) {
    // Iterate over the chars in the file map.
    let mut trims = vec![];
    let mut last_wspace: Option<usize> = None;
    let mut line_len = 0;
    let mut cur_line = 1;
    let mut newline_count = 0;
    let mut errors = vec![];
    let mut issue_seeker = BadIssueSeeker::new(config.report_todo(), config.report_fixme());
    let mut prev_char: Option<char> = None;
    let mut is_comment = false;
    let mut line_buffer = String::with_capacity(config.max_width() * 2);
    let mut b = 0;

    for c in text.chars() {
        b += 1;
        if c == '\r' {
            continue;
        }

        let format_line = config.file_lines().contains_line(name, cur_line as usize);

        if format_line {
            // Add warnings for bad todos/ fixmes
            if let Some(issue) = issue_seeker.inspect(c) {
                errors.push(FormattingError {
                    line: cur_line,
                    kind: ErrorKind::BadIssue(issue),
                    is_comment: false,
                    line_buffer: String::new(),
                });
            }
        }

        if c == '\n' {
            if format_line {
                // Check for (and record) trailing whitespace.
                if let Some(lw) = last_wspace {
                    trims.push((cur_line, lw, b, line_buffer.clone()));
                    line_len -= 1;
                }

                // Check for any line width errors we couldn't correct.
                let report_error_on_line_overflow = config.error_on_line_overflow()
                    && !is_skipped_line(cur_line, skipped_range)
                    && (config.error_on_line_overflow_comments() || !is_comment);
                if report_error_on_line_overflow && line_len > config.max_width() {
                    errors.push(FormattingError {
                        line: cur_line,
                        kind: ErrorKind::LineOverflow(line_len, config.max_width()),
                        is_comment: is_comment,
                        line_buffer: line_buffer.clone(),
                    });
                }
            }

            line_len = 0;
            cur_line += 1;
            newline_count += 1;
            last_wspace = None;
            prev_char = None;
            is_comment = false;
            line_buffer.clear();
        } else {
            newline_count = 0;
            line_len += 1;
            if c.is_whitespace() {
                if last_wspace.is_none() {
                    last_wspace = Some(b);
                }
            } else if c == '/' {
                if let Some('/') = prev_char {
                    is_comment = true;
                }
                last_wspace = None;
            } else {
                last_wspace = None;
            }
            prev_char = Some(c);
            line_buffer.push(c);
        }
    }

    if newline_count > 1 {
        debug!("track truncate: {} {}", text.len(), newline_count);
        let line = text.len() - newline_count + 1;
        text.truncate(line);
    }

    for &(l, _, _, ref b) in &trims {
        if !is_skipped_line(l, skipped_range) {
            errors.push(FormattingError {
                line: l,
                kind: ErrorKind::TrailingWhitespace,
                is_comment: false,
                line_buffer: b.clone(),
            });
        }
    }

    report.file_error_map.insert(name.to_owned(), errors);
}

fn parse_input(
    input: Input,
    parse_session: &ParseSess,
) -> Result<ast::Crate, Option<DiagnosticBuilder>> {
    let result = match input {
        Input::File(file) => {
            let mut parser = parse::new_parser_from_file(parse_session, &file);
            parser.cfg_mods = false;
            parser.parse_crate_mod()
        }
        Input::Text(text) => {
            let mut parser =
                parse::new_parser_from_source_str(parse_session, "stdin".to_owned(), text);
            parser.cfg_mods = false;
            parser.parse_crate_mod()
        }
    };

    match result {
        Ok(c) => {
            if parse_session.span_diagnostic.has_errors() {
                // Bail out if the parser recovered from an error.
                Err(None)
            } else {
                Ok(c)
            }
        }
        Err(e) => Err(Some(e)),
    }
}

pub fn format_input<T: Write>(
    input: Input,
    config: &Config,
    mut out: Option<&mut T>,
) -> Result<(Summary, FileMap, FormatReport), (io::Error, Summary)> {
    let mut summary = Summary::default();
    if config.disable_all_formatting() {
        // When the input is from stdin, echo back the input.
        if let Input::Text(ref buf) = input {
            if let Err(e) = io::stdout().write_all(buf.as_bytes()) {
                return Err((e, summary));
            }
        }
        return Ok((summary, FileMap::new(), FormatReport::new()));
    }
    let codemap = Rc::new(CodeMap::new(FilePathMapping::empty()));

    let tty_handler =
        Handler::with_tty_emitter(ColorConfig::Auto, true, false, Some(codemap.clone()));
    let mut parse_session = ParseSess::with_span_handler(tty_handler, codemap.clone());

    let main_file = match input {
        Input::File(ref file) => file.clone(),
        Input::Text(..) => PathBuf::from("stdin"),
    };

    let krate = match parse_input(input, &parse_session) {
        Ok(krate) => krate,
        Err(diagnostic) => {
            if let Some(mut diagnostic) = diagnostic {
                diagnostic.emit();
            }
            summary.add_parsing_error();
            return Ok((summary, FileMap::new(), FormatReport::new()));
        }
    };

    if parse_session.span_diagnostic.has_errors() {
        summary.add_parsing_error();
    }

    // Suppress error output after parsing.
    let silent_emitter = Box::new(EmitterWriter::new(
        Box::new(Vec::new()),
        Some(codemap.clone()),
        false,
    ));
    parse_session.span_diagnostic = Handler::with_emitter(true, false, silent_emitter);

    let mut report = FormatReport::new();

    match format_ast(
        &krate,
        &mut parse_session,
        &main_file,
        config,
        |file_name, file, skipped_range| {
            // For some reason, the codemap does not include terminating
            // newlines so we must add one on for each file. This is sad.
            filemap::append_newline(file);

            format_lines(file, file_name, skipped_range, config, &mut report);

            if let Some(ref mut out) = out {
                return filemap::write_file(file, file_name, out, config);
            }
            Ok(false)
        },
    ) {
        Ok((file_map, has_diff)) => {
            if report.has_warnings() {
                summary.add_formatting_error();
            }

            if has_diff {
                summary.add_diff();
            }

            Ok((summary, file_map, report))
        }
        Err(e) => Err((e, summary)),
    }
}

#[derive(Debug)]
pub enum Input {
    File(PathBuf),
    Text(String),
}

pub fn run(input: Input, config: &Config) -> Summary {
    let out = &mut stdout();
    output_header(out, config.write_mode()).ok();
    match format_input(input, config, Some(out)) {
        Ok((summary, _, report)) => {
            output_footer(out, config.write_mode()).ok();

            if report.has_warnings() {
                match term::stderr() {
                    Some(ref t)
                        if use_colored_tty(config.color()) && t.supports_color()
                            && t.supports_attr(term::Attr::Bold) =>
                    {
                        match report.print_warnings_fancy(term::stderr().unwrap()) {
                            Ok(..) => (),
                            Err(..) => panic!("Unable to write to stderr: {}", report),
                        }
                    }
                    _ => msg!("{}", report),
                }
            }

            summary
        }
        Err((msg, mut summary)) => {
            msg!("Error writing files: {}", msg);
            summary.add_operational_error();
            summary
        }
    }
}
