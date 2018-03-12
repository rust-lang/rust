// Copyright 2015-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(custom_attribute)]
#![feature(decl_macro)]
#![feature(match_default_bindings)]
#![feature(type_ascription)]

#[macro_use]
extern crate derive_new;
extern crate diff;
extern crate itertools;
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
extern crate toml;
extern crate unicode_segmentation;

use std::collections::HashMap;
use std::fmt;
use std::io::{self, stdout, BufRead, Write};
use std::iter::repeat;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use errors::{DiagnosticBuilder, Handler};
use errors::emitter::{ColorConfig, EmitterWriter};
use syntax::ast;
use syntax::codemap::{CodeMap, FilePathMapping};
pub use syntax::codemap::FileName;
use syntax::parse::{self, ParseSess};

use checkstyle::{output_footer, output_header};
use comment::{CharClasses, FullCodeCharKind};
use issues::{BadIssueSeeker, Issue};
use shape::Indent;
use utils::use_colored_tty;
use visitor::{FmtVisitor, SnippetProvider};

pub use config::Config;
pub use config::summary::Summary;

#[macro_use]
mod utils;

mod attr;
mod chains;
mod checkstyle;
mod closures;
pub mod codemap;
mod comment;
pub mod config;
mod expr;
pub mod filemap;
mod imports;
mod issues;
mod items;
mod lists;
mod macros;
mod missed_spans;
pub mod modules;
mod overflow;
mod patterns;
mod reorder;
mod rewrite;
pub mod rustfmt_diff;
mod shape;
mod spanned;
mod string;
mod types;
mod vertical;
pub mod visitor;

const STDIN: &str = "<stdin>";

// A map of the files of a crate, with their new content
pub type FileMap = Vec<FileRecord>;

pub type FileRecord = (FileName, String);

#[derive(Clone, Copy)]
pub enum ErrorKind {
    // Line has exceeded character limit (found, maximum)
    LineOverflow(usize, usize),
    // Line ends in whitespace
    TrailingWhitespace,
    // TO-DO or FIX-ME item without an issue number
    BadIssue(Issue),
    // License check has failed
    LicenseCheck,
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
            ErrorKind::LicenseCheck => write!(fmt, "license check failed"),
        }
    }
}

// Formatting errors that are identified *after* rustfmt has run.
pub struct FormattingError {
    line: usize,
    kind: ErrorKind,
    is_comment: bool,
    is_string: bool,
    line_buffer: String,
}

impl FormattingError {
    fn msg_prefix(&self) -> &str {
        match self.kind {
            ErrorKind::LineOverflow(..)
            | ErrorKind::TrailingWhitespace
            | ErrorKind::LicenseCheck => "error:",
            ErrorKind::BadIssue(_) => "WARNING:",
        }
    }

    fn msg_suffix(&self) -> &str {
        if self.is_comment || self.is_string {
            "set `error_on_unformatted = false` to suppress \
             the warning against comments or string literals\n"
        } else {
            ""
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
    file_error_map: HashMap<FileName, Vec<FormattingError>>,
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
            .sum()
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

fn should_emit_verbose<F>(path: &FileName, config: &Config, f: F)
where
    F: Fn(),
{
    if config.verbose() && path.to_string() != STDIN {
        f();
    }
}

// Formatting which depends on the AST.
fn format_ast<F>(
    krate: &ast::Crate,
    parse_session: &mut ParseSess,
    main_file: &FileName,
    config: &Config,
    mut after_file: F,
) -> Result<(FileMap, bool), io::Error>
where
    F: FnMut(&FileName, &mut String, &[(usize, usize)]) -> Result<bool, io::Error>,
{
    let mut result = FileMap::new();
    // diff mode: check if any files are differing
    let mut has_diff = false;

    // We always skip children for the "Plain" write mode, since there is
    // nothing to distinguish the nested module contents.
    let skip_children = config.skip_children() || config.write_mode() == config::WriteMode::Plain;
    for (path, module) in modules::list_files(krate, parse_session.codemap())? {
        if (skip_children && path != *main_file) || config.ignore().skip_file(&path) {
            continue;
        }
        should_emit_verbose(&path, config, || println!("Formatting {}", path));
        let filemap = parse_session
            .codemap()
            .lookup_char_pos(module.inner.lo())
            .file;
        let big_snippet = filemap.src.as_ref().unwrap();
        let snippet_provider = SnippetProvider::new(filemap.start_pos, big_snippet);
        let mut visitor = FmtVisitor::from_codemap(parse_session, config, &snippet_provider);
        // Format inner attributes if available.
        if !krate.attrs.is_empty() && path == *main_file {
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

        debug_assert_eq!(
            visitor.line_number,
            ::utils::count_newlines(&format!("{}", visitor.buffer))
        );

        let filename = path.clone();
        has_diff |= match after_file(&filename, &mut visitor.buffer, &visitor.skipped_range) {
            Ok(result) => result,
            Err(e) => {
                // Create a new error with path_str to help users see which files failed
                let err_msg = format!("{}: {}", path, e);
                return Err(io::Error::new(e.kind(), err_msg));
            }
        };

        result.push((filename, visitor.buffer));
    }

    Ok((result, has_diff))
}

/// Returns true if the line with the given line number was skipped by `#[rustfmt_skip]`.
fn is_skipped_line(line_number: usize, skipped_range: &[(usize, usize)]) -> bool {
    skipped_range
        .iter()
        .any(|&(lo, hi)| lo <= line_number && line_number <= hi)
}

fn should_report_error(
    config: &Config,
    char_kind: FullCodeCharKind,
    is_string: bool,
    error_kind: ErrorKind,
) -> bool {
    let allow_error_report = if char_kind.is_comment() || is_string {
        config.error_on_unformatted()
    } else {
        true
    };

    match error_kind {
        ErrorKind::LineOverflow(..) => config.error_on_line_overflow() && allow_error_report,
        ErrorKind::TrailingWhitespace => allow_error_report,
        _ => true,
    }
}

// Formatting done on a char by char or line by line basis.
// FIXME(#20) other stuff for parity with make tidy
fn format_lines(
    text: &mut String,
    name: &FileName,
    skipped_range: &[(usize, usize)],
    config: &Config,
    report: &mut FormatReport,
) {
    let mut trims = vec![];
    let mut last_wspace: Option<usize> = None;
    let mut line_len = 0;
    let mut cur_line = 1;
    let mut newline_count = 0;
    let mut errors = vec![];
    let mut issue_seeker = BadIssueSeeker::new(config.report_todo(), config.report_fixme());
    let mut line_buffer = String::with_capacity(config.max_width() * 2);
    let mut is_string = false; // true if the current line contains a string literal.
    let mut format_line = config.file_lines().contains_line(name, cur_line);
    let allow_issue_seek = !issue_seeker.is_disabled();

    // Check license.
    if let Some(ref license_template) = config.license_template {
        if !license_template.is_match(text) {
            errors.push(FormattingError {
                line: cur_line,
                kind: ErrorKind::LicenseCheck,
                is_comment: false,
                is_string: false,
                line_buffer: String::new(),
            });
        }
    }

    // Iterate over the chars in the file map.
    for (kind, (b, c)) in CharClasses::new(text.chars().enumerate()) {
        if c == '\r' {
            continue;
        }

        if allow_issue_seek && format_line {
            // Add warnings for bad todos/ fixmes
            if let Some(issue) = issue_seeker.inspect(c) {
                errors.push(FormattingError {
                    line: cur_line,
                    kind: ErrorKind::BadIssue(issue),
                    is_comment: false,
                    is_string: false,
                    line_buffer: String::new(),
                });
            }
        }

        if c == '\n' {
            if format_line {
                // Check for (and record) trailing whitespace.
                if let Some(..) = last_wspace {
                    if should_report_error(config, kind, is_string, ErrorKind::TrailingWhitespace) {
                        trims.push((cur_line, kind, line_buffer.clone()));
                    }
                    line_len -= 1;
                }

                // Check for any line width errors we couldn't correct.
                let error_kind = ErrorKind::LineOverflow(line_len, config.max_width());
                if line_len > config.max_width() && !is_skipped_line(cur_line, skipped_range)
                    && should_report_error(config, kind, is_string, error_kind)
                {
                    errors.push(FormattingError {
                        line: cur_line,
                        kind: error_kind,
                        is_comment: kind.is_comment(),
                        is_string,
                        line_buffer: line_buffer.clone(),
                    });
                }
            }

            line_len = 0;
            cur_line += 1;
            format_line = config.file_lines().contains_line(name, cur_line);
            newline_count += 1;
            last_wspace = None;
            line_buffer.clear();
            is_string = false;
        } else {
            newline_count = 0;
            line_len += if c == '\t' { config.tab_spaces() } else { 1 };
            if c.is_whitespace() {
                if last_wspace.is_none() {
                    last_wspace = Some(b);
                }
            } else {
                last_wspace = None;
            }
            line_buffer.push(c);
            if kind.is_string() {
                is_string = true;
            }
        }
    }

    if newline_count > 1 {
        debug!("track truncate: {} {}", text.len(), newline_count);
        let line = text.len() - newline_count + 1;
        text.truncate(line);
    }

    for &(l, kind, ref b) in &trims {
        if !is_skipped_line(l, skipped_range) {
            errors.push(FormattingError {
                line: l,
                kind: ErrorKind::TrailingWhitespace,
                is_comment: kind.is_comment(),
                is_string: kind.is_string(),
                line_buffer: b.clone(),
            });
        }
    }

    report.file_error_map.insert(name.clone(), errors);
}

fn parse_input<'sess>(
    input: Input,
    parse_session: &'sess ParseSess,
    config: &Config,
) -> Result<ast::Crate, ParseError<'sess>> {
    let mut parser = match input {
        Input::File(file) => parse::new_parser_from_file(parse_session, &file),
        Input::Text(text) => parse::new_parser_from_source_str(
            parse_session,
            FileName::Custom("stdin".to_owned()),
            text,
        ),
    };

    parser.cfg_mods = false;
    if config.skip_children() {
        parser.recurse_into_file_modules = false;
    }

    let mut parser = AssertUnwindSafe(parser);
    let result = catch_unwind(move || parser.0.parse_crate_mod());

    match result {
        Ok(Ok(c)) => {
            if parse_session.span_diagnostic.has_errors() {
                // Bail out if the parser recovered from an error.
                Err(ParseError::Recovered)
            } else {
                Ok(c)
            }
        }
        Ok(Err(e)) => Err(ParseError::Error(e)),
        Err(_) => Err(ParseError::Panic),
    }
}

/// All the ways that parsing can fail.
enum ParseError<'sess> {
    /// There was an error, but the parser recovered.
    Recovered,
    /// There was an error (supplied) and parsing failed.
    Error(DiagnosticBuilder<'sess>),
    /// The parser panicked.
    Panic,
}

/// Format the given snippet. The snippet is expected to be *complete* code.
/// When we cannot parse the given snippet, this function returns `None`.
pub fn format_snippet(snippet: &str, config: &Config) -> Option<String> {
    let mut out: Vec<u8> = Vec::with_capacity(snippet.len() * 2);
    let input = Input::Text(snippet.into());
    let mut config = config.clone();
    config.set().write_mode(config::WriteMode::Plain);
    config.set().hide_parse_errors(true);
    match format_input(input, &config, Some(&mut out)) {
        // `format_input()` returns an empty string on parsing error.
        Ok(..) if out.is_empty() && !snippet.is_empty() => None,
        Ok(..) => String::from_utf8(out).ok(),
        Err(..) => None,
    }
}

const FN_MAIN_PREFIX: &str = "fn main() {\n";

fn enclose_in_main_block(s: &str, config: &Config) -> String {
    let indent = Indent::from_width(config, config.tab_spaces());
    FN_MAIN_PREFIX.to_owned() + &indent.to_string(config)
        + &s.lines()
            .collect::<Vec<_>>()
            .join(&indent.to_string_with_newline(config)) + "\n}"
}

/// Format the given code block. Mainly targeted for code block in comment.
/// The code block may be incomplete (i.e. parser may be unable to parse it).
/// To avoid panic in parser, we wrap the code block with a dummy function.
/// The returned code block does *not* end with newline.
pub fn format_code_block(code_snippet: &str, config: &Config) -> Option<String> {
    // Wrap the given code block with `fn main()` if it does not have one.
    let snippet = enclose_in_main_block(code_snippet, config);
    let mut result = String::with_capacity(snippet.len());
    let mut is_first = true;

    // Trim "fn main() {" on the first line and "}" on the last line,
    // then unindent the whole code block.
    let formatted = format_snippet(&snippet, config)?;
    // 2 = "}\n"
    let block_len = formatted.len().checked_sub(2).unwrap_or(0);
    for line in formatted[FN_MAIN_PREFIX.len()..block_len].lines() {
        if !is_first {
            result.push('\n');
        } else {
            is_first = false;
        }
        let trimmed_line = if line.len() > config.max_width() {
            // If there are lines that are larger than max width, we cannot tell
            // whether we have succeeded but have some comments or strings that
            // are too long, or we have failed to format code block. We will be
            // conservative and just return `None` in this case.
            return None;
        } else if line.len() > config.tab_spaces() {
            // Make sure that the line has leading whitespaces.
            let indent_str = Indent::from_width(config, config.tab_spaces()).to_string(config);
            if line.starts_with(indent_str.as_ref()) {
                let offset = if config.hard_tabs() {
                    1
                } else {
                    config.tab_spaces()
                };
                &line[offset..]
            } else {
                line
            }
        } else {
            line
        };
        result.push_str(trimmed_line);
    }
    Some(result)
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

    let tty_handler = if config.hide_parse_errors() {
        let silent_emitter = Box::new(EmitterWriter::new(
            Box::new(Vec::new()),
            Some(codemap.clone()),
            false,
            false,
        ));
        Handler::with_emitter(true, false, silent_emitter)
    } else {
        let supports_color = term::stderr().map_or(false, |term| term.supports_color());
        let color_cfg = if supports_color {
            ColorConfig::Auto
        } else {
            ColorConfig::Never
        };
        Handler::with_tty_emitter(color_cfg, true, false, Some(codemap.clone()))
    };
    let mut parse_session = ParseSess::with_span_handler(tty_handler, codemap.clone());

    let main_file = match input {
        Input::File(ref file) => FileName::Real(file.clone()),
        Input::Text(..) => FileName::Custom("stdin".to_owned()),
    };

    let krate = match parse_input(input, &parse_session, config) {
        Ok(krate) => krate,
        Err(err) => {
            match err {
                ParseError::Error(mut diagnostic) => diagnostic.emit(),
                ParseError::Panic => {
                    // Note that if you see this message and want more information,
                    // then go to `parse_input` and run the parse function without
                    // `catch_unwind` so rustfmt panics and you can get a backtrace.
                    should_emit_verbose(&main_file, config, || {
                        println!("The Rust parser panicked")
                    });
                }
                ParseError::Recovered => {}
            }
            summary.add_parsing_error();
            return Ok((summary, FileMap::new(), FormatReport::new()));
        }
    };

    summary.mark_parse_time();

    // Suppress error output after parsing.
    let silent_emitter = Box::new(EmitterWriter::new(
        Box::new(Vec::new()),
        Some(codemap.clone()),
        false,
        false,
    ));
    parse_session.span_diagnostic = Handler::with_emitter(true, false, silent_emitter);

    let mut report = FormatReport::new();

    let format_result = format_ast(
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
    );

    summary.mark_format_time();

    should_emit_verbose(&main_file, config, || {
        fn duration_to_f32(d: Duration) -> f32 {
            d.as_secs() as f32 + d.subsec_nanos() as f32 / 1_000_000_000f32
        }

        println!(
            "Spent {0:.3} secs in the parsing phase, and {1:.3} secs in the formatting phase",
            duration_to_f32(summary.get_parse_time().unwrap()),
            duration_to_f32(summary.get_format_time().unwrap()),
        )
    });

    match format_result {
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

/// A single span of changed lines, with 0 or more removed lines
/// and a vector of 0 or more inserted lines.
#[derive(Debug, PartialEq, Eq)]
pub struct ModifiedChunk {
    /// The first to be removed from the original text
    pub line_number_orig: u32,
    /// The number of lines which have been replaced
    pub lines_removed: u32,
    /// The new lines
    pub lines: Vec<String>,
}

/// Set of changed sections of a file.
#[derive(Debug, PartialEq, Eq)]
pub struct ModifiedLines {
    /// The set of changed chunks.
    pub chunks: Vec<ModifiedChunk>,
}

/// The successful result of formatting via `get_modified_lines()`.
pub struct ModifiedLinesResult {
    /// The high level summary details
    pub summary: Summary,
    /// The result Filemap
    pub filemap: FileMap,
    /// Map of formatting errors
    pub report: FormatReport,
    /// The sets of updated lines.
    pub modified_lines: ModifiedLines,
}

/// Format a file and return a `ModifiedLines` data structure describing
/// the changed ranges of lines.
pub fn get_modified_lines(
    input: Input,
    config: &Config,
) -> Result<ModifiedLinesResult, (io::Error, Summary)> {
    let mut data = Vec::new();

    let mut config = config.clone();
    config.set().write_mode(config::WriteMode::Modified);
    let (summary, filemap, report) = format_input(input, &config, Some(&mut data))?;

    let mut lines = data.lines();
    let mut chunks = Vec::new();
    while let Some(Ok(header)) = lines.next() {
        // Parse the header line
        let values: Vec<_> = header
            .split(' ')
            .map(|s| s.parse::<u32>().unwrap())
            .collect();
        assert_eq!(values.len(), 3);
        let line_number_orig = values[0];
        let lines_removed = values[1];
        let num_added = values[2];
        let mut added_lines = Vec::new();
        for _ in 0..num_added {
            added_lines.push(lines.next().unwrap().unwrap());
        }
        chunks.push(ModifiedChunk {
            line_number_orig,
            lines_removed,
            lines: added_lines,
        });
    }
    Ok(ModifiedLinesResult {
        summary,
        filemap,
        report,
        modified_lines: ModifiedLines { chunks },
    })
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

#[cfg(test)]
mod test {
    use super::{format_code_block, format_snippet, Config};

    #[test]
    fn test_no_panic_on_format_snippet_and_format_code_block() {
        // `format_snippet()` and `format_code_block()` should not panic
        // even when we cannot parse the given snippet.
        let snippet = "let";
        assert!(format_snippet(snippet, &Config::default()).is_none());
        assert!(format_code_block(snippet, &Config::default()).is_none());
    }

    fn test_format_inner<F>(formatter: F, input: &str, expected: &str) -> bool
    where
        F: Fn(&str, &Config) -> Option<String>,
    {
        let output = formatter(input, &Config::default());
        output.is_some() && output.unwrap() == expected
    }

    #[test]
    fn test_format_snippet() {
        let snippet = "fn main() { println!(\"hello, world\"); }";
        let expected = "fn main() {\n    \
                        println!(\"hello, world\");\n\
                        }\n";
        assert!(test_format_inner(format_snippet, snippet, expected));
    }

    #[test]
    fn test_format_code_block_fail() {
        #[rustfmt_skip]
        let code_block = "this_line_is_100_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(x, y, z);";
        assert!(format_code_block(code_block, &Config::default()).is_none());
    }

    #[test]
    fn test_format_code_block() {
        // simple code block
        let code_block = "let x=3;";
        let expected = "let x = 3;";
        assert!(test_format_inner(format_code_block, code_block, expected));

        // more complex code block, taken from chains.rs.
        let code_block =
"let (nested_shape, extend) = if !parent_rewrite_contains_newline && is_continuable(&parent) {
(
chain_indent(context, shape.add_offset(parent_rewrite.len())),
context.config.indent_style() == IndentStyle::Visual || is_small_parent,
)
} else if is_block_expr(context, &parent, &parent_rewrite) {
match context.config.indent_style() {
// Try to put the first child on the same line with parent's last line
IndentStyle::Block => (parent_shape.block_indent(context.config.tab_spaces()), true),
// The parent is a block, so align the rest of the chain with the closing
// brace.
IndentStyle::Visual => (parent_shape, false),
}
} else {
(
chain_indent(context, shape.add_offset(parent_rewrite.len())),
false,
)
};
";
        let expected =
"let (nested_shape, extend) = if !parent_rewrite_contains_newline && is_continuable(&parent) {
    (
        chain_indent(context, shape.add_offset(parent_rewrite.len())),
        context.config.indent_style() == IndentStyle::Visual || is_small_parent,
    )
} else if is_block_expr(context, &parent, &parent_rewrite) {
    match context.config.indent_style() {
        // Try to put the first child on the same line with parent's last line
        IndentStyle::Block => (parent_shape.block_indent(context.config.tab_spaces()), true),
        // The parent is a block, so align the rest of the chain with the closing
        // brace.
        IndentStyle::Visual => (parent_shape, false),
    }
} else {
    (
        chain_indent(context, shape.add_offset(parent_rewrite.len())),
        false,
    )
};";
        assert!(test_format_inner(format_code_block, code_block, expected));
    }
}
