// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// TODO we're going to allocate a whole bunch of temp Strings, is it worth
// keeping some scratch mem for this and running our own StrPool?
// TODO for lint violations of names, emit a refactor script

#[macro_use]
extern crate log;

extern crate syntex_syntax as syntax;
extern crate rustc_serialize;

extern crate strings;

extern crate unicode_segmentation;
extern crate regex;
extern crate diff;
extern crate term;

use syntax::ast;
use syntax::codemap::{mk_sp, CodeMap, Span};
use syntax::errors::{Handler, DiagnosticBuilder};
use syntax::errors::emitter::{ColorConfig, EmitterWriter};
use syntax::parse::{self, ParseSess};

use std::io::{stdout, Write};
use std::ops::{Add, Sub};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::collections::HashMap;
use std::fmt;

use issues::{BadIssueSeeker, Issue};
use filemap::FileMap;
use visitor::FmtVisitor;
use config::Config;

pub use self::summary::Summary;

#[macro_use]
mod utils;
pub mod config;
pub mod filemap;
pub mod visitor;
mod checkstyle;
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

const MIN_STRING: usize = 10;
// When we get scoped annotations, we should have rustfmt::skip.
const SKIP_ANNOTATION: &'static str = "rustfmt_skip";

pub trait Spanned {
    fn span(&self) -> Span;
}

impl Spanned for ast::Expr {
    fn span(&self) -> Span {
        self.span
    }
}

impl Spanned for ast::Pat {
    fn span(&self) -> Span {
        self.span
    }
}

impl Spanned for ast::Ty {
    fn span(&self) -> Span {
        self.span
    }
}

impl Spanned for ast::Arg {
    fn span(&self) -> Span {
        if items::is_named_arg(self) {
            mk_sp(self.pat.span.lo, self.ty.span.hi)
        } else {
            self.ty.span
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Indent {
    // Width of the block indent, in characters. Must be a multiple of
    // Config::tab_spaces.
    pub block_indent: usize,
    // Alignment in characters.
    pub alignment: usize,
}

impl Indent {
    pub fn new(block_indent: usize, alignment: usize) -> Indent {
        Indent {
            block_indent: block_indent,
            alignment: alignment,
        }
    }

    pub fn empty() -> Indent {
        Indent::new(0, 0)
    }

    pub fn block_indent(mut self, config: &Config) -> Indent {
        self.block_indent += config.tab_spaces;
        self
    }

    pub fn block_unindent(mut self, config: &Config) -> Indent {
        self.block_indent -= config.tab_spaces;
        self
    }

    pub fn width(&self) -> usize {
        self.block_indent + self.alignment
    }

    pub fn to_string(&self, config: &Config) -> String {
        let (num_tabs, num_spaces) = if config.hard_tabs {
            (self.block_indent / config.tab_spaces, self.alignment)
        } else {
            (0, self.block_indent + self.alignment)
        };
        let num_chars = num_tabs + num_spaces;
        let mut indent = String::with_capacity(num_chars);
        for _ in 0..num_tabs {
            indent.push('\t')
        }
        for _ in 0..num_spaces {
            indent.push(' ')
        }
        indent
    }
}

impl Add for Indent {
    type Output = Indent;

    fn add(self, rhs: Indent) -> Indent {
        Indent {
            block_indent: self.block_indent + rhs.block_indent,
            alignment: self.alignment + rhs.alignment,
        }
    }
}

impl Sub for Indent {
    type Output = Indent;

    fn sub(self, rhs: Indent) -> Indent {
        Indent::new(self.block_indent - rhs.block_indent,
                    self.alignment - rhs.alignment)
    }
}

impl Add<usize> for Indent {
    type Output = Indent;

    fn add(self, rhs: usize) -> Indent {
        Indent::new(self.block_indent, self.alignment + rhs)
    }
}

impl Sub<usize> for Indent {
    type Output = Indent;

    fn sub(self, rhs: usize) -> Indent {
        Indent::new(self.block_indent, self.alignment - rhs)
    }
}

pub enum ErrorKind {
    // Line has exceeded character limit
    LineOverflow,
    // Line ends in whitespace
    TrailingWhitespace,
    // TO-DO or FIX-ME item without an issue number
    BadIssue(Issue),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            ErrorKind::LineOverflow => write!(fmt, "line exceeded maximum length"),
            ErrorKind::TrailingWhitespace => write!(fmt, "left behind trailing whitespace"),
            ErrorKind::BadIssue(issue) => write!(fmt, "found {}", issue),
        }
    }
}

// Formatting errors that are identified *after* rustfmt has run.
pub struct FormattingError {
    line: u32,
    kind: ErrorKind,
}

impl FormattingError {
    fn msg_prefix(&self) -> &str {
        match self.kind {
            ErrorKind::LineOverflow |
            ErrorKind::TrailingWhitespace => "Rustfmt failed at",
            ErrorKind::BadIssue(_) => "WARNING:",
        }
    }

    fn msg_suffix(&self) -> &str {
        match self.kind {
            ErrorKind::LineOverflow |
            ErrorKind::TrailingWhitespace => "(sorry)",
            ErrorKind::BadIssue(_) => "",
        }
    }
}

pub struct FormatReport {
    // Maps stringified file paths to their associated formatting errors.
    file_error_map: HashMap<String, Vec<FormattingError>>,
}

impl FormatReport {
    fn new() -> FormatReport {
        FormatReport { file_error_map: HashMap::new() }
    }

    pub fn warning_count(&self) -> usize {
        self.file_error_map.iter().map(|(_, ref errors)| errors.len()).fold(0, |acc, x| acc + x)
    }

    pub fn has_warnings(&self) -> bool {
        self.warning_count() > 0
    }
}

impl fmt::Display for FormatReport {
    // Prints all the formatting errors.
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for (file, errors) in &self.file_error_map {
            for error in errors {
                try!(write!(fmt,
                            "{} {}:{}: {} {}\n",
                            error.msg_prefix(),
                            file,
                            error.line,
                            error.kind,
                            error.msg_suffix()));
            }
        }
        Ok(())
    }
}

// Formatting which depends on the AST.
fn format_ast(krate: &ast::Crate,
              parse_session: &ParseSess,
              main_file: &Path,
              config: &Config)
              -> FileMap {
    let mut file_map = FileMap::new();
    // We always skip children for the "Plain" write mode, since there is
    // nothing to distinguish the nested module contents.
    let skip_children = config.skip_children || config.write_mode == config::WriteMode::Plain;
    for (path, module) in modules::list_files(krate, parse_session.codemap()) {
        if skip_children && path.as_path() != main_file {
            continue;
        }
        let path = path.to_str().unwrap();
        if config.verbose {
            println!("Formatting {}", path);
        }
        let mut visitor = FmtVisitor::from_codemap(parse_session, config);
        visitor.format_separate_mod(module);
        file_map.insert(path.to_owned(), visitor.buffer);
    }
    file_map
}

// Formatting done on a char by char or line by line basis.
// TODO(#209) warn on bad license
// TODO(#20) other stuff for parity with make tidy
fn format_lines(file_map: &mut FileMap, config: &Config) -> FormatReport {
    let mut truncate_todo = Vec::new();
    let mut report = FormatReport::new();

    // Iterate over the chars in the file map.
    for (f, text) in file_map.iter() {
        let mut trims = vec![];
        let mut last_wspace: Option<usize> = None;
        let mut line_len = 0;
        let mut cur_line = 1;
        let mut newline_count = 0;
        let mut errors = vec![];
        let mut issue_seeker = BadIssueSeeker::new(config.report_todo, config.report_fixme);

        for (c, b) in text.chars() {
            if c == '\r' {
                line_len += c.len_utf8();
                continue;
            }

            // Add warnings for bad todos/ fixmes
            if let Some(issue) = issue_seeker.inspect(c) {
                errors.push(FormattingError {
                    line: cur_line,
                    kind: ErrorKind::BadIssue(issue),
                });
            }

            if c == '\n' {
                // Check for (and record) trailing whitespace.
                if let Some(lw) = last_wspace {
                    trims.push((cur_line, lw, b));
                    line_len -= b - lw;
                }
                // Check for any line width errors we couldn't correct.
                if line_len > config.max_width {
                    errors.push(FormattingError {
                        line: cur_line,
                        kind: ErrorKind::LineOverflow,
                    });
                }
                line_len = 0;
                cur_line += 1;
                newline_count += 1;
                last_wspace = None;
            } else {
                newline_count = 0;
                line_len += c.len_utf8();
                if c.is_whitespace() {
                    if last_wspace.is_none() {
                        last_wspace = Some(b);
                    }
                } else {
                    last_wspace = None;
                }
            }
        }

        if newline_count > 1 {
            debug!("track truncate: {} {} {}", f, text.len, newline_count);
            truncate_todo.push((f.to_owned(), text.len - newline_count + 1))
        }

        for &(l, _, _) in &trims {
            errors.push(FormattingError {
                line: l,
                kind: ErrorKind::TrailingWhitespace,
            });
        }

        report.file_error_map.insert(f.to_owned(), errors);
    }

    for (f, l) in truncate_todo {
        file_map.get_mut(&f).unwrap().truncate(l);
    }

    report
}

fn parse_input(input: Input,
               parse_session: &ParseSess)
               -> Result<ast::Crate, Option<DiagnosticBuilder>> {
    let result = match input {
        Input::File(file) => parse::parse_crate_from_file(&file, Vec::new(), &parse_session),
        Input::Text(text) => {
            parse::parse_crate_from_source_str("stdin".to_owned(), text, Vec::new(), &parse_session)
        }
    };

    // Bail out if the parser recovered from an error.
    if parse_session.span_diagnostic.has_errors() {
        return Err(None);
    }

    result.map_err(|e| Some(e))
}

pub fn format_input(input: Input, config: &Config) -> (Summary, FileMap, FormatReport) {
    let mut summary = Summary::new();
    let codemap = Rc::new(CodeMap::new());

    let tty_handler = Handler::with_tty_emitter(ColorConfig::Auto,
                                                None,
                                                true,
                                                false,
                                                codemap.clone());
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
            return (summary, FileMap::new(), FormatReport::new());
        }
    };

    if parse_session.span_diagnostic.has_errors() {
        summary.add_parsing_error();
    }

    // Suppress error output after parsing.
    let silent_emitter = Box::new(EmitterWriter::new(Box::new(Vec::new()), None, codemap.clone()));
    parse_session.span_diagnostic = Handler::with_emitter(true, false, silent_emitter);

    let mut file_map = format_ast(&krate, &parse_session, &main_file, config);

    // For some reason, the codemap does not include terminating
    // newlines so we must add one on for each file. This is sad.
    filemap::append_newlines(&mut file_map);

    let report = format_lines(&mut file_map, config);
    if report.has_warnings() {
        summary.add_formatting_error();
    }
    (summary, file_map, report)
}

pub enum Input {
    File(PathBuf),
    Text(String),
}

pub fn run(input: Input, config: &Config) -> Summary {
    let (mut summary, file_map, report) = format_input(input, config);
    if report.has_warnings() {
        msg!("{}", report);
    }

    let mut out = stdout();
    let write_result = filemap::write_all_files(&file_map, &mut out, config);

    if let Err(msg) = write_result {
        msg!("Error writing files: {}", msg);
        summary.add_operational_error();
    }

    summary
}
