// Copyright 2015-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(tool_attributes)]
#![feature(decl_macro)]
#![allow(unused_attributes)]
#![feature(type_ascription)]
#![feature(unicode_internals)]
#![feature(extern_prelude)]
#![feature(nll)]

#[macro_use]
extern crate derive_new;
extern crate diff;
extern crate failure;
extern crate isatty;
extern crate itertools;
#[cfg(test)]
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate regex;
extern crate rustc_target;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate syntax;
extern crate syntax_pos;
extern crate toml;
extern crate unicode_segmentation;

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};
use std::mem;
use std::path::PathBuf;
use std::rc::Rc;
use syntax::ast;

use comment::LineClasses;
use failure::Fail;
use formatting::{FileMap, FormatErrorMap, FormattingError, ReportedErrors, Summary};
use issues::Issue;
use shape::Indent;

pub use config::{
    load_config, CliOptions, Color, Config, EmitMode, FileLines, FileName, NewlineStyle, Range,
    Verbosity,
};

#[macro_use]
mod utils;

mod attr;
mod chains;
pub(crate) mod checkstyle;
mod closures;
pub(crate) mod codemap;
mod comment;
pub(crate) mod config;
mod expr;
pub(crate) mod filemap;
pub(crate) mod formatting;
mod imports;
mod issues;
mod items;
mod lists;
mod macros;
mod matches;
mod missed_spans;
pub(crate) mod modules;
mod overflow;
mod pairs;
mod patterns;
mod reorder;
mod rewrite;
pub(crate) mod rustfmt_diff;
mod shape;
mod spanned;
mod string;
#[cfg(test)]
mod test;
mod types;
mod vertical;
pub(crate) mod visitor;

/// The various errors that can occur during formatting. Note that not all of
/// these can currently be propagated to clients.
#[derive(Fail, Debug)]
pub enum ErrorKind {
    /// Line has exceeded character limit (found, maximum).
    #[fail(
        display = "line formatted, but exceeded maximum width \
                   (maximum: {} (see `max_width` option), found: {})",
        _0,
        _1
    )]
    LineOverflow(usize, usize),
    /// Line ends in whitespace.
    #[fail(display = "left behind trailing whitespace")]
    TrailingWhitespace,
    /// TODO or FIXME item without an issue number.
    #[fail(display = "found {}", _0)]
    BadIssue(Issue),
    /// License check has failed.
    #[fail(display = "license check failed")]
    LicenseCheck,
    /// Used deprecated skip attribute.
    #[fail(display = "`rustfmt_skip` is deprecated; use `rustfmt::skip`")]
    DeprecatedAttr,
    /// Used a rustfmt:: attribute other than skip.
    #[fail(display = "invalid attribute")]
    BadAttr,
    /// An io error during reading or writing.
    #[fail(display = "io error: {}", _0)]
    IoError(io::Error),
    /// Parse error occured when parsing the Input.
    #[fail(display = "parse error")]
    ParseError,
    /// The user mandated a version and the current version of Rustfmt does not
    /// satisfy that requirement.
    #[fail(display = "version mismatch")]
    VersionMismatch,
    /// If we had formatted the given node, then we would have lost a comment.
    #[fail(display = "not formatted because a comment would be lost")]
    LostComment,
}

impl ErrorKind {
    fn is_comment(&self) -> bool {
        match self {
            ErrorKind::LostComment => true,
            _ => false,
        }
    }
}

impl From<io::Error> for ErrorKind {
    fn from(e: io::Error) -> ErrorKind {
        ErrorKind::IoError(e)
    }
}

/// Reports on any issues that occurred during a run of Rustfmt.
///
/// Can be reported to the user via its `Display` implementation of `print_fancy`.
#[derive(Clone)]
pub struct FormatReport {
    // Maps stringified file paths to their associated formatting errors.
    internal: Rc<RefCell<(FormatErrorMap, ReportedErrors)>>,
}

impl FormatReport {
    fn new() -> FormatReport {
        FormatReport {
            internal: Rc::new(RefCell::new((HashMap::new(), ReportedErrors::default()))),
        }
    }

    fn append(&self, f: FileName, mut v: Vec<FormattingError>) {
        self.track_errors(&v);
        self.internal
            .borrow_mut()
            .0
            .entry(f)
            .and_modify(|fe| fe.append(&mut v))
            .or_insert(v);
    }

    fn track_errors(&self, new_errors: &[FormattingError]) {
        let errs = &mut self.internal.borrow_mut().1;
        if errs.has_operational_errors && errs.has_check_errors {
            return;
        }
        for err in new_errors {
            match err.kind {
                ErrorKind::LineOverflow(..) | ErrorKind::TrailingWhitespace => {
                    errs.has_operational_errors = true;
                }
                ErrorKind::BadIssue(_)
                | ErrorKind::LicenseCheck
                | ErrorKind::DeprecatedAttr
                | ErrorKind::BadAttr
                | ErrorKind::VersionMismatch => {
                    errs.has_check_errors = true;
                }
                _ => {}
            }
        }
    }

    fn warning_count(&self) -> usize {
        self.internal
            .borrow()
            .0
            .iter()
            .map(|(_, errors)| errors.len())
            .sum()
    }

    /// Whether any warnings or errors are present in the report.
    pub fn has_warnings(&self) -> bool {
        self.warning_count() > 0
    }

    /// Print the report to a terminal using colours and potentially other
    /// fancy output.
    pub fn fancy_print(
        &self,
        mut t: Box<term::Terminal<Output = io::Stderr>>,
    ) -> Result<(), term::Error> {
        for (file, errors) in &self.internal.borrow().0 {
            for error in errors {
                let prefix_space_len = error.line.to_string().len();
                let prefix_spaces = " ".repeat(1 + prefix_space_len);

                // First line: the overview of error
                t.fg(term::color::RED)?;
                t.attr(term::Attr::Bold)?;
                write!(t, "{} ", error.msg_prefix())?;
                t.reset()?;
                t.attr(term::Attr::Bold)?;
                writeln!(t, "{}", error.kind)?;

                // Second line: file info
                write!(t, "{}--> ", &prefix_spaces[1..])?;
                t.reset()?;
                writeln!(t, "{}:{}", file, error.line)?;

                // Third to fifth lines: show the line which triggered error, if available.
                if !error.line_buffer.is_empty() {
                    let (space_len, target_len) = error.format_len();
                    t.attr(term::Attr::Bold)?;
                    write!(t, "{}|\n{} | ", prefix_spaces, error.line)?;
                    t.reset()?;
                    writeln!(t, "{}", error.line_buffer)?;
                    t.attr(term::Attr::Bold)?;
                    write!(t, "{}| ", prefix_spaces)?;
                    t.fg(term::color::RED)?;
                    writeln!(t, "{}", FormatReport::target_str(space_len, target_len))?;
                    t.reset()?;
                }

                // The last line: show note if available.
                let msg_suffix = error.msg_suffix();
                if !msg_suffix.is_empty() {
                    t.attr(term::Attr::Bold)?;
                    write!(t, "{}= note: ", prefix_spaces)?;
                    t.reset()?;
                    writeln!(t, "{}", error.msg_suffix())?;
                } else {
                    writeln!(t)?;
                }
                t.reset()?;
            }
        }

        if !self.internal.borrow().0.is_empty() {
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

    fn target_str(space_len: usize, target_len: usize) -> String {
        let empty_line = " ".repeat(space_len);
        let overflowed = "^".repeat(target_len);
        empty_line + &overflowed
    }
}

impl fmt::Display for FormatReport {
    // Prints all the formatting errors.
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for (file, errors) in &self.internal.borrow().0 {
            for error in errors {
                let prefix_space_len = error.line.to_string().len();
                let prefix_spaces = " ".repeat(1 + prefix_space_len);

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
                        FormatReport::target_str(space_len, target_len)
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

                writeln!(
                    fmt,
                    "{}\n{}\n{}\n{}{}",
                    error_info,
                    file_info,
                    error_line_buffer,
                    note,
                    error.msg_suffix()
                )?;
            }
        }
        if !self.internal.borrow().0.is_empty() {
            writeln!(
                fmt,
                "warning: rustfmt may have failed to format. See previous {} errors.",
                self.warning_count(),
            )?;
        }
        Ok(())
    }
}

/// Format the given snippet. The snippet is expected to be *complete* code.
/// When we cannot parse the given snippet, this function returns `None`.
fn format_snippet(snippet: &str, config: &Config) -> Option<String> {
    let mut out: Vec<u8> = Vec::with_capacity(snippet.len() * 2);
    let input = Input::Text(snippet.into());
    let mut config = config.clone();
    config.set().emit_mode(config::EmitMode::Stdout);
    config.set().verbose(Verbosity::Quiet);
    config.set().hide_parse_errors(true);
    {
        let mut session = Session::new(config, Some(&mut out));
        let result = session.format(input);
        let formatting_error = session.summary.has_macro_formatting_failure()
            || session.out.as_ref().unwrap().is_empty() && !snippet.is_empty();
        if formatting_error || result.is_err() {
            return None;
        }
    }
    String::from_utf8(out).ok()
}

/// Format the given code block. Mainly targeted for code block in comment.
/// The code block may be incomplete (i.e. parser may be unable to parse it).
/// To avoid panic in parser, we wrap the code block with a dummy function.
/// The returned code block does *not* end with newline.
fn format_code_block(code_snippet: &str, config: &Config) -> Option<String> {
    const FN_MAIN_PREFIX: &str = "fn main() {\n";

    fn enclose_in_main_block(s: &str, config: &Config) -> String {
        let indent = Indent::from_width(config, config.tab_spaces());
        let mut result = String::with_capacity(s.len() * 2);
        result.push_str(FN_MAIN_PREFIX);
        let mut need_indent = true;
        for (kind, line) in LineClasses::new(s) {
            if need_indent {
                result.push_str(&indent.to_string(config));
            }
            result.push_str(&line);
            result.push('\n');
            need_indent = !kind.is_string() || line.ends_with('\\');
        }
        result.push('}');
        result
    }

    // Wrap the given code block with `fn main()` if it does not have one.
    let snippet = enclose_in_main_block(code_snippet, config);
    let mut result = String::with_capacity(snippet.len());
    let mut is_first = true;

    // While formatting the code, ignore the config's newline style setting and always use "\n"
    // instead of "\r\n" for the newline characters. This is okay because the output here is
    // not directly outputted by rustfmt command, but used by the comment formatter's input.
    // We have output-file-wide "\n" ==> "\r\n" conversion proccess after here if it's necessary.
    let mut config_with_unix_newline = config.clone();
    config_with_unix_newline
        .set()
        .newline_style(NewlineStyle::Unix);
    let formatted = format_snippet(&snippet, &config_with_unix_newline)?;

    // Trim "fn main() {" on the first line and "}" on the last line,
    // then unindent the whole code block.
    let block_len = formatted.rfind('}').unwrap_or(formatted.len());
    let mut is_indented = true;
    for (kind, ref line) in LineClasses::new(&formatted[FN_MAIN_PREFIX.len()..block_len]) {
        if !is_first {
            result.push('\n');
        } else {
            is_first = false;
        }
        let trimmed_line = if !is_indented {
            line
        } else if line.len() > config.max_width() {
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
        is_indented = !kind.is_string() || line.ends_with('\\');
    }
    Some(result)
}

/// A session is a run of rustfmt across a single or multiple inputs.
pub struct Session<'b, T: Write + 'b> {
    pub config: Config,
    pub out: Option<&'b mut T>,
    pub summary: Summary,
    filemap: FileMap,
}

impl<'b, T: Write + 'b> Session<'b, T> {
    pub fn new(config: Config, out: Option<&'b mut T>) -> Session<'b, T> {
        if config.emit_mode() == EmitMode::Checkstyle {
            println!("{}", checkstyle::header());
        }

        Session {
            config,
            out,
            summary: Summary::default(),
            filemap: FileMap::new(),
        }
    }

    /// The main entry point for Rustfmt. Formats the given input according to the
    /// given config. `out` is only necessary if required by the configuration.
    pub fn format(&mut self, input: Input) -> Result<FormatReport, ErrorKind> {
        self.format_input_inner(input)
    }

    pub fn override_config<F, U>(&mut self, mut config: Config, f: F) -> U
    where
        F: FnOnce(&mut Session<'b, T>) -> U,
    {
        mem::swap(&mut config, &mut self.config);
        let result = f(self);
        mem::swap(&mut config, &mut self.config);
        result
    }
}

impl<'b, T: Write + 'b> Drop for Session<'b, T> {
    fn drop(&mut self) {
        if self.config.emit_mode() == EmitMode::Checkstyle {
            println!("{}", checkstyle::footer());
        }
    }
}

#[derive(Debug)]
pub enum Input {
    File(PathBuf),
    Text(String),
}

impl Input {
    fn is_text(&self) -> bool {
        match *self {
            Input::File(_) => false,
            Input::Text(_) => true,
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

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
        #[cfg(not(windows))]
        let expected = "fn main() {\n    \
                        println!(\"hello, world\");\n\
                        }\n";
        #[cfg(windows)]
        let expected = "fn main() {\r\n    \
                        println!(\"hello, world\");\r\n\
                        }\r\n";
        assert!(test_format_inner(format_snippet, snippet, expected));
    }

    #[test]
    fn test_format_code_block_fail() {
        #[rustfmt::skip]
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
