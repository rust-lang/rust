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
#![feature(str_escape)]
#![feature(str_char)]
#![feature(custom_attribute)]
#![allow(unused_attributes)]

// TODO we're going to allocate a whole bunch of temp Strings, is it worth
// keeping some scratch mem for this and running our own StrPool?
// TODO for lint violations of names, emit a refactor script


#[macro_use]
extern crate log;

extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate syntax;
extern crate rustc_serialize;

extern crate strings;

use rustc::session::Session;
use rustc::session::config as rustc_config;
use rustc::session::config::Input;
use rustc_driver::{driver, CompilerCalls, Compilation};

use syntax::ast;
use syntax::codemap::CodeMap;
use syntax::diagnostics;

use std::path::PathBuf;
use std::collections::HashMap;
use std::fmt;
use std::mem::swap;
use std::str::FromStr;

use issues::{BadIssueSeeker, Issue};
use filemap::FileMap;
use visitor::FmtVisitor;
use config::Config;

#[macro_use]
mod utils;
pub mod config;
mod filemap;
mod visitor;
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
mod modules;

const MIN_STRING: usize = 10;
// When we get scoped annotations, we should have rustfmt::skip.
const SKIP_ANNOTATION: &'static str = "rustfmt_skip";

#[derive(Copy, Clone)]
pub enum WriteMode {
    // Backups the original file and overwrites the orignal.
    Replace,
    // Overwrites original file without backup.
    Overwrite,
    // str is the extension of the new file.
    NewFile(&'static str),
    // Write the output to stdout.
    Display,
    // Return the result as a mapping from filenames to Strings.
    Return(&'static Fn(HashMap<String, String>)),
}

impl FromStr for WriteMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "replace" => Ok(WriteMode::Replace),
            "display" => Ok(WriteMode::Display),
            "overwrite" => Ok(WriteMode::Overwrite),
            _ => Err(()),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum NewlineStyle {
    Windows, // \r\n
    Unix, // \n
}

impl_enum_decodable!(NewlineStyle, Windows, Unix);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum BraceStyle {
    AlwaysNextLine,
    PreferSameLine,
    // Prefer same line except where there is a where clause, in which case force
    // the brace to the next line.
    SameLineWhere,
}

impl_enum_decodable!(BraceStyle, AlwaysNextLine, PreferSameLine, SameLineWhere);

// How to indent a function's return type.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ReturnIndent {
    // Aligned with the arguments
    WithArgs,
    // Aligned with the where clause
    WithWhereClause,
}

impl_enum_decodable!(ReturnIndent, WithArgs, WithWhereClause);

// How to stle a struct literal.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum StructLitStyle {
    // First line on the same line as the opening brace, all lines aligned with
    // the first line.
    Visual,
    // First line is on a new line and all lines align with block indent.
    Block,
    // FIXME Maybe we should also have an option to align types.
}

impl_enum_decodable!(StructLitStyle, Visual, Block);

enum ErrorKind {
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
            ErrorKind::LineOverflow => {
                write!(fmt, "line exceeded maximum length")
            }
            ErrorKind::TrailingWhitespace => {
                write!(fmt, "left behind trailing whitespace")
            }
            ErrorKind::BadIssue(issue) => {
                write!(fmt, "found {}", issue)
            }
        }
    }
}

// Formatting errors that are identified *after* rustfmt has run
struct FormattingError {
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

struct FormatReport {
    // Maps stringified file paths to their associated formatting errors
    file_error_map: HashMap<String, Vec<FormattingError>>,
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
fn fmt_ast(krate: &ast::Crate, codemap: &CodeMap, config: &Config) -> FileMap {
    let mut file_map = FileMap::new();
    for (path, module) in modules::list_files(krate, codemap) {
        let path = path.to_str().unwrap();
        let mut visitor = FmtVisitor::from_codemap(codemap, config);
        visitor.format_separate_mod(module, path);
        file_map.insert(path.to_owned(), visitor.buffer);
    }
    file_map
}

// Formatting done on a char by char or line by line basis.
// TODO warn on bad license
// TODO other stuff for parity with make tidy
// FIXME skipping due to `continue`, #184.
#[rustfmt_skip]
fn fmt_lines(file_map: &mut FileMap, config: &Config) -> FormatReport {
    let mut truncate_todo = Vec::new();
    let mut report = FormatReport { file_error_map: HashMap::new() };

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
                continue;
            }

            // Add warnings for bad todos/ fixmes
            if let Some(issue) = issue_seeker.inspect(c) {
                errors.push(FormattingError {
                    line: cur_line,
                    kind: ErrorKind::BadIssue(issue)
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
                        kind: ErrorKind::LineOverflow
                    });
                }
                line_len = 0;
                cur_line += 1;
                newline_count += 1;
                last_wspace = None;
            } else {
                newline_count = 0;
                line_len += 1;
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
                kind: ErrorKind::TrailingWhitespace
            });
        }

        report.file_error_map.insert(f.to_owned(), errors);
    }

    for (f, l) in truncate_todo {
        file_map.get_mut(&f).unwrap().truncate(l);
    }

    report
}

struct RustFmtCalls {
    write_mode: WriteMode,
    config: Option<Box<config::Config>>,
}

impl<'a> CompilerCalls<'a> for RustFmtCalls {
    fn no_input(&mut self,
                _: &getopts::Matches,
                _: &rustc_config::Options,
                _: &Option<PathBuf>,
                _: &Option<PathBuf>,
                _: &diagnostics::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        panic!("No input supplied to RustFmt");
    }

    #[rustfmt_skip]
    // FIXME(#195): closure is formatted poorly.
    fn build_controller(&mut self, _: &Session) -> driver::CompileController<'a> {
        let write_mode = self.write_mode;

        let mut config_option = None;
        swap(&mut self.config, &mut config_option);
        let config = config_option.unwrap();

        let mut control = driver::CompileController::basic();
        control.after_parse.stop = Compilation::Stop;
        control.after_parse.callback = Box::new(move |state| {
            let krate = state.krate.unwrap();
            let codemap = state.session.codemap();
            let mut file_map = fmt_ast(krate, codemap, &*config);
            // For some reason, the codemap does not include terminating newlines
            // so we must add one on for each file. This is sad.
            filemap::append_newlines(&mut file_map);
            println!("{}", fmt_lines(&mut file_map, &*config));

            let result = filemap::write_all_files(&file_map, write_mode, &*config);

            match result {
                Err(msg) => println!("Error writing files: {}", msg),
                Ok(result) => {
                    if let WriteMode::Return(callback) = write_mode {
                        callback(result);
                    }
                }
            }
        });

        control
    }
}

// args are the arguments passed on the command line, generally passed through
// to the compiler.
// write_mode determines what happens to the result of running rustfmt, see
// WriteMode.
pub fn run(args: Vec<String>, write_mode: WriteMode, config: Box<Config>) {
    let mut call_ctxt = RustFmtCalls { write_mode: write_mode, config: Some(config) };
    rustc_driver::run_compiler(&args, &mut call_ctxt);
}
