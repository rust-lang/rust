// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(rustc_private)]
#![feature(collections)]
#![feature(exit_status)]
#![feature(str_char)]

// TODO we're going to allocate a whole bunch of temp Strings, is it worth
// keeping some scratch mem for this and running our own StrPool?
// TODO for lint violations of names, emit a refactor script

// TODO priorities
// Fix fns and methods properly
//   dead spans (comments) - in where clause (wait for fixed spans, test)
//
// Smoke testing till we can use it
//   ** no newline at the end of doc.rs
// take config options from a file

#[macro_use]
extern crate log;

extern crate getopts;
extern crate rustc;
extern crate rustc_driver;
extern crate syntax;

extern crate strings;

use rustc::session::Session;
use rustc::session::config::{self, Input};
use rustc_driver::{driver, CompilerCalls, Compilation};

use syntax::ast;
use syntax::codemap::CodeMap;
use syntax::diagnostics;
use syntax::visit;

use std::path::PathBuf;
use std::collections::HashMap;

use changes::ChangeSet;
use visitor::FmtVisitor;

mod changes;
mod visitor;
mod functions;
mod missed_spans;
mod lists;
mod utils;
mod types;
mod expr;
mod imports;

const IDEAL_WIDTH: usize = 80;
const LEEWAY: usize = 5;
const MAX_WIDTH: usize = 100;
const MIN_STRING: usize = 10;
const TAB_SPACES: usize = 4;
const FN_BRACE_STYLE: BraceStyle = BraceStyle::SameLineWhere;
const FN_RETURN_INDENT: ReturnIndent = ReturnIndent::WithArgs;
// When we get scoped annotations, we should have rustfmt::skip.
const SKIP_ANNOTATION: &'static str = "rustfmt_skip";

#[derive(Copy, Clone)]
pub enum WriteMode {
    Overwrite,
    // str is the extension of the new file
    NewFile(&'static str),
    // Write the output to stdout.
    Display,
    // Return the result as a mapping from filenames to StringBuffers.
    Return(&'static Fn(HashMap<String, String>)),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum BraceStyle {
    AlwaysNextLine,
    PreferSameLine,
    // Prefer same line except where there is a where clause, in which case force
    // the brace to the next line.
    SameLineWhere,
}

// How to indent a function's return type.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum ReturnIndent {
    // Aligned with the arguments
    WithArgs,
    // Aligned with the where clause
    WithWhereClause,
}

// Formatting which depends on the AST.
fn fmt_ast<'a>(krate: &ast::Crate, codemap: &'a CodeMap) -> ChangeSet<'a> {
    let mut visitor = FmtVisitor::from_codemap(codemap);
    visit::walk_crate(&mut visitor, krate);
    let files = codemap.files.borrow();
    if let Some(last) = files.last() {
        visitor.format_missing(last.end_pos);
    }

    visitor.changes
}

// Formatting done on a char by char or line by line basis.
// TODO warn on TODOs and FIXMEs without an issue number
// TODO warn on bad license
// TODO other stuff for parity with make tidy
fn fmt_lines(changes: &mut ChangeSet) {
    // Iterate over the chars in the change set.
    for (f, text) in changes.text() {
        let mut trims = vec![];
        let mut last_wspace: Option<usize> = None;
        let mut line_len = 0;
        let mut cur_line = 1;
        for (c, b) in text.chars() {
            if c == '\n' { // TOOD test for \r too
                // Check for (and record) trailing whitespace.
                if let Some(lw) = last_wspace {
                    trims.push((cur_line, lw, b));
                    line_len -= b - lw;
                }
                // Check for any line width errors we couldn't correct.
                if line_len > MAX_WIDTH {
                    // TODO store the error rather than reporting immediately.
                    println!("Rustfmt couldn't fix (sorry). {}:{}: line longer than {} characters",
                             f, cur_line, MAX_WIDTH);
                }
                line_len = 0;
                cur_line += 1;
                last_wspace = None;
            } else {
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

        for &(l, _, _) in trims.iter() {
            // TODO store the error rather than reporting immediately.
            println!("Rustfmt left trailing whitespace at {}:{} (sorry)", f, l);
        }
    }
}

struct RustFmtCalls {
    input_path: Option<PathBuf>,
    write_mode: WriteMode,
}

impl<'a> CompilerCalls<'a> for RustFmtCalls {
    fn early_callback(&mut self,
                      _: &getopts::Matches,
                      _: &diagnostics::registry::Registry)
                      -> Compilation {
        Compilation::Continue
    }

    fn some_input(&mut self,
                  input: Input,
                  input_path: Option<PathBuf>)
                  -> (Input, Option<PathBuf>) {
        match input_path {
            Some(ref ip) => self.input_path = Some(ip.clone()),
            _ => {
                // FIXME should handle string input and write to stdout or something
                panic!("No input path");
            }
        }
        (input, input_path)
    }

    fn no_input(&mut self,
                _: &getopts::Matches,
                _: &config::Options,
                _: &Option<PathBuf>,
                _: &Option<PathBuf>,
                _: &diagnostics::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        panic!("No input supplied to RustFmt");
    }

    fn late_callback(&mut self,
                     _: &getopts::Matches,
                     _: &Session,
                     _: &Input,
                     _: &Option<PathBuf>,
                     _: &Option<PathBuf>)
                     -> Compilation {
        Compilation::Continue
    }

    fn build_controller(&mut self, _: &Session) -> driver::CompileController<'a> {
        let write_mode = self.write_mode;
        let mut control = driver::CompileController::basic();
        control.after_parse.stop = Compilation::Stop;
        control.after_parse.callback = box move |state| {
            let krate = state.krate.unwrap();
            let codemap = state.session.codemap();
            let mut changes = fmt_ast(krate, codemap);
            fmt_lines(&mut changes);

            // FIXME(#5) Should be user specified whether to show or replace.
            let result = changes.write_all_files(write_mode);

            match result {
                Err(msg) => println!("Error writing files: {}", msg),
                Ok(result) => {
                    if let WriteMode::Return(callback) = write_mode {
                        callback(result);
                    }
                }
            }
        };

        control
    }
}

fn run(args: Vec<String>, write_mode: WriteMode) {
    let mut call_ctxt = RustFmtCalls { input_path: None, write_mode: write_mode };
    rustc_driver::run_compiler(&args, &mut call_ctxt);    
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    run(args, WriteMode::Display);
    std::env::set_exit_status(0);

    // TODO unit tests
    // let fmt = ListFormatting {
    //     tactic: ListTactic::Horizontal,
    //     separator: ",",
    //     trailing_separator: SeparatorTactic::Vertical,
    //     indent: 2,
    //     h_width: 80,
    //     v_width: 100,
    // };
    // let inputs = vec![(format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new()),
    //                   (format!("foo"), String::new())];
    // let s = write_list(&inputs, &fmt);
    // println!("  {}", s);
}

// FIXME comments
// comments aren't in the AST, which makes processing them difficult, but then
// comments are complicated anyway. I think I am happy putting off tackling them
// for now. Long term the soluton is for comments to be in the AST, but that means
// only the libsyntax AST, not the rustc one, which means waiting for the ASTs
// to diverge one day....

// Once we do have comments, we just have to implement a simple word wrapping
// algorithm to keep the width under IDEAL_WIDTH. We should also convert multiline
// /* ... */ comments to // and check doc comments are in the right place and of
// the right kind.

// Should also make sure comments have the right indent

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::fs;
    use std::io::Read;
    use super::*;
    use super::run;

    // For now, the only supported regression tests are idempotent tests - the input and
    // output must match exactly.
    // TODO would be good to check for error messages and fail on them, or at least report.
    #[test]
    fn idempotent_tests() {
        println!("Idempotent tests:");
        unsafe { FAILURES = 0; }

        // Get all files in the tests/idem directory
        let files = fs::read_dir("tests/idem").unwrap();
        // For each file, run rustfmt and collect the output
        let mut count = 0;
        for entry in files {
            let path = entry.unwrap().path();
            let file_name = path.to_str().unwrap();
            println!("Testing '{}'...", file_name);
            run(vec!["rustfmt".to_string(), file_name.to_string()], WriteMode::Return(HANDLE_RESULT));
            count += 1;
        }
        // And also dogfood ourselves!
        println!("Testing 'src/mod.rs'...");
        run(vec!["rustfmt".to_string(), "src/mod.rs".to_string()], WriteMode::Return(HANDLE_RESULT));
        count += 1;

        // Display results
        let fails = unsafe { FAILURES };
        println!("Ran {} idempotent tests; {} failures.", count, fails);
        assert!(fails == 0, "{} idempotent tests failed", fails);
    }

    // 'global' used by sys_tests and handle_result.
    static mut FAILURES: i32 = 0;
    // Ick, just needed to get a &'static to handle_result.
    static HANDLE_RESULT: &'static Fn(HashMap<String, String>) = &handle_result;

    // Compare output to input.
    fn handle_result(result: HashMap<String, String>) {
        let mut fails = 0;

        for file_name in result.keys() {
            let mut f = fs::File::open(file_name).unwrap();
            let mut text = String::new();
            f.read_to_string(&mut text).unwrap();
            if result[file_name] != text {
                fails += 1;
                println!("Mismatch in {}.", file_name);
            }
        }

        if fails > 0 {
            unsafe {
                FAILURES += 1;
            }
        }
    }
}
