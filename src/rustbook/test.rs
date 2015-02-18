// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the `test` subcommand. Just a stub for now.

use subcommand::Subcommand;
use error::CliResult;
use error::CommandResult;
use error::Error;
use term::Term;
use book;
use std::old_io::{Command, File};
use std::os;

struct Test;

pub fn parse_cmd(name: &str) -> Option<Box<Subcommand>> {
    if name == "test" {
        Some(box Test as Box<Subcommand>)
    } else {
        None
    }
}

impl Subcommand for Test {
    fn parse_args(&mut self, _: &[String]) -> CliResult<()> {
        Ok(())
    }
    fn usage(&self) {}
    fn execute(&mut self, term: &mut Term) -> CommandResult<()> {
        let cwd = os::getcwd().unwrap();
        let src = cwd.clone();

        let summary = File::open(&src.join("SUMMARY.md"));
        match book::parse_summary(summary, &src) {
            Ok(book) => {
                for (_, item) in book.iter() {
                    let output_result = Command::new("rustdoc")
                        .arg(&item.path)
                        .arg("--test")
                        .output();
                    match output_result {
                        Ok(output) => {
                            if !output.status.success() {
                                term.err(&format!("{}\n{}",
                                         String::from_utf8_lossy(&output.output[]),
                                         String::from_utf8_lossy(&output.error[]))[]);
                                return Err(box "Some tests failed." as Box<Error>);
                            }

                        }
                        Err(e) => {
                            let message = format!("Could not execute `rustdoc`: {}", e);
                            return Err(box message as Box<Error>);
                        }
                    }
                }
            }
            Err(errors) => {
                for err in errors {
                    term.err(&err[..]);
                }
                return Err(box "There was an error." as Box<Error>);
            }
        }
        Ok(()) // lol
    }
}
