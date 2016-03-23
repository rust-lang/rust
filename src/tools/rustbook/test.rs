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
use error::{err, CliResult, CommandResult};
use term::Term;
use book;

use std::fs::File;
use std::env;
use std::process::Command;

struct Test;

pub fn parse_cmd(name: &str) -> Option<Box<Subcommand>> {
    if name == "test" {
        Some(Box::new(Test))
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
        let cwd = env::current_dir().unwrap();
        let src = cwd.clone();

        let mut summary = File::open(&src.join("SUMMARY.md"))?;
        match book::parse_summary(&mut summary, &src) {
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
                                         String::from_utf8_lossy(&output.stdout),
                                         String::from_utf8_lossy(&output.stderr)));
                                return Err(err("some tests failed"));
                            }

                        }
                        Err(e) => {
                            let message = format!("could not execute `rustdoc`: {}", e);
                            return Err(err(&message))
                        }
                    }
                }
            }
            Err(errors) => {
                for err in errors {
                    term.err(&err[..]);
                }
                return Err(err("there was an error"))
            }
        }
        Ok(()) // lol
    }
}
