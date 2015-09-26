// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the `help` subcommand. Currently just prints basic usage info.

use subcommand::Subcommand;
use error::CliResult;
use error::CommandResult;
use term::Term;

struct Help;

pub fn parse_cmd(name: &str) -> Option<Box<Subcommand>> {
    match name {
        "help" | "--help" | "-h" | "-?" => Some(Box::new(Help)),
        _ => None
    }
}

impl Subcommand for Help {
    fn parse_args(&mut self, _: &[String]) -> CliResult<()> {
        Ok(())
    }
    fn usage(&self) {}
    fn execute(&mut self, _: &mut Term) -> CommandResult<()> {
        usage();
        Ok(())
    }
}

pub fn usage() {
    println!("Usage: rustbook <command> [<args>]");
    println!("");
    println!("The <command> must be one of:");
    println!("  help    Print this message.");
    println!("  build   Build the book in subdirectory _book");
    println!("  serve   --NOT YET IMPLEMENTED--");
    println!("  test    --NOT YET IMPLEMENTED--");
}
