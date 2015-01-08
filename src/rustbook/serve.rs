// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the `serve` subcommand. Just a stub for now.

use subcommand::Subcommand;
use error::CliResult;
use error::CommandResult;
use term::Term;

struct Serve;

pub fn parse_cmd(name: &str) -> Option<Box<Subcommand>> {
    if name == "serve" {
        Some(box Serve as Box<Subcommand>)
    } else {
        None
    }
}

impl Subcommand for Serve {
    fn parse_args(&mut self, _: &[String]) -> CliResult<()> {
        Ok(())
    }
    fn usage(&self) {}
    fn execute(&mut self, _: &mut Term) -> CommandResult<()> {
        Ok(())
    }
}
