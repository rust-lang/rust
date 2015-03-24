// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error handling utilities. WIP.

use std::error::Error;
use std::fmt;

pub type CliError = Box<Error + 'static>;
pub type CliResult<T> = Result<T, CliError>;

pub type CommandError = Box<Error + 'static>;
pub type CommandResult<T> = Result<T, CommandError>;

pub fn err(s: &str) -> CliError {
    #[derive(Debug)]
    struct E(String);

    impl Error for E {
        fn description(&self) -> &str { &self.0 }
    }
    impl fmt::Display for E {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.0.fmt(f)
        }
    }

    Box::new(E(s.to_string()))
}
