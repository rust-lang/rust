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

use std::fmt;
use std::fmt::{Debug, Formatter};

use std::old_io::IoError;

pub type CliError = Box<Error + 'static>;
pub type CliResult<T> = Result<T, CliError>;

pub type CommandError = Box<Error + 'static>;
pub type CommandResult<T> = Result<T, CommandError>;

pub trait Error {
    fn description(&self) -> &str;

    fn detail(&self) -> Option<&str> { None }
    fn cause(&self) -> Option<&Error> { None }
}

pub trait FromError<E> {
    fn from_err(err: E) -> Self;
}

impl Debug for Box<Error + 'static> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

impl<E: Error + 'static> FromError<E> for Box<Error + 'static> {
    fn from_err(err: E) -> Box<Error + 'static> {
        box err as Box<Error>
    }
}

impl<'a> Error for &'a str {
    fn description<'b>(&'b self) -> &'b str {
        *self
    }
}

impl Error for String {
    fn description<'a>(&'a self) -> &'a str {
        &self[..]
    }
}

impl<'a> Error for Box<Error + 'a> {
    fn description(&self) -> &str { (**self).description() }
    fn detail(&self) -> Option<&str> { (**self).detail() }
    fn cause(&self) -> Option<&Error> { (**self).cause() }
}

impl FromError<()> for () {
    fn from_err(_: ()) -> () { () }
}

impl FromError<IoError> for IoError {
    fn from_err(error: IoError) -> IoError { error }
}

impl Error for IoError {
    fn description(&self) -> &str {
        self.desc
    }
    fn detail(&self) -> Option<&str> {
        self.detail.as_ref().map(|s| &s[..])
    }
}


//fn iter_map_err<T, U, E, I: Iterator<Result<T,E>>>(iter: I,
