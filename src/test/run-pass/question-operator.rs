// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::error::FromError;
use std::fs::File;
use std::io::{Read, self};
use std::num::ParseIntError;
use std::str::FromStr;

fn parse<T: FromStr>(s: &str) -> Result<T, T::Err> {
    s.parse()
}

fn on_method() -> Result<i32, ParseIntError> {
    Ok("1".parse::<i32>()? + "2".parse()?)
}

fn in_chain() -> Result<String, ParseIntError> {
    Ok("3".parse::<i32>()?.to_string())
}

fn on_call() -> Result<i32, ParseIntError> {
    Ok(parse("4")?)
}

fn nested() -> Result<i32, ParseIntError> {
    Ok("5".parse::<i32>()?.to_string().parse()?)
}

fn main() {
    assert_eq!(Ok(3), on_method());

    assert_eq!(Ok("3".to_string()), in_chain());

    assert_eq!(Ok(4), on_call());

    assert_eq!(Ok(5), nested());
}

enum Error {
    Io(io::Error),
    Parse(ParseIntError),
}

// just type check
fn merge_error() -> Result<i32, Error> {
    let mut s = String::new();

    File::open("foo.txt")?.read_to_string(&mut s)?;

    Ok(s.parse::<i32>()? + 1)
}

impl FromError<io::Error> for Error {
    fn from_error(e: io::Error) -> Error {
        Error::Io(e)
    }
}

impl FromError<ParseIntError> for Error {
    fn from_error(e: ParseIntError) -> Error {
        Error::Parse(e)
    }
}
