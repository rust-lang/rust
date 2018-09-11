// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::io::{self, Error, Write, sink};

struct ErrorDisplay;

impl fmt::Display for ErrorDisplay {
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        Err(fmt::Error)
    }
}

struct ErrorWriter;

const FORMAT_ERROR: io::ErrorKind = io::ErrorKind::Other;
const WRITER_ERROR: io::ErrorKind = io::ErrorKind::NotConnected;

impl Write for ErrorWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(Error::new(WRITER_ERROR, "not connected"))
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

fn main() {
    // Test that the error from the formatter is propagated.
    let res = write!(sink(), "{} {} {}", 1, ErrorDisplay, "bar");
    assert!(res.is_err(), "formatter error did not propagate");
    assert_eq!(res.unwrap_err().kind(), FORMAT_ERROR);

    // Test that an underlying error is propagated
    let res = write!(ErrorWriter, "abc");
    assert!(res.is_err(), "writer error did not propagate");

    // Writer error
    let res = write!(ErrorWriter, "abc {}", ErrorDisplay);
    assert!(res.is_err(), "writer error did not propagate");
    assert_eq!(res.unwrap_err().kind(), WRITER_ERROR);

    // Formatter error
    let res = write!(ErrorWriter, "{} abc", ErrorDisplay);
    assert!(res.is_err(), "formatter error did not propagate");
    assert_eq!(res.unwrap_err().kind(), FORMAT_ERROR);
}
