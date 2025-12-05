//! Tests that errors from both the writer (`Write::write`) and formatter (`Display::fmt`)
//! are correctly propagated: writer errors return `Err`, formatter errors cause panics.

//@ run-pass
//@ needs-unwind

#![feature(io_error_uncategorized)]

use std::fmt;
use std::io::{self, Error, Write};
use std::panic::catch_unwind;

struct ErrorDisplay;

impl fmt::Display for ErrorDisplay {
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        Err(fmt::Error)
    }
}

struct ErrorWriter;

const WRITER_ERROR: io::ErrorKind = io::ErrorKind::NotConnected;

impl Write for ErrorWriter {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(Error::new(WRITER_ERROR, "not connected"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn main() {
    // Test that an underlying error is propagated
    let res = write!(ErrorWriter, "abc");
    assert!(res.is_err(), "writer error did not propagate");

    // Test that the error from the formatter is detected.
    let res = catch_unwind(|| write!(vec![], "{} {} {}", 1, ErrorDisplay, "bar"));
    let err = res.expect_err("formatter error did not lead to panic").downcast::<&str>().unwrap();
    assert!(
        err.contains("formatting trait implementation returned an error"),
        "unexpected panic: {}",
        err
    );

    // Writer error when there's some string before the first `{}`
    let res = write!(ErrorWriter, "abc {}", ErrorDisplay);
    assert!(res.is_err(), "writer error did not propagate");
    assert_eq!(res.unwrap_err().kind(), WRITER_ERROR);

    // Formatter error when the `{}` comes first
    let res = catch_unwind(|| write!(ErrorWriter, "{} abc", ErrorDisplay));
    let err = res.expect_err("formatter error did not lead to panic").downcast::<&str>().unwrap();
    assert!(
        err.contains("formatting trait implementation returned an error"),
        "unexpected panic: {}",
        err
    );
}
