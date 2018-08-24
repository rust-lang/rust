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
