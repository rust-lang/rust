use wasip2::cli;
use wasip2::io::streams::{Error, InputStream, OutputStream, StreamError};

use crate::io::{self, BorrowedBuf, BorrowedCursor};

pub struct Stdin(Option<InputStream>);
pub struct Stdout(Option<OutputStream>);
pub struct Stderr(Option<OutputStream>);

fn error_to_io(err: Error) -> io::Error {
    // There exists a function in `wasi:filesystem` to optionally acquire an
    // error code from an error, but the streams in use in this module are
    // exclusively used with stdio meaning that a filesystem error is not
    // possible here.
    //
    // In lieu of an error code, which WASIp2 does not specify, this instead
    // carries along the `to_debug_string` implementation that the host
    // supplies. If this becomes too expensive in the future this could also
    // become `io::Error::from_raw_os_error(libc::EIO)` or similar.
    io::Error::new(io::ErrorKind::Other, err.to_debug_string())
}

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin(None)
    }

    fn stream(&mut self) -> &InputStream {
        self.0.get_or_insert_with(cli::stdin::get_stdin)
    }
}

impl io::Read for Stdin {
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        let mut buf = BorrowedBuf::from(data);
        self.read_buf(buf.unfilled())?;
        Ok(buf.len())
    }

    fn read_buf(&mut self, mut buf: BorrowedCursor<'_>) -> io::Result<()> {
        match self.stream().blocking_read(u64::try_from(buf.capacity()).unwrap()) {
            Ok(result) => {
                buf.append(&result);
                Ok(())
            }
            Err(StreamError::Closed) => Ok(()),
            Err(StreamError::LastOperationFailed(e)) => Err(error_to_io(e)),
        }
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout(None)
    }

    fn stream(&mut self) -> &OutputStream {
        self.0.get_or_insert_with(cli::stdout::get_stdout)
    }
}

fn write(stream: &OutputStream, buf: &[u8]) -> io::Result<usize> {
    // WASIp2's `blocking_write_and_flush` function is defined as accepting no
    // more than 4096 bytes. Larger writes can be issued by manually using
    // `check_write`, `write`, and `blocking_flush`, but for now just go ahead
    // and use `blocking_write_and_flush` and report a short write and let a
    // higher level loop over the result.
    const MAX: usize = 4096;
    let buf = &buf[..buf.len().min(MAX)];
    match stream.blocking_write_and_flush(buf) {
        Ok(()) => Ok(buf.len()),
        Err(StreamError::Closed) => Ok(0),
        Err(StreamError::LastOperationFailed(e)) => Err(error_to_io(e)),
    }
}

impl io::Write for Stdout {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        write(self.stream(), data)
    }

    fn flush(&mut self) -> io::Result<()> {
        // Note that `OutputStream` has a `flush` function but for stdio all
        // writes are accompanied with a flush which means that this flush
        // doesn't need to do anything.
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr(None)
    }

    fn stream(&mut self) -> &OutputStream {
        self.0.get_or_insert_with(cli::stderr::get_stderr)
    }
}

impl io::Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        write(self.stream(), data)
    }

    fn flush(&mut self) -> io::Result<()> {
        // See `Stdout::flush` for why this is a noop.
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = crate::sys::io::DEFAULT_BUF_SIZE;

pub fn is_ebadf(_err: &io::Error) -> bool {
    // WASIp2 stdio streams are always available so ebadf never shows up.
    false
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
