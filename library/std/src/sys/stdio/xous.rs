#[expect(dead_code)]
#[path = "unsupported.rs"]
mod unsupported_stdio;

use crate::io;
use crate::os::xous::ffi::{Connection, lend, try_lend, try_scalar};
use crate::os::xous::services::{LogLend, LogScalar, log_server, try_connect};

pub type Stdin = unsupported_stdio::Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        #[repr(C, align(4096))]
        struct LendBuffer([u8; 4096]);
        let mut lend_buffer = LendBuffer([0u8; 4096]);
        let connection = log_server();
        for chunk in buf.chunks(lend_buffer.0.len()) {
            for (dest, src) in lend_buffer.0.iter_mut().zip(chunk) {
                *dest = *src;
            }
            lend(connection, LogLend::StandardOutput.into(), &lend_buffer.0, 0, chunk.len())
                .unwrap();
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        #[repr(C, align(4096))]
        struct LendBuffer([u8; 4096]);
        let mut lend_buffer = LendBuffer([0u8; 4096]);
        let connection = log_server();
        for chunk in buf.chunks(lend_buffer.0.len()) {
            for (dest, src) in lend_buffer.0.iter_mut().zip(chunk) {
                *dest = *src;
            }
            lend(connection, LogLend::StandardError.into(), &lend_buffer.0, 0, chunk.len())
                .unwrap();
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = unsupported_stdio::STDIN_BUF_SIZE;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

#[derive(Copy, Clone)]
pub struct PanicWriter {
    log: Connection,
    gfx: Option<Connection>,
}

impl io::Write for PanicWriter {
    fn write(&mut self, s: &[u8]) -> core::result::Result<usize, io::Error> {
        for c in s.chunks(size_of::<usize>() * 4) {
            // Text is grouped into 4x `usize` words. The id is 1100 plus
            // the number of characters in this message.
            // Ignore errors since we're already panicking.
            try_scalar(self.log, LogScalar::AppendPanicMessage(&c).into()).ok();
        }

        // Serialize the text to the graphics panic handler, only if we were able
        // to acquire a connection to it. Text length is encoded in the `valid` field,
        // the data itself in the buffer. Typically several messages are require to
        // fully transmit the entire panic message.
        if let Some(gfx) = self.gfx {
            #[repr(C, align(4096))]
            struct Request([u8; 4096]);
            let mut request = Request([0u8; 4096]);
            for (&s, d) in s.iter().zip(request.0.iter_mut()) {
                *d = s;
            }
            try_lend(gfx, 0 /* AppendPanicText */, &request.0, 0, s.len()).ok();
        }
        Ok(s.len())
    }

    // Tests show that this does not seem to be reliably called at the end of a panic
    // print, so, we can't rely on this to e.g. trigger a graphics update.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn panic_output() -> Option<impl io::Write> {
    // Generally this won't fail because every server has already connected, so
    // this is likely to succeed.
    let log = log_server();

    // Send the "We're panicking" message (1000).
    try_scalar(log, LogScalar::BeginPanic.into()).ok();

    // This is will fail in the case that the connection table is full, or if the
    // graphics server is not running. Most servers do not already have this connection.
    let gfx = try_connect("panic-to-screen!");

    Some(PanicWriter { log, gfx })
}
