use crate::io;
use crate::mem::MaybeUninit;
use crate::os::xous::ffi::{Connection, lend, try_lend, try_scalar};
use crate::os::xous::services::{LogLend, LogScalar, log_server, try_connect};

pub type Stdin = super::unsupported_stdio::Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout {}
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write(LogLend::StandardOutput, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        write_all(LogLend::StandardOutput, buf)
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write(LogLend::StandardError, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        write_all(LogLend::StandardError, buf)
    }
}

#[repr(C, align(4096))]
struct AlignedBuffer([MaybeUninit<u8>; 4096]);

impl AlignedBuffer {
    #[inline]
    fn new() -> Self {
        AlignedBuffer([MaybeUninit::uninit(); 4096])
    }

    #[inline]
    fn fill(&mut self, buf: &[u8]) -> &[u8] {
        let len = buf.len().min(self.0.len());
        self.0[..len].write_copy_of_slice(&buf[..len]);
        // SAFETY: This range was just initialized.
        unsafe { self.0[..len].assume_init_ref() }
    }
}

fn write(opcode: LogLend, buf: &[u8]) -> io::Result<usize> {
    let mut aligned_buffer = AlignedBuffer::new();
    let aligned = aligned_buffer.fill(buf);
    lend(log_server(), opcode.into(), aligned, 0, aligned.len()).unwrap();
    Ok(aligned.len())
}

fn write_all(opcode: LogLend, buf: &[u8]) -> io::Result<()> {
    let mut aligned_buffer = AlignedBuffer::new();
    let connection = log_server();
    for chunk in buf.chunks(aligned_buffer.0.len()) {
        let aligned = aligned_buffer.fill(chunk);
        lend(connection, opcode.into(), aligned, 0, aligned.len()).unwrap();
    }
    Ok(())
}

pub const STDIN_BUF_SIZE: usize = super::unsupported_stdio::STDIN_BUF_SIZE;

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
        for c in s.chunks(core::mem::size_of::<usize>() * 4) {
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
            let mut request = AlignedBuffer::new();
            let request = request.fill(s);
            _ = try_lend(gfx, 0 /* AppendPanicText */, request, 0, request.len());
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
