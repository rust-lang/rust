use crate::io;

pub struct Stdin;
pub struct Stdout;
pub type Stderr = Stdout;

pub const STDIO_CHANNEL: u32 = 1;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut count = 0;

        for out_byte in buf.iter_mut() {
            let byte = unsafe { vex_sdk::vexSerialReadChar(STDIO_CHANNEL) };
            if byte < 0 {
                break;
            }

            *out_byte = byte as u8;
            count += 1;
        }

        Ok(count)
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut written = 0;

        // HACK: VEXos holds an internal ringbuffer for serial writes that is flushed to USB1
        // roughly every millisecond by `vexTasksRun`. For writes larger than 2048 bytes, we
        // must block until that buffer is flushed to USB1 before writing the rest of `buf`.
        //
        // This is fairly nonstandard for a `write` implementation, but it avoids a guaranteed
        // recursive panic when using macros such as `print!` to write large amounts of data
        // (buf.len() > 2048) to stdout at once.
        for chunk in buf.chunks(STDOUT_BUF_SIZE) {
            if unsafe { vex_sdk::vexSerialWriteFree(STDIO_CHANNEL) as usize } < chunk.len() {
                self.flush().unwrap();
            }

            let count: usize = unsafe {
                vex_sdk::vexSerialWriteBuffer(STDIO_CHANNEL, chunk.as_ptr(), chunk.len() as u32)
            }
            .try_into()
            .map_err(|_| {
                io::const_error!(io::ErrorKind::Uncategorized, "internal write error occurred")
            })?;

            written += count;

            // This is a sanity check to ensure that we don't end up with non-contiguous
            // buffer writes. e.g. a chunk gets only partially written, but we continue
            // attempting to write the remaining chunks.
            //
            // In practice, this should never really occur since the previous flush ensures
            // enough space in FIFO to write the entire chunk to vexSerialWriteBuffer.
            if count != chunk.len() {
                break;
            }
        }

        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        // This may block for up to a millisecond.
        unsafe {
            while (vex_sdk::vexSerialWriteFree(STDIO_CHANNEL) as usize) != STDOUT_BUF_SIZE {
                vex_sdk::vexTasksRun();
            }
        }

        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = 4096;
pub const STDOUT_BUF_SIZE: usize = 2048;

pub fn is_ebadf(_err: &io::Error) -> bool {
    false
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stdout::new())
}
