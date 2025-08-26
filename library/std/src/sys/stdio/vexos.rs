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
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        let mut written = 0;

        while let Some((out_byte, new_buf)) = buf.split_first_mut() {
            buf = new_buf;

            let byte = unsafe { vex_sdk::vexSerialReadChar(STDIO_CHANNEL) };
            if byte < 0 {
                break;
            }

            *out_byte = byte as u8;
            written += 1;
        }

        Ok(written)
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut count = 0;

        // HACK: VEXos holds an internal write buffer for serial that is flushed to USB1 roughly every
        // millisecond by `vexTasksRun`. For writes larger than 2048 bytes, we must block until that buffer
        // is flushed to USB1 before writing the rest of `buf`. In practice, this is fairly nonstandard for
        // a `write` implementation but it avoids an guaranteed recursive panic when using macros such as
        // `print!` to write large amounts of data to stdout at once.
        for chunk in buf.chunks(STDOUT_BUF_SIZE) {
            if unsafe { vex_sdk::vexSerialWriteFree(STDIO_CHANNEL) as usize } < chunk.len() {
                self.flush().unwrap();
            }

            count += unsafe { vex_sdk::vexSerialWriteBuffer(STDIO_CHANNEL, chunk.as_ptr(), chunk.len() as u32) };
    
            if count < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Uncategorized,
                    "Internal write error occurred.",
                ));
            }
        }

        Ok(count as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
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
