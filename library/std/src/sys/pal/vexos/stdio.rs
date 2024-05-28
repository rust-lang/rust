use crate::io;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

const STDIO_CHANNEL: u32 = 1;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin(())
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
        Stdout(())
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written =
            unsafe { vex_sdk::vexSerialWriteBuffer(STDIO_CHANNEL, buf.as_ptr(), buf.len() as u32) };

        if written < 0 {
            return Err(io::Error::new(io::ErrorKind::Other, "Internal write error occurred."));
        }

        Ok(written as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        unsafe {
            vex_sdk::vexTasksRun();
        }

        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr(())
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = 0;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stdout::new())
}
