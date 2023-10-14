#![unstable(issue = "none", feature = "std_internals")]
use crate::custom_os_impl;
use crate::io;

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        custom_os_impl!(stdio, read_stdin, buf)
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        custom_os_impl!(stdio, write_stdout, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        custom_os_impl!(stdio, flush_stdout)
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        custom_os_impl!(stdio, write_stderr, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        custom_os_impl!(stdio, flush_stderr)
    }
}

pub const STDIN_BUF_SIZE: usize = 0;

pub fn is_ebadf(err: &io::Error) -> bool {
    custom_os_impl!(stdio, is_ebadf, err)
}

pub fn panic_output() -> Option<Vec<u8>> {
    custom_os_impl!(stdio, panic_output)
}
