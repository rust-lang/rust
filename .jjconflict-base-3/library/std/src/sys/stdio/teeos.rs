#![deny(unsafe_op_in_unsafe_fn)]

#[expect(dead_code)]
#[path = "unsupported.rs"]
mod unsupported_stdio;

use core::arch::asm;

use crate::io;

pub type Stdin = unsupported_stdio::Stdin;
pub struct Stdout;
pub type Stderr = Stdout;

const KCALL_DEBUG_CMD_PUT_BYTES: i64 = 2;

unsafe fn debug_call(cap_ref: u64, call_no: i64, arg1: u64, arg2: u64) -> i32 {
    let ret: u64;
    unsafe {
        asm!(
            "svc #99",
            inout("x0") cap_ref => ret,
            in("x1") call_no,
            in("x2") arg1,
            in("x3") arg2,
        );
    }

    ret as i32
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Corresponds to `HM_DEBUG_PUT_BYTES_LIMIT`.
        const MAX_LEN: usize = 512;
        let len = buf.len().min(MAX_LEN);
        let result =
            unsafe { debug_call(0, KCALL_DEBUG_CMD_PUT_BYTES, buf.as_ptr() as u64, len as u64) };

        if result == 0 { Ok(len) } else { Err(io::Error::from(io::ErrorKind::InvalidInput)) }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = unsupported_stdio::STDIN_BUF_SIZE;

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(libc::EBADF as i32)
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
