use libc;
use io;
use sys_common::backtrace::output;

#[inline(never)]
pub fn write(w: &mut io::Write) -> io::Result<()> {
    output(w, 0, 0 as *mut libc::c_void, None)
}
