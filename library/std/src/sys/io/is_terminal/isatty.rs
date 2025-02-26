use crate::os::fd::{AsFd, AsRawFd};

pub fn is_terminal(fd: &impl AsFd) -> bool {
    let fd = fd.as_fd();
    unsafe { libc::isatty(fd.as_raw_fd()) != 0 }
}
