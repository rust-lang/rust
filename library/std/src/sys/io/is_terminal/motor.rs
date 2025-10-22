use crate::os::fd::{AsFd, AsRawFd};

pub fn is_terminal(fd: &impl AsFd) -> bool {
    let fd = fd.as_fd();
    moto_rt::fs::is_terminal(fd.as_raw_fd())
}
