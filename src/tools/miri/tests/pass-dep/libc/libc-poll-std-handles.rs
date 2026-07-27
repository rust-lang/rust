//! Ensure we can poll the std handles, both the normal ones and the "null" ones.
//@ignore-target: windows # no libc
//@revisions: normal null
//@[null]compile-flags: -Zmiri-mute-stdout-stderr
//@run-native

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

fn main() {
    let pfds: &mut [_] = &mut [
        libc::pollfd { fd: 0, events: libc::POLLOUT | libc::POLLIN, revents: 0 },
        libc::pollfd { fd: 1, events: libc::POLLOUT | libc::POLLIN, revents: 0 },
        libc::pollfd { fd: 2, events: libc::POLLOUT | libc::POLLIN, revents: 0 },
    ];
    let num = errno_result(unsafe { libc::poll(pfds.as_mut_ptr(), 3, 0) }).unwrap();
    assert_eq!(num, 3);

    assert_eq!(pfds[0].revents, libc::POLLIN | libc::POLLOUT);
    assert_eq!(pfds[1].revents, libc::POLLOUT);
    assert_eq!(pfds[2].revents, libc::POLLOUT);
}
