//@only-target: linux android illumos

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

// This is a test for registering unsupported fd with epoll.
// Register epoll fd with epoll is allowed in real system, but we do not support this.
fn main() {
    // Create two epoll instance.
    let epfd0 = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();
    let epfd1 = errno_result(unsafe { libc::epoll_create1(0) }).unwrap();

    // Register epoll with epoll.
    let mut ev =
        libc::epoll_event { events: (libc::EPOLLIN | libc::EPOLLET) as _, u64: epfd1 as u64 };
    let res = unsafe { libc::epoll_ctl(epfd0, libc::EPOLL_CTL_ADD, epfd1, &mut ev) };
    //~^ERROR: I/O readiness watching not supported
    assert_eq!(res, 0);
}
