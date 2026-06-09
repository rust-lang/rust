//@only-target: linux android illumos

// This is a test for registering unsupported fd with epoll.
// Register epoll fd with epoll is allowed in real system, but we do not support this.
fn main() {
    // Create two epoll instance.
    let epfd0 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd0, -1);
    let epfd1 = unsafe { libc::epoll_create1(0) };
    assert_ne!(epfd1, -1);

    // Register epoll with epoll.
    let mut ev =
        libc::epoll_event { events: (libc::EPOLLIN | libc::EPOLLET) as _, u64: epfd1 as u64 };
    let res = unsafe { libc::epoll_ctl(epfd0, libc::EPOLL_CTL_ADD, epfd1, &mut ev) };
    //~^ERROR: epoll does not support this file description
    assert_eq!(res, 0);
}
