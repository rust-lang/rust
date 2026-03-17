//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

fn main() {
    let sockfd =
        unsafe { errno_result(libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)).unwrap() };
    let addr = net::ipv4_sock_addr(net::IPV4_LOCALHOST, 2345);
    unsafe {
        errno_check(libc::bind(
            sockfd,
            (&addr as *const libc::sockaddr_in).cast::<libc::sockaddr>(),
            size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ));
    }

    // Only supported value is 128, thus we use 127 to trigger the warning.
    let backlog = 127;

    unsafe {
        let errno = libc::listen(sockfd, backlog);
        errno_check(errno);
    }
}
