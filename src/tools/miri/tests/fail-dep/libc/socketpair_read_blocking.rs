//@ignore-target: windows # no libc socketpair on Windows

// This is temporarily here because blocking on fd is not supported yet.
// When blocking is eventually supported, this will be moved to pass-dep/libc/libc-socketpair

fn main() {
    let mut fds = [-1, -1];
    let _ = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    // The read below will be blocked because the buffer is empty.
    let mut buf: [u8; 3] = [0; 3];
    let _res = unsafe { libc::read(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) }; //~ERROR: blocking isn't supported
}
