//@ignore-target: windows # no libc socketpair on Windows
// This is temporarily here because blocking on fd is not supported yet.
// When blocking is eventually supported, this will be moved to pass-dep/libc/libc-socketpair
fn main() {
    let mut fds = [-1, -1];
    let _ = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    // Write size > buffer capacity
    // Used up all the space in the buffer.
    let arr1: [u8; 212992] = [1; 212992];
    let _ = unsafe { libc::write(fds[0], arr1.as_ptr() as *const libc::c_void, 212992) };
    let data = "abc".as_bytes().as_ptr();
    // The write below will be blocked as the buffer is full.
    let _ = unsafe { libc::write(fds[0], data as *const libc::c_void, 3) }; //~ERROR: blocking isn't supported
    let mut buf: [u8; 3] = [0; 3];
    let _res = unsafe { libc::read(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
}
