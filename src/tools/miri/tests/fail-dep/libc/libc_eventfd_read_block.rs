//@only-target: linux
fn main() {
    // eventfd read will block when EFD_NONBLOCK flag is clear and counter = 0.
    // This will pass when blocking is implemented.
    let flags = libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
    let mut buf: [u8; 8] = [0; 8];
    let _res: i32 = unsafe {
        libc::read(fd, buf.as_mut_ptr().cast(), buf.len() as libc::size_t).try_into().unwrap() //~ERROR: blocking is unsupported
    };
}
