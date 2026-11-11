//@ignore-target: windows # No libc IO on Windows

fn main() {
    let mut buf = [0u8; 4];
    let iov = [libc::iovec { iov_base: buf.as_mut_ptr().cast(), iov_len: 5 }];
    // Reading from stdin blocks or returns EOF, but we first check the buffer.
    unsafe { libc::readv(0, iov.as_ptr(), 1) };
    //~^ ERROR: only 4 bytes from the end of the allocation
}
