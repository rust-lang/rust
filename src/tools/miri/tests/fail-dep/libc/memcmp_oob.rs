// We could return a result after just the first 2 bytes, but we still require the given length to
// fully be inbounds.
fn main() {
    unsafe {
        libc::memcmp(b"123".as_ptr().cast(), b"132".as_ptr().cast(), 13); //~ERROR: attempting to access 13 bytes
    }
}
