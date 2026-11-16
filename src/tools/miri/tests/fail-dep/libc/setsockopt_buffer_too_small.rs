//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation

fn main() {
    let sockfd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
    assert!(sockfd >= 0);

    // The buffer is large enough for the value, but the length claims it is much larger.
    let val: libc::c_int = 1;
    unsafe {
        libc::setsockopt(
            //~^ERROR: memory access failed
            sockfd,
            libc::SOL_SOCKET,
            libc::SO_REUSEADDR,
            (&raw const val).cast(),
            1024,
        );
    }
}
