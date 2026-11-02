fn main() {
    std::thread::spawn(|| unsafe {
        // Have main thread exit first, for a consistent error location.
        for _ in 0..64 {
            std::thread::yield_now();
        }
        libc::exit(0);
        //~^ERROR: `exit` called more than once
    });
}
