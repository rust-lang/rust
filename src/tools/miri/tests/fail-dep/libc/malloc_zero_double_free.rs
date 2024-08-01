fn main() {
    unsafe {
        let ptr = libc::malloc(0);
        libc::free(ptr);
        libc::free(ptr); //~ERROR: dangling
    }
}
