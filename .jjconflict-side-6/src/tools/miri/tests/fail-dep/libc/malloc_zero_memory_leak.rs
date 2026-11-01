fn main() {
    unsafe {
        let _ptr = libc::malloc(0); //~ERROR: memory leak
    }
}
