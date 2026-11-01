fn main() {
    unsafe {
        let p1 = libc::malloc(20);
        // C made this UB...
        let p2 = libc::realloc(p1, 0); //~ERROR: `realloc` with a size of zero
        assert!(p2.is_null());
    }
}
