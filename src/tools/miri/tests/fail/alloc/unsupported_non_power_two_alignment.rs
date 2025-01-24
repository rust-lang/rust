// Test non-power-of-two alignment.
extern "Rust" {
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
}

fn main() {
    unsafe {
        __rust_alloc(1, 3);
        //~^ERROR: creating allocation with non-power-of-two alignment
    }
}
