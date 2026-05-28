// Previously, attempting to allocate with an alignment greater than 2^29 would cause miri to ICE
// because rustc does not support alignments that large.
// https://github.com/rust-lang/miri/issues/3687

#![feature(rustc_attrs)]

extern "Rust" {
    #[rustc_std_internal_symbol]
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
}

fn main() {
    unsafe {
        __rust_alloc(1, 1 << 30);
        //~^ERROR: exceeding rustc's maximum supported value
    }
}
