// Previously, attempting to allocate with an alignment greater than 2^29 would cause miri to ICE
// because rustc does not support alignments that large.
// https://github.com/rust-lang/miri/issues/3687

#![feature(rustc_attrs)]
#![feature(ptr_alignment_type)]

extern "Rust" {
    #[rustc_std_internal_symbol]
    fn __rust_alloc(size: usize, align: core::mem::Alignment) -> *mut u8;
}

fn main() {
    unsafe {
        __rust_alloc(1, (1 << 30).try_into().unwrap());
        //~^ERROR: exceeding rustc's maximum supported value
    }
}
