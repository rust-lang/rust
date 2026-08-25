#![feature(rustc_attrs)]
#![feature(ptr_alignment_type)]

extern "Rust" {
    #[rustc_std_internal_symbol]
    fn __rust_alloc(size: usize, align: core::mem::Alignment) -> *mut u8;
}

fn main() {
    let bytes = isize::MAX as usize + 1;
    unsafe {
        __rust_alloc(bytes, 1usize.try_into().unwrap()); //~ERROR: larger than half the address space
    }
}
