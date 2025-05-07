#![feature(rustc_attrs)]

extern "Rust" {
    #[rustc_std_internal_symbol]
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
}

fn main() {
    let bytes = isize::MAX as usize + 1;
    unsafe {
        __rust_alloc(bytes, 1); //~ERROR: larger than half the address space
    }
}
