//@compile-flags: -Cpanic=abort
//@normalize-stderr-test: "OS `.*`" -> "$$OS"
// Make sure we pretend the allocation symbols don't exist when there is no allocator

#![feature(start)]
#![no_std]

extern "Rust" {
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
}

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    unsafe {
        __rust_alloc(1, 1); //~ERROR: unsupported operation: can't call foreign function `__rust_alloc`
    }

    0
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
