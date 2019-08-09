// compile-flags: -C panic=abort
// no-prefer-dynamic

#![no_std]
#![crate_type = "staticlib"]
#![feature(panic_handler, alloc_error_handler)]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

extern crate alloc;

#[global_allocator]
static A: MyAlloc = MyAlloc;

struct MyAlloc;

unsafe impl core::alloc::GlobalAlloc for MyAlloc {
    unsafe fn alloc(&self, _: core::alloc::Layout) -> *mut u8 { 0 as _ }
    unsafe fn dealloc(&self, _: *mut u8, _: core::alloc::Layout) {}
}
