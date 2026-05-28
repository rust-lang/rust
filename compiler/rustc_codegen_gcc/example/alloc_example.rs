#![feature(core_intrinsics, alloc_error_handler, lang_items)]
#![no_std]
#![no_main]
#![allow(internal_features)]

extern crate alloc;
extern crate alloc_system;

use alloc::boxed::Box;

use alloc_system::System;

#[global_allocator]
static ALLOC: System = System;

#[link(name = "c")]
extern "C" {
    fn puts(s: *const u8) -> i32;
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo<'_>) -> ! {
    core::intrinsics::abort();
}

#[alloc_error_handler]
fn alloc_error_handler(_: alloc::alloc::Layout) -> ! {
    core::intrinsics::abort();
}

#[lang = "eh_personality"]
fn eh_personality() -> ! {
    loop {}
}

#[no_mangle]
unsafe extern "C" fn _Unwind_Resume() {
    core::intrinsics::unreachable();
}

#[no_mangle]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const u8) -> core::ffi::c_int {
    let world: Box<&str> = Box::new("Hello World!\0");
    unsafe {
        puts(*world as *const str as *const u8);
    }

    0
}
