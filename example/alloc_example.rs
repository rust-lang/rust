#![feature(start, core_intrinsics, alloc_error_handler, lang_items)]
#![no_std]

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
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
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

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let world: Box<&str> = Box::new("Hello World!\0");
    unsafe {
        puts(*world as *const str as *const u8);
    }

    0
}
