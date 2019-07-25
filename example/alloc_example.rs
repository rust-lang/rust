#![feature(start, box_syntax, core_intrinsics, alloc_error_handler)]
#![no_std]

#[link(name = "c")]
extern "C" {
    fn puts(s: *const u8);
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::intrinsics::abort();
    }
}

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    extern "C" {
        fn __rust_u128_mulo(a: u128, b: u128) -> (u128, bool);
    }

    assert_eq!(unsafe { __rust_u128_mulo(353985398u128,  932490u128).0 }, 330087843781020u128);
    0
}
