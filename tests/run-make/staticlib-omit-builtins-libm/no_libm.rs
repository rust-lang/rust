#![no_std]
#![crate_type = "staticlib"]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

// No `#[link(name = "m")]`: without a system libm recorded, compiler-builtins' weak f32/f64 math
// definitions are the only provider of `ceilf`/`sqrtf`, so they must be kept.
extern "C" {
    fn ceilf(x: f32) -> f32;
    fn sqrtf(x: f32) -> f32;
}

#[no_mangle]
pub extern "C" fn use_mathf(x: f32) -> f32 {
    unsafe { ceilf(x) + sqrtf(x) }
}
