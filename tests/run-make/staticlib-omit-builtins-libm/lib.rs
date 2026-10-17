#![no_std]
#![crate_type = "staticlib"]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

// Declaring `ceilf`/`sqrtf` pulls compiler-builtins' weak definitions into the archive, while
// `#[link(name = "m")]` makes the staticlib record system libm (`-lm`) as a dependency. When
// libm is recorded, rustc must skip those weak definitions so that a later link against `-lm`
// resolves the strong libm versions (rust-lang/rust#142119).
#[link(name = "m")]
extern "C" {
    fn ceilf(x: f32) -> f32;
    fn sqrtf(x: f32) -> f32;
}

#[no_mangle]
pub extern "C" fn use_mathf(x: f32) -> f32 {
    unsafe { ceilf(x) + sqrtf(x) }
}
