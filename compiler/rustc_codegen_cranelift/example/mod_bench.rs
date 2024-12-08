#![feature(start, core_intrinsics, lang_items)]
#![allow(internal_features)]
#![no_std]

#[cfg_attr(unix, link(name = "c"))]
#[cfg_attr(target_env = "msvc", link(name = "msvcrt"))]
extern "C" {}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo<'_>) -> ! {
    core::intrinsics::abort();
}

#[lang = "eh_personality"]
fn eh_personality() {}

// Required for rustc_codegen_llvm
#[no_mangle]
unsafe extern "C" fn _Unwind_Resume() {
    core::intrinsics::unreachable();
}

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize {
    for i in 2..10_000_000 {
        black_box((i + 1) % i);
    }

    0
}

#[inline(never)]
fn black_box(i: u32) {
    if i != 1 {
        core::intrinsics::abort();
    }
}
