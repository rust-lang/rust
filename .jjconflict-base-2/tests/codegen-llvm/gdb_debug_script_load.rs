//
//@ ignore-windows
//@ ignore-apple
//@ ignore-wasm
//@ ignore-emscripten

//@ compile-flags: -g -C no-prepopulate-passes -Cpanic=abort

#![feature(lang_items)]
#![no_std]

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
extern "C" fn rust_eh_personality() {
    loop {}
}

// Needs rustc to generate `main` as that's where the magic load is inserted.
// IOW, we cannot write this test with `#![no_main]`.
// CHECK-LABEL: @main
// CHECK: load volatile i8, {{.+}} @__rustc_debug_gdb_scripts_section__

#[lang = "start"]
fn lang_start<T: 'static>(
    _main: fn() -> T,
    _argc: isize,
    _argv: *const *const u8,
    _sigpipe: u8,
) -> isize {
    return 0;
}

fn main() {}
