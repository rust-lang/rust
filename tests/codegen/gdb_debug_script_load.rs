//
//@ ignore-windows
//@ ignore-apple
//@ ignore-wasm
//@ ignore-emscripten

//@ compile-flags: -g -Cpanic=abort

#![no_std]
#![no_main]

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// CHECK: @llvm.used = {{.+}} @__rustc_debug_gdb_scripts_section
