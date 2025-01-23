//@ revisions: emscripten wasi seh
//@[emscripten] compile-flags: --target wasm32-unknown-emscripten -Z emscripten-wasm-eh
//@[wasi] compile-flags: --target wasm32-wasip1 -C panic=unwind
//@[seh] compile-flags: --target x86_64-pc-windows-msvc
//@[emscripten] needs-llvm-components: webassembly
//@[wasi] needs-llvm-components: webassembly
//@[seh] needs-llvm-components: x86

// Ensure a catch-all generates:
// - `catchpad ... [ptr null]` on Wasm (otherwise LLVM gets confused)
// - `catchpad ... [ptr null, i32 64, ptr null]` on Windows (otherwise we catch SEH exceptions)

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

unsafe extern "C-unwind" {
    safe fn unwinds();
}

#[lang = "panic_cannot_unwind"]
fn panic_cannot_unwind() -> ! {
    loop {}
}

#[no_mangle]
#[rustc_nounwind]
pub fn doesnt_unwind() {
    // emscripten: %catchpad = catchpad within %catchswitch [ptr null]
    // wasi: %catchpad = catchpad within %catchswitch [ptr null]
    // seh: %catchpad = catchpad within %catchswitch [ptr null, i32 64, ptr null]
    unwinds();
}
