//@ add-core-stubs
//@ compile-flags: --target x86_64-pc-windows-msvc
//@ compile-flags: --crate-type lib --emit link
//@ needs-llvm-components: x86
#![no_core]
#![feature(no_core)]
extern crate minicore;

// It may seem weird this is a cross-platform-testable thing, since doesn't it test linkage?
// However the main thing we are testing is an *error*, so it works fine!

#[link(name = "foo", kind = "raw-dylib")]
extern "stdcall" {
//~^ WARN: calling convention not supported on this target
//~| WARN: previously accepted
    fn f(x: i32);
    //~^ ERROR ABI not supported by `#[link(kind = "raw-dylib")]` on this architecture
}

pub fn lib_main() {
    unsafe { f(42); }
}
