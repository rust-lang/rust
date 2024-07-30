//@ revisions: normal windows
//@ compile-flags: -C no-prepopulate-passes
//@[normal] ignore-windows
//@[windows] only-windows
//@[windows] only-x86_64

#![crate_type = "lib"]
#![feature(rust_cold_cc)]

// wasm marks the definition as `dso_local`, so allow that as optional.

// CHECK-NORMAL: define{{( dso_local)?}} preserve_mostcc void @this_should_never_happen(i16
// CHECK-NORMAL: call preserve_mostcc void @this_should_never_happen(i16

// See the comment in `Target::adjust_abi` for why this differs

// CHECK-WINDOWS: define void @this_should_never_happen(i16
// CHECK-WINDOWS: call void @this_should_never_happen(i16

#[no_mangle]
pub extern "rust-cold" fn this_should_never_happen(x: u16) {}

pub fn do_things(x: u16) {
    if x == 12345 {
        this_should_never_happen(54321);
    }
}
