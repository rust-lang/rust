//@ revisions: NORMAL WIN
//@ compile-flags: -C no-prepopulate-passes
//@[NORMAL] ignore-windows
//@[WIN] only-windows
//@[WIN] only-x86_64

#![crate_type = "lib"]
#![feature(rust_cold_cc)]

// wasm marks the definition as `dso_local`, so allow that as optional.

// NORMAL: define{{( dso_local)?}} preserve_mostcc void @this_should_never_happen(i16
// NORMAL: call preserve_mostcc void @this_should_never_happen(i16

// See the comment in `Target::adjust_abi` for why this differs

// WIN: define void @this_should_never_happen(i16
// WIN: call void @this_should_never_happen(i16

#[no_mangle]
pub extern "rust-cold" fn this_should_never_happen(x: u16) {}

pub fn do_things(x: u16) {
    if x == 12345 {
        this_should_never_happen(54321);
    }
}
