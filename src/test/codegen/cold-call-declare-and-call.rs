// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(rust_cold_cc)]

// CHECK: define coldcc void @this_should_never_happen(i16
// CHECK: call coldcc void @this_should_never_happen(i16

#[no_mangle]
pub extern "rust-cold" fn this_should_never_happen(x: u16) {}

pub fn do_things(x: u16) {
    if x == 12345 {
        this_should_never_happen(54321);
    }
}
