//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(rust_preserve_none_cc)]

// CHECK: define{{( dso_local)?}} preserve_nonecc void @peach(i16
#[no_mangle]
#[inline(never)]
pub extern "rust-preserve-none" fn peach(x: u16) {
    panic!("unwinding works too")
}

// CHECK: call preserve_nonecc void @peach(i16
pub fn quince(x: u16) {
    if x == 12345 {
        peach(54321);
    }
}
