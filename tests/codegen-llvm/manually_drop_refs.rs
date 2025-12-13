//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
#![crate_type = "lib"]

use std::mem::ManuallyDrop;

// CHECK: define noundef nonnull ptr @f(ptr noundef nonnull readnone returned {{(captures\(ret: address, provenance\) )?}}%x) unnamed_addr
#[no_mangle]
pub fn f(x: ManuallyDrop<Box<u8>>) -> ManuallyDrop<Box<u8>> {
    x
}

// CHECK: define noundef nonnull ptr @g(ptr noundef nonnull readnone returned {{(captures\(ret: address, provenance\) )?}}%x) unnamed_addr
#[no_mangle]
pub fn g(x: ManuallyDrop<&u8>) -> ManuallyDrop<&u8> {
    x
}

// CHECK: define noundef nonnull ptr @h(ptr noundef nonnull readnone returned {{(captures\(ret: address, provenance\) )?}}%x) unnamed_addr
#[no_mangle]
pub fn h(x: ManuallyDrop<&mut u8>) -> ManuallyDrop<&mut u8> {
    x
}
