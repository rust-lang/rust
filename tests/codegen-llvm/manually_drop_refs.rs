//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
#![crate_type = "lib"]

use std::mem::ManuallyDrop;

// CHECK: define noundef nonnull align 1 ptr @f(ptr noalias noundef nonnull readnone returned align 1 {{(captures\(ret: address, provenance\))?}} %x) unnamed_addr
#[no_mangle]
pub fn f(x: ManuallyDrop<Box<u8>>) -> ManuallyDrop<Box<u8>> {
    x
}

// CHECK: define noundef nonnull align 1 dereferenceable(1) ptr @g(ptr noalias noundef readonly returned align 1 {{(captures\(ret: address, read_provenance\))?}} dereferenceable(1) %x) unnamed_addr
#[no_mangle]
pub fn g(x: ManuallyDrop<&u8>) -> ManuallyDrop<&u8> {
    x
}

// CHECK: define noundef nonnull align 1 dereferenceable(1) ptr @h(ptr noalias noundef readnone returned align 1 {{(captures\(ret: address, provenance\))?}} dereferenceable(1) %x) unnamed_addr
#[no_mangle]
pub fn h(x: ManuallyDrop<&mut u8>) -> ManuallyDrop<&mut u8> {
    x
}
