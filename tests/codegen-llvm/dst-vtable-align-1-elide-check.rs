//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
//@ min-llvm-version: 21

#![crate_type = "lib"]

pub trait Trait {
    fn f(&self);
}

pub struct WrapperWithAlign1<T: ?Sized> {
    x: u8,
    y: T,
}

pub struct Struct<W: ?Sized> {
    _field: i8,
    dst: W,
}

// CHECK-LABEL: @eliminates_runtime_check_when_align_1
#[no_mangle]
pub fn eliminates_runtime_check_when_align_1(
    x: &Struct<WrapperWithAlign1<dyn Trait>>,
) -> &WrapperWithAlign1<dyn Trait> {
    // CHECK: load i{{[0-9]+}}, {{.+}} !range
    // CHECK-NOT: llvm.umax
    // CHECK-NOT: select
    // CHECK: ret
    &x.dst
}
