//! This test checks that match branches which all access a field
//! at the same offset are merged together.
//!
//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[repr(C)]
pub struct A {
    x: f64,
    y: u64,
}
#[repr(C)]
pub struct B {
    x: f64,
    y: u32,
}
#[repr(C)]
pub struct C {
    x: f64,
    y: u16,
}
#[repr(C)]
pub struct D {
    x: f64,
    y: u8,
}

pub enum E {
    A(A),
    B(B),
    C(C),
    D(D),
}

// CHECK-LABEL: @match_on_e
#[no_mangle]
pub fn match_on_e(e: &E) -> &f64 {
    // CHECK: start:
    // CHECK-NEXT: getelementptr
    // CHECK-NEXT: ret
    match e {
        E::A(A { x, .. }) | E::B(B { x, .. }) | E::C(C { x, .. }) | E::D(D { x, .. }) => x,
    }
}
