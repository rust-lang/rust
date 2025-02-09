//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

pub enum Foo {
    A,
    B,
}

// CHECK-LABEL: @lookup
#[no_mangle]
pub fn lookup(buf: &[u8; 2], f: Foo) -> u8 {
    // CHECK-NOT: panic_bounds_check
    buf[f as usize]
}

pub enum Bar {
    A = 2,
    B = 3,
}

// CHECK-LABEL: @lookup_unmodified
#[no_mangle]
pub fn lookup_unmodified(buf: &[u8; 5], f: Bar) -> u8 {
    // CHECK-NOT: panic_bounds_check
    buf[f as usize]
}
