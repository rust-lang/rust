// compile-flags: -O

#![crate_type = "lib"]

pub enum Foo {
    A, B
}

// CHECK-LABEL: @lookup
#[no_mangle]
pub fn lookup(buf: &[u8; 2], f: Foo) -> u8 {
    // CHECK-NOT: panic_bounds_check
    buf[f as usize]
}
