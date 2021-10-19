// compile-flags: -O

#![crate_type = "lib"]

#[repr(C)]
pub enum E {
    A,
}

// CHECK-LABEL: @index
#[no_mangle]
pub fn index(x: &[u32; 3], ind: E) -> u32{
    // CHECK-NOT: panic_bounds_check
    x[ind as usize]
}
