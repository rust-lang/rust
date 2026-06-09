//@ compile-flags: -C opt-level=0

#![crate_type = "lib"]

#[repr(C)]
pub enum E {
    A,
}

// CHECK-LABEL: @index
#[no_mangle]
pub fn index(x: &[u32; 3], ind: E) -> u32 {
    // Canary: we should be able to optimize out the bounds check, but we need
    // to track the range of the discriminant result in order to be able to do that.
    // oli-obk tried to add that, but that caused miscompilations all over the place.
    // CHECK: panic_bounds_check
    x[ind as usize]
}
