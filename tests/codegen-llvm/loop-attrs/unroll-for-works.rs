//@ compile-flags: -O

#![crate_type = "lib"]
#![feature(loop_hints)]

unsafe extern "C" {
    fn maybe_has_side_effect();
}

#[no_mangle]
pub fn unroll_full() {
    // CHECK-LABEL: @unroll_full
    // CHECK-COUNT-512: tail call void @maybe_has_side_effect()
    #[unroll(full)]
    for _ in 0..512 {
        unsafe { maybe_has_side_effect() }
    }
}

#[no_mangle]
pub fn unroll_never() {
    // CHECK-LABEL: @unroll_never
    // CHECK: tail call void @maybe_has_side_effect()
    // CHECK-NOT: tail call void @maybe_has_side_effect()
    #[unroll(never)]
    for _ in 0..3 {
        unsafe { maybe_has_side_effect() }
    }
}

#[no_mangle]
pub fn unroll_count() {
    // CHECK-LABEL: @unroll_count
    // CHECK-COUNT-5: tail call void @maybe_has_side_effect()
    #[unroll(5)]
    for _ in 0..10 {
        unsafe { maybe_has_side_effect() }
    }
}
