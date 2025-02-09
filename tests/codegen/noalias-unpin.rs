//@ compile-flags: -Copt-level=3 -Z mutable-noalias=yes

#![crate_type = "lib"]

pub struct SelfRef {
    self_ref: *mut SelfRef,
    _pin: std::marker::PhantomPinned,
}

// CHECK-LABEL: @test_self_ref(
// CHECK-NOT: noalias
#[no_mangle]
pub unsafe fn test_self_ref(s: &mut SelfRef) {
    (*s.self_ref).self_ref = std::ptr::null_mut();
}
