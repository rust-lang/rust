// compile-flags: -C opt-level=3 -C no-prepopulate-passes
// min-llvm-version: 15.0 (for opaque pointers)

#![crate_type = "lib"]

// Test for the absence of `readonly` on the argument when it is mutated via `&raw const`.
// See <https://github.com/rust-lang/rust/issues/111502>.

// CHECK: i8 @foo(ptr noalias nocapture noundef align 1 dereferenceable(128) %x)
#[no_mangle]
pub fn foo(x: [u8; 128]) -> u8 {
    let ptr = core::ptr::addr_of!(x).cast_mut();
    unsafe {
        (*ptr)[0] = 1;
    }
    x[0]
}

// CHECK: i1 @second(ptr noalias nocapture noundef align {{[0-9]+}} dereferenceable({{[0-9]+}}) %a_ptr_and_b)
#[no_mangle]
pub unsafe fn second(a_ptr_and_b: (*mut (i32, bool), (i64, bool))) -> bool {
    let b_bool_ptr = core::ptr::addr_of!(a_ptr_and_b.1.1).cast_mut();
    (*b_bool_ptr) = true;
    a_ptr_and_b.1.1
}

// If going through a deref (and there are no other mutating accesses), then `readonly` is fine.
// CHECK: i1 @third(ptr noalias nocapture noundef readonly align {{[0-9]+}} dereferenceable({{[0-9]+}}) %a_ptr_and_b)
#[no_mangle]
pub unsafe fn third(a_ptr_and_b: (*mut (i32, bool), (i64, bool))) -> bool {
    let b_bool_ptr = core::ptr::addr_of!((*a_ptr_and_b.0).1).cast_mut();
    (*b_bool_ptr) = true;
    a_ptr_and_b.1.1
}
