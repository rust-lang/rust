//@ test-mir-pass: InstSimplify
#![crate_type = "lib"]
#![feature(raw_ref_op)]

// For each of these, only 2 of the 6 should simplify,
// as the others have the wrong types.

// EMIT_MIR ref_of_deref.references.InstSimplify.diff
// CHECK-LABEL: references
pub fn references(const_ref: &i32, mut_ref: &mut [i32]) {
    // CHECK: _3 = _1;
    let _a = &*const_ref;
    // CHECK: _4 = &(*_2);
    let _b = &*mut_ref;
    // CHECK: _5 = _2;
    let _c = &mut *mut_ref;
    // CHECK: _6 = &raw const (*_1);
    let _d = &raw const *const_ref;
    // CHECK: _7 = &raw const (*_2);
    let _e = &raw const *mut_ref;
    // CHECK: _8 = &raw mut (*_2);
    let _f = &raw mut *mut_ref;
}

// EMIT_MIR ref_of_deref.pointers.InstSimplify.diff
// CHECK-LABEL: pointers
pub unsafe fn pointers(const_ptr: *const [i32], mut_ptr: *mut i32) {
    // CHECK: _3 = &(*_1);
    let _a = &*const_ptr;
    // CHECK: _4 = &(*_2);
    let _b = &*mut_ptr;
    // CHECK: _5 = &mut (*_2);
    let _c = &mut *mut_ptr;
    // CHECK: _6 = _1;
    let _d = &raw const *const_ptr;
    // CHECK: _7 = &raw const (*_2);
    let _e = &raw const *mut_ptr;
    // CHECK: _8 = _2;
    let _f = &raw mut *mut_ptr;
}
