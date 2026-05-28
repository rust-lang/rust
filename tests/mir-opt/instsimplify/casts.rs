//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ compile-flags: -Zinline-mir
#![crate_type = "lib"]
#![feature(core_intrinsics)]

#[inline(always)]
fn generic_cast<T, U>(x: *const T) -> *const U {
    x as *const U
}

// EMIT_MIR casts.redundant.InstSimplify-after-simplifycfg.diff
pub fn redundant<'a, 'b: 'a>(x: *const &'a u8) -> *const &'a u8 {
    // CHECK-LABEL: fn redundant(
    // CHECK: inlined generic_cast
    // CHECK-NOT: as
    generic_cast::<&'a u8, &'b u8>(x) as *const &'a u8
}

// EMIT_MIR casts.roundtrip.InstSimplify-after-simplifycfg.diff
pub fn roundtrip(x: *const u8) -> *const u8 {
    // CHECK-LABEL: fn roundtrip(
    // CHECK: _4 = copy _1;
    // CHECK: _3 = move _4 as *mut u8 (PtrToPtr);
    // CHECK: _2 = move _3 as *const u8 (PtrToPtr);
    x as *mut u8 as *const u8
}

// EMIT_MIR casts.roundtrip.InstSimplify-after-simplifycfg.diff
pub fn cast_thin_via_aggregate(x: *const u8) -> *const () {
    // CHECK-LABEL: fn cast_thin_via_aggregate(
    // CHECK: _2 = copy _1;
    // CHECK: _0 = move _2 as *const () (PtrToPtr);
    std::intrinsics::aggregate_raw_ptr(x, ())
}
