// Checks that only functions with compatible attributes are inlined.
//@ only-x86_64
//@ compile-flags: -Cpanic=abort

#![crate_type = "lib"]
#![feature(no_sanitize)]
#![feature(c_variadic)]

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn sse2() {}

#[inline]
fn nop() {}

// CHECK-LABEL: fn f0()
// CHECK:       bb0: {
// CHECK-NEXT:  return;
#[target_feature(enable = "sse2")]
pub unsafe fn f0() {
    sse2();
}

// CHECK-LABEL: fn f1()
// CHECK:       bb0: {
// CHECK-NEXT:  sse2()
pub unsafe fn f1() {
    sse2();
}

// CHECK-LABEL: fn f2()
// CHECK:       bb0: {
// CHECK-NEXT:  nop()
#[target_feature(enable = "avx")]
pub unsafe fn f2() {
    nop();
}

#[inline]
#[no_sanitize(address)]
pub unsafe fn no_sanitize() {}

// CHECK-LABEL: fn inlined_no_sanitize()
// CHECK:       bb0: {
// CHECK-NEXT:  return;
#[no_sanitize(address)]
pub unsafe fn inlined_no_sanitize() {
    no_sanitize();
}

// CHECK-LABEL: fn not_inlined_no_sanitize()
// CHECK:       bb0: {
// CHECK-NEXT:  no_sanitize()
pub unsafe fn not_inlined_no_sanitize() {
    no_sanitize();
}

// CHECK-LABEL: fn not_inlined_c_variadic()
// CHECK:       bb0: {
// CHECK-NEXT:  StorageLive(_1)
// CHECK-NEXT:  _1 = sum
pub unsafe fn not_inlined_c_variadic() {
    let _ = sum(4u32, 4u32, 30u32, 200u32, 1000u32);
}

#[inline(always)]
#[no_mangle]
unsafe extern "C" fn sum(n: u32, mut vs: ...) -> u32 {
    let mut s = 0;
    let mut i = 0;
    while i != n {
        s += vs.arg::<u32>();
        i += 1;
    }
    s
}
