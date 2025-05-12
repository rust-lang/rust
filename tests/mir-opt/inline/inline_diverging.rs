// Tests inlining of diverging calls.
//
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -Zinline-mir-hint-threshold=1000 -C debuginfo=full
#![crate_type = "lib"]

// EMIT_MIR inline_diverging.f.Inline.diff
pub fn f() {
    // CHECK-LABEL: fn f(
    // CHECK: (inlined sleep)
    sleep();
}

// EMIT_MIR inline_diverging.g.Inline.diff
pub fn g(i: i32) -> u32 {
    if i > 0 {
        i as u32
    } else {
        // CHECK-LABEL: fn g(
        // CHECK: (inlined panic)
        panic();
    }
}

// EMIT_MIR inline_diverging.h.Inline.diff
pub fn h() {
    // CHECK-LABEL: fn h(
    // CHECK: (inlined call_twice::<!, fn() -> ! {sleep}>)
    // CHECK: (inlined <fn() -> ! {sleep} as Fn<()>>::call - shim(fn() -> ! {sleep}))
    // CHECK: (inlined sleep)
    call_twice(sleep);
}

#[inline(always)]
pub fn call_twice<R, F: Fn() -> R>(f: F) -> (R, R) {
    let a = f();
    let b = f();
    (a, b)
}

#[inline(always)]
fn panic() -> ! {
    panic!();
}

#[inline(always)]
fn sleep() -> ! {
    loop {}
}
