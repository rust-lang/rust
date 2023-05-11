// Tests inlining of diverging calls.
//
// ignore-wasm32-bare compiled with panic=abort by default
#![crate_type = "lib"]

// EMIT_MIR inline_diverging.f.Inline.diff
pub fn f() {
    sleep();
}

// EMIT_MIR inline_diverging.g.Inline.diff
pub fn g(i: i32) -> u32 {
    if i > 0 {
        i as u32
    } else {
        panic();
    }
}

// EMIT_MIR inline_diverging.h.Inline.diff
pub fn h() {
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
