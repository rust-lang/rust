//@ compile-flags: -C debuginfo=full
// Verify that we do not ICE inlining a function which uses _0 as an index.
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

pub fn outer() -> usize {
    // CHECK-LABEL: fn outer(
    // CHECK: = {{.*}}[_0];
    inner()
}

#[inline(never)]
fn index() -> usize {
    loop {}
}

#[inline]
fn inner() -> usize {
    // CHECK-LABEL: fn inner(
    // CHECK: = {{.*}}[_0];
    let buffer = &[true];
    let index = index();
    if buffer[index] { index } else { 0 }
}

fn main() {
    outer();
}

// EMIT_MIR issue_106141.outer.Inline.diff
