//@ needs-unwind
#![feature(unsized_fn_params)]

#[inline(always)]
fn callee(y: [i32]) {}

// EMIT_MIR unsized_argument.caller.Inline.diff
fn caller(x: Box<[i32]>) {
    // CHECK-LABEL: fn caller(
    // CHECK-NOT: (inlined callee)
    callee(*x);
}

fn main() {
    let b = Box::new([1]);
    caller(b);
}
