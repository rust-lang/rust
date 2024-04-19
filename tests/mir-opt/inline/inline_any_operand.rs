//@ compile-flags: -Z span_free_formats -C debuginfo=full

// Tests that MIR inliner works for any operand

fn main() {
    println!("{}", bar());
}

// EMIT_MIR inline_any_operand.bar.Inline.after.mir
fn bar() -> bool {
    // CHECK-LABEL: fn bar(
    // CHECK: (inlined foo)
    let f = foo;
    f(1, -1)
}

#[inline(always)]
fn foo(x: i32, y: i32) -> bool {
    x == y
}
