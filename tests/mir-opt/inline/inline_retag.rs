//@ compile-flags: -Z span_free_formats -Z mir-emit-retag -C debuginfo=full

// Tests that MIR inliner fixes up `Retag`'s `fn_entry` flag

fn main() {
    println!("{}", bar());
}

// EMIT_MIR inline_retag.bar.Inline.after.mir
fn bar() -> bool {
    // CHECK-LABEL: fn bar(
    // CHECK: (inlined foo)
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: bb0: {
    // CHECK: Retag
    // CHECK: Retag
    // CHECK: Retag([[x]]);
    // CHECK: Retag([[y]]);
    // CHECK: return;
    // CHECK-NEXT: }
    let f = foo;
    f(&1, &-1)
}

#[inline(always)]
fn foo(x: &i32, y: &i32) -> bool {
    *x == *y
}
