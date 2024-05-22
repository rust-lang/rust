//@ compile-flags: -Z span_free_formats -C debuginfo=full

// Tests that MIR inliner can handle closure arguments. (#45894)

fn main() {
    println!("{}", foo(0, 14));
}

// EMIT_MIR inline_closure.foo.Inline.after.mir
fn foo<T: Copy>(_t: T, q: i32) -> i32 {
    let x = |_t, _q| _t;

    // CHECK-LABEL: fn foo(
    // CHECK: (inlined foo::<T>::{closure#0})
    x(q, q)
}
