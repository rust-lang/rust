// compile-flags: -Z span_free_formats

// Tests that MIR inliner can handle closure captures.

fn main() {
    println!("{:?}", foo(0, 14));
}

// EMIT_MIR inline_closure_captures.foo.Inline.after.mir
#[inline(never)]
fn foo<T: Copy>(t: T, q: i32) -> (i32, T) {
    let x = |_q| (q, t);
    x(q)
}
