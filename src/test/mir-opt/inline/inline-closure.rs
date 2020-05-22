// compile-flags: -Z span_free_formats

// Tests that MIR inliner can handle closure arguments. (#45894)

fn main() {
    println!("{}", foo(0, 14));
}

// EMIT_MIR rustc.foo.Inline.after.mir
fn foo<T: Copy>(_t: T, q: i32) -> i32 {
    let x = |_t, _q| _t;
    x(q, q)
}
