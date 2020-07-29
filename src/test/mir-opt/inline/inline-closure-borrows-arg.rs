// compile-flags: -Z span_free_formats

// Tests that MIR inliner can handle closure arguments,
// even when (#45894)

fn main() {
    println!("{}", foo(0, &14));
}

// EMIT_MIR inline_closure_borrows_arg.foo.Inline.after.mir
fn foo<T: Copy>(_t: T, q: &i32) -> i32 {
    let x = |r: &i32, _s: &i32| {
        let variable = &*r;
        *variable
    };
    x(q, q)
}
