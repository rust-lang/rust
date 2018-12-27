// compile-flags: -Z span_free_formats

// Tests that MIR inliner works for any operand

fn main() {
    println!("{}", bar());
}

#[inline(always)]
fn foo(x: i32, y: i32) -> bool {
    x == y
}

fn bar() -> bool {
    let f = foo;
    f(1, -1)
}

// END RUST SOURCE
// START rustc.bar.Inline.after.mir
// ...
// bb0: {
//     ...
//     _0 = Eq(move _3, move _4);
//     ...
//     return;
// }
// ...
// END rustc.bar.Inline.after.mir
