// Regression test for issue #98833.
//@ compile-flags: -Zinline-mir -Cdebug-assertions=off

fn main() {
    println!("{}", live::<false>());

    let f = |x: bool| {
        debug_assert!(x);
    };
    f(false);
}

#[inline]
fn live<const B: bool>() -> u32 {
    if B {
        dead() //
    } else {
        0
    }
}

#[inline]
fn dead() -> u32 {
    42
}
