// Regression test for issue #98833.
// compile-flags: -Zinline-mir

fn main() {
    println!("{}", live::<false>());
}

#[inline]
fn live<const B: bool>() -> u32 {
    if B {
        dead()
    } else {
        0
    }
}

#[inline]
fn dead() -> u32 {
    42
}
