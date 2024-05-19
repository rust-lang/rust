// Regression test for #114918
// Test that a const generic enclosed in a block in a struct's type arg
// produces a type mismatch error instead of triggering a const eval cycle

#[allow(unused_braces)]
struct S<const N: usize> {
        arr: [u8; N]
}

fn main() {
    let s = S::<{ () }> { arr: [5, 6, 7]}; //~ ERROR mismatched types
}
