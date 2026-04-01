//! regression test for <https://github.com/rust-lang/rust/issues/148467>
//! Ensure the diagnostic suggests `for &(mut x) ...` (parenthesized) instead of `&mut x`.

fn main() {
    let nums: &[u32] = &[1, 2, 3];
    for &num in nums {
        num *= 2; //~ ERROR cannot assign twice to immutable variable `num`
        println!("{num}");
    }
}
