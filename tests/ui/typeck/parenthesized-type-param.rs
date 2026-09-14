//! Regression test for <https://github.com/rust-lang/rust/issues/23589>.
//! Test that we don't ICE on parenthesized type params.

fn main() {
    let v: Vec(&str) = vec!['1', '2'];
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| ERROR mismatched types
}
