//! Regression test for https://github.com/rust-lang/rust/issues/3521
//!
//@ run-rustfix
fn main() {
    let foo = 100;

    static y: isize = foo + 1;
    //~^ ERROR attempt to use a non-constant value in a constant

    println!("{}", y);
}
