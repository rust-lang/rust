//! Regression test for issue https://github.com/rust-lang/rust/issues/4968
//@ dont-require-annotations: NOTE

const A: (isize, isize) = (4, 2);
fn main() {
    match 42 {
        A => (),
        //~^ ERROR mismatched types
        //~| NOTE expected type `{integer}`
        //~| NOTE found tuple `(isize, isize)`
        //~| NOTE expected integer, found `(isize, isize)`
    }
}
