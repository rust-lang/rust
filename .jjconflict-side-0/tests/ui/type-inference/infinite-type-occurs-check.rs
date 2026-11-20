//! Test that occurs check prevents infinite types during type inference.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/768>.

fn main() {
    let f;
    let g;

    g = f;
    //~^ ERROR overflow assigning `Box<_>` to `_`
    f = Box::new(g);
}
