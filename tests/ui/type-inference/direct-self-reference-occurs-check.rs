//! Test that occurs check prevents direct self-reference in variable assignment.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/768>.

fn main() {
    let f;
    f = Box::new(f);
    //~^ ERROR overflow assigning `Box<_>` to `_`
}
