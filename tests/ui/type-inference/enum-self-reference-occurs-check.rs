//! Test that occurs check prevents infinite types with enum self-references.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/778>.

enum Clam<T> {
    A(T),
}

fn main() {
    let c;
    c = Clam::A(c);
    //~^ ERROR overflow assigning `Clam<_>` to `_`
    match c {
        Clam::A::<isize>(_) => {}
    }
}
