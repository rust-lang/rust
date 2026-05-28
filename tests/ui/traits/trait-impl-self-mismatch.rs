//@compile-flags: -Zvalidate-mir -Zinline-mir -Zinline-mir-threshold=300

//! Ensure that a trait method implemented with the wrong signature
//! correctly triggers a compile error and not an ICE.
//! Regression test for <https://github.com/rust-lang/rust/issues/133065>.

trait Bar {
    fn bar(&self) {}
}

impl<T> Bar for T {
    fn bar() { //~ ERROR method `bar` has a `&self` declaration in the trait, but not in the impl
        let _ = "Hello".bytes().nth(3);
    }
}

fn main() {
    ().bar();
}
