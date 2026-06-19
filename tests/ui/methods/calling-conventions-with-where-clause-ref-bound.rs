//! Regression test for <https://github.com/rust-lang/rust/issues/20414>.
//!
//! Test both UFCS and dot syntax should work for trait methods when the
//! impl has a where-clause bound on a reference type.

//@ check-pass
#![allow(dead_code)]

trait Trait {
        fn method(self) -> isize;
}

struct Wrapper<T> {
        field: T
}

impl<'a, T> Trait for &'a Wrapper<T> where &'a T: Trait {
    fn method(self) -> isize {
        let r: &'a T = &self.field;
        Trait::method(r); // these should both work
        r.method()
    }
}

fn main() {}
