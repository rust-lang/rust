// issue: <https://github.com/rust-lang/rust/issues/142473>
// Also related to #71054 / #120871: `Self::Assoc()` now resolves when Assoc is an
// associated type set to a unit struct.
//
//@ check-pass
#![allow(unused)]

struct T();

trait Trait {
    type Assoc;

    fn f();
}

impl Trait for () {
    type Assoc = T;

    fn f() {
        <Self>::Assoc();
    }
}

fn main() {}
