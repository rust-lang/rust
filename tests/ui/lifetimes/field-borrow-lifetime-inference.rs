//! Regression test for <https://github.com/rust-lang/rust/issues/28999>.
//! `this.v` was not constrained and inferred `'a`.
//@ check-pass

pub struct Xyz<'a, V> {
    pub v: (V, &'a u32),
}

pub fn eq<'a, 's, 't, V>(this: &'s Xyz<'a, V>, other: &'t Xyz<'a, V>) -> bool
        where V: PartialEq {
    this.v == other.v
}

fn main() {}
