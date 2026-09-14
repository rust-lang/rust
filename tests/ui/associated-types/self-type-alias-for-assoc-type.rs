//! Regression test for https://github.com/rust-lang/rust/issues/27281

//@ check-pass
pub trait Trait<'a> {
    type T;
    type U;
    fn foo(&self, s: &'a ()) -> &'a ();
}

impl<'a> Trait<'a> for () {
    type T = &'a ();
    type U = Self::T;

    fn foo(&self, s: &'a ()) -> &'a () {
        let t: Self::T = s; t
    }
}

fn main() {}
