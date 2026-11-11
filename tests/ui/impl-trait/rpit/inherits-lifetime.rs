//! Check that lifetimes are inherited in RPIT.
//! Previously, the hidden lifetime of T::Bar would be overlooked
//! and would instead end up as <T as Foo<'static>>::Bar.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/51525>.

//@ check-pass

trait Foo<'a> {
    type Bar;
}

impl<'a> Foo<'a> for u32 {
    type Bar = &'a ();
}

fn baz<'a, T>() -> impl IntoIterator<Item = T::Bar>
where
    T: Foo<'a>,
{
    None
}

fn main() {}
