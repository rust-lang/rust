//! regression test for <https://github.com/rust-lang/rust/issues/11384>
//@ check-pass

trait Common {
    fn dummy(&self) {}
}

impl<'t, T> Common for (T, &'t T) {}

impl<'t, T> Common for (&'t T, T) {}

fn main() {}
