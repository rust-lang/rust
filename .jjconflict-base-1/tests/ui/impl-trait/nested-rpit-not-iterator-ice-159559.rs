//! Regression test for <https://github.com/rust-lang/rust/issues/159559>.
//! Reporting the `E0277` for the unsatisfied `IntoIterator` bound on the
//! nested opaque type used to ICE ("Normalizing ... without wrapping in a
//! `Binder`") in the RPIT method-chain suggestion when the return type
//! captures a lifetime.

trait Cap<'a> {}

impl<T> Cap<'_> for T {}

fn fail_late_bound<'a>(
    a: &u8,
    _: &'a u8,
) -> impl IntoIterator<Item = impl Cap<'a> + IntoIterator<Item = impl Cap<'a>>> {
    //~^ ERROR `&u8` is not an iterator
    [a]
}

fn main() {}
