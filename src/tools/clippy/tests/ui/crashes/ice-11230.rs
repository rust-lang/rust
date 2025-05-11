// Test for https://github.com/rust-lang/rust-clippy/issues/11230
#![warn(clippy::explicit_iter_loop)]
#![warn(clippy::needless_collect)]

// explicit_iter_loop
fn main() {
    const A: &[for<'a> fn(&'a ())] = &[];
    for v in A.iter() {}
    //~^ explicit_iter_loop
}

// needless_collect
trait Helper<'a>: Iterator<Item = fn()> {}

// Should not be linted because we have no idea whether the iterator has side effects
fn x(w: &mut dyn for<'a> Helper<'a>) {
    w.collect::<Vec<_>>().is_empty();
}
