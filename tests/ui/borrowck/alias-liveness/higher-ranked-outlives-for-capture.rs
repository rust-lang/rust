//@ known-bug: #42940

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

trait Outlives<'a>: 'a {}
impl<'a, T: 'a> Outlives<'a> for T {}

// Test that we treat `for<'a> Opaque: 'a` as `Opaque: 'static`
fn test<'o>(v: &'o Vec<i32>) -> impl Captures<'o> + for<'a> Outlives<'a> {}

fn statik() -> impl Sized {
    test(&vec![])
}

fn main() {}
