//@ check-pass

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

trait Outlives<'a>: 'a {}
impl<'a, T: 'a> Outlives<'a> for T {}

// Test that we treat `for<'a> Opaque: 'a` as `Opaque: 'static`
fn test<'o>(v: &'o Vec<i32>) -> impl Captures<'o> + for<'a> Outlives<'a> {}

fn opaque_doesnt_use_temporary() {
    let a = test(&vec![]);
}

fn main() {}
