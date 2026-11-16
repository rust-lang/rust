//@ check-pass

// This is like `higher-ranked-outlives-for-capture`, but represents a case that
// we can accept. Specifically, although `'o` is `use`d here, it is not named in
// any trait or projection bounds.

trait Outlives<'a>: 'a {}
impl<'a, T: 'a> Outlives<'a> for T {}

fn test<'o>(v: &'o Vec<i32>) -> impl use<'o> + for<'a> Outlives<'a> {}

fn opaque_doesnt_use_temporary() {
    let a = test(&vec![]);
}

fn main() {}
