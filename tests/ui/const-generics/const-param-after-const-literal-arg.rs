//@ check-pass

struct Foo<const A: usize, const B: usize>;

impl<const A: usize> Foo<1, A> {} // ok

fn main() {}
