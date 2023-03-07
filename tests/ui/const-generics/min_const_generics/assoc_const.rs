// check-pass
struct Foo<const N: usize>;

impl<const N: usize> Foo<N> {
    const VALUE: usize = N * 2;
}

trait Bar {
    const ASSOC: usize;
}

impl<const N: usize> Bar for Foo<N> {
    const ASSOC: usize = N * 3;
}

fn main() {}
