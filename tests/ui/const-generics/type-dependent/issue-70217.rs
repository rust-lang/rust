//@ check-pass

struct Struct<const N: usize>;

impl<const N: usize> Struct<N> {
    fn method<const M: usize>(&self) {}
}

fn test<const N: usize, const M: usize>(x: Struct<N>) {
    Struct::<N>::method::<M>(&x);
    x.method::<N>();
}

fn main() {}
