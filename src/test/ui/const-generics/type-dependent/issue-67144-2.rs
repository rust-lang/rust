// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct A<const N: usize>;

struct X;

impl X {
    fn inner<const N: usize>() -> A<N> {
        outer::<N>()
    }
}

fn outer<const N: usize>() -> A<N> {
    A
}

fn main() {
    let i: A<3usize> = outer::<3usize>();
    let o: A<3usize> = X::inner::<3usize>();
}
