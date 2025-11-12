//@ check-pass
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

struct S<const M: usize>();

impl<const M: usize> S<M> {
    const N: usize = M;

    fn f() {
        let arr = [0; Self::N + 1];
    }
}

fn main() {
    S::<3>::f();
}
