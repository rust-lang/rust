#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Bar<const M: usize> {}
impl<const N: usize> Bar<N> for A<{ 6 + 1 }> {}

struct A<const N: usize>
where
    A<N>: Bar<N>;

fn main() {
    let _ = A; //~ERROR the trait bound
}
