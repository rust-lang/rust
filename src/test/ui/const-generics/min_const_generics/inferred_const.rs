#![feature(min_const_generics)]
fn foo<const N: usize, const K: usize>(data: [u32; N]) -> [u32; K] {
    [0; K]
}
fn main() {
    let a = foo::<_, 2>([0, 1, 2]);
               //~^ ERROR type provided when a constant was expected
}
