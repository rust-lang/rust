#![feature(min_const_generics)]

type N = u32;
struct Foo<const M: usize>;
fn test<const N: usize>() -> Foo<N> { //~ ERROR type provided when
    Foo
}

fn main() {}
