//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Foo<const B: bool>;

fn test<const N: usize>() -> Foo<{ N > 10 }> {
    Foo
}

fn main() {
    let _: Foo<true> = test::<12>();
    let _: Foo<false> = test::<9>();
}
