#![feature(const_generics)]
#![allow(incomplete_features)]

struct Foo<const N: [u8; {
//~^ ERROR cycle detected
//~| ERROR cycle detected
    struct Foo<const N: usize>;

    impl<const N: usize> Foo<N> {
        fn value() -> usize {
            N
        }
    }

    Foo::<17>::value()
}]>;

fn main() {}
