// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct Foo<const N: [u8; { //[min]~ ERROR `[u8; _]` is forbidden
    struct Foo<const N: usize>;

    impl<const N: usize> Foo<N> {
        fn value() -> usize {
            N
        }
    }

    Foo::<17>::value()
    //~^ ERROR calls in constants are limited to constant functions
}]>;

fn main() {}
