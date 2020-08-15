// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct Foo<const N: [u8; {
//~^ ERROR cycle detected
//~| ERROR cycle detected
//[min]~| ERROR using `[u8; _]` as const generic
    struct Foo<const N: usize>;

    impl<const N: usize> Foo<N> {
        fn value() -> usize {
            N
        }
    }

    Foo::<17>::value()
}]>;

fn main() {}
