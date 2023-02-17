// revisions: full min

#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

struct Foo<const N: [u8; { //[min]~ ERROR `[u8; _]` is forbidden
    struct Foo<const N: usize>;

    impl<const N: usize> Foo<N> {
        fn value() -> usize {
            N
        }
    }

    Foo::<17>::value()
    //~^ ERROR cannot call non-const fn
}]>;

fn main() {}
