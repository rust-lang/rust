//@ revisions: full min

#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

struct Foo<const N: [u8; {
    struct Foo<const N: usize>;

    impl<const N: usize> Foo<N> {
        fn value() -> usize {
            N
        }
    }

    Foo::<17>::value()
    //~^ ERROR cannot call non-const associated function
}]>;
//[min]~^^^^^^^^^^^^ ERROR `[u8; {

// N.B. it is important that the comment above is not inside the array length,
//      otherwise it may check for itself, instead of the actual error

fn main() {}
