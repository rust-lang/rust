#![feature(fn_delegation)]

mod original_ice {
    struct S<const N: usize>;

    trait Trait<T, const B: bool> {
        fn fun();
    }

    impl<const N: usize> S<N> {
        reuse Trait::<S<N>, N>::fun;
        //~^ ERROR: the constant `N` is not of type `bool`
    }
}

mod with_child_constants {
    struct S<const N: usize>;

    trait Trait<T, const B: bool> {
        fn fun<const C: char>();
    }

    impl<const N: usize> S<N> {
        reuse Trait::<S<N>, N>::fun::<N>;
        //~^ ERROR: the constant `N` is not of type `bool`
        //~| ERROR: the constant `N` is not of type `char`
    }
}

fn main() {}
