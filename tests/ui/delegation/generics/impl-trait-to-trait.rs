#![feature(fn_delegation)]
#![allow(incomplete_features)]


mod bounds {
    trait Trait0 {}

    trait Trait1<T> {
        fn foo<U>(&self)
        where
            T: Trait0,
            U: Trait0,
            Self: Trait0,
            //~^ ERROR the trait bound `bounds::S: Trait0` is not satisfied
        {
        }
    }

    struct F;
    impl<T> Trait1<T> for F {}

    struct S(F);

    impl<T> Trait1<T> for S {
        reuse Trait1::<T>::foo { &self.0 }
        //~^ ERROR the trait bound `bounds::F: Trait0` is not satisfied
    }
}

mod constants {
    trait Trait<const N1: i32> {
        fn foo<const N2: i32>(&self) {}
    }

    struct F;
    impl<const N: i32> Trait<N> for F {}
    struct S(F);

    impl<const N: i32> Trait<N> for S {
        // FIXME(fn_delegation): this is supposed to work eventually
        reuse Trait::foo { &self.0 }
        //~^ ERROR type annotations needed
    }
}

mod non_trait_method_path_resolution {
    trait Trait {
        fn foo(&self) {}
    }

    struct F;
    struct S<T>(F, T);
    impl Trait for F {}

    fn to_reuse(_: &F) {}

    impl<T> Trait for S<T> {
        reuse to_reuse as foo { &self.0 }
    }
}

fn main() {}
