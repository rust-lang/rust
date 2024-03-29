#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod infer_types {
    trait Trait {}

    mod to_reuse {
        use super::*;

        pub fn foo<T, U>(x: T, y: U) -> (T, U) { (x, y) }
        pub fn foo2<T: Trait>(x: T) -> T { x }
    }

    fn check() {
        {
            reuse to_reuse::foo::<_>;
            //~^ ERROR function takes 2 generic arguments but 1 generic argument was supplied
            //~| ERROR function takes 2 generic arguments but 1 generic argument was supplied
        }
        {
            reuse to_reuse::foo2::<u32>;
            //~^ ERROR the trait bound `u32: infer_types::Trait` is not satisfied
        }
        {
            reuse to_reuse::foo2<u32>;
            //~^ ERROR expected one of `::`, `;`, `as`, or `{`, found `<`
        }
        {
            reuse to_reuse::foo2::<T>;
            //~^ ERROR cannot find type `T` in this scope
        }
    }
}

mod infer_late_bound_regions {
    mod to_reuse {
        pub fn foo<T>(x: &T) -> &T { x }
    }

    fn check() {
        let x = 1;
        {
            reuse to_reuse::foo::<'_>;
            //~^ ERROR cannot specify lifetime arguments explicitly if late bound lifetime parameters are present
            //~| ERROR cannot specify lifetime arguments explicitly if late bound lifetime parameters are present
            assert_eq!(*foo(&x), 1);
        }
    }
}

mod infer_early_bound_regions {
    mod to_reuse {
        pub fn foo<'a: 'a, T>(x: &'a T) -> &'a T { x }
    }

    fn check() {
        let x = 1;
        {
            reuse to_reuse::foo::<'static, _>;
            assert_eq!(*foo(&x), 1);
            //~^ ERROR `x` does not live long enough
        }
        {
            fn bar<'a: 'a>(_: &'a i32) {
                reuse to_reuse::foo::<'a, _>;
                //~^ ERROR can't use generic parameters from outer item
            }

            reuse to_reuse::foo::<'a, _>;
            //~^ ERROR use of undeclared lifetime name `'a`
            assert_eq!(*foo(&x), 1);
        }
    }
}

fn main() {}
