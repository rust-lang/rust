#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod first_example {
    mod to_reuse { pub fn foo() {} }
    struct S< S >;
    //~^ ERROR type parameter `S` is never used

    impl Item for S<S> {
        //~^ ERROR cannot find trait `Item` in this scope
        //~| ERROR missing generics for struct `S`
        reuse to_reuse::foo;
    }
}

mod second_example {
    trait Trait {
        reuse to_reuse::foo;
        //~^ ERROR function `foo` is private
    }
    mod to_reuse {
        fn foo() {}
    }
    impl Trait for S {
        //~^ ERROR cannot find type `S` in this scope
        reuse foo;
        //~^ ERROR cannot find function `foo` in this scope
    }
}

fn main() {}
