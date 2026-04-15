//@ compile-flags: -Z deduplicate-diagnostics=yes

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod test_1 {
    trait Trait<T> {
        fn foo(&self, x: T) -> S { x }
        //~^ ERROR: missing generics for struct `test_1::S`
    }
    struct F;

    struct S<T>(F, T);

    impl<T, U> Trait<T> for S<U> {
        reuse to_reuse::foo { &self.0 }
        //~^ ERROR: cannot find module or crate `to_reuse` in this scope
    }
}

mod test_2 {
    trait Trait {
        fn foo() -> Self::Assoc;
        //~^ ERROR: associated type `Assoc` not found for `Self`
        //~| ERROR: this function takes 0 arguments but 1 argument was supplied
        fn bar(&self) -> u8;
    }

    impl Trait for u8 {
    //~^ ERROR: not all trait items implemented, missing: `foo`
        fn bar(&self) -> u8 { 1 }
    }

    struct S(u8);

    impl Trait for S {
        reuse Trait::* { &self.0 }
        fn bar(&self) -> u8 { 2 }
    }
}

mod test_3 {
    trait Trait {
        fn foo(&self) -> Self::Assoc<3> { //~ ERROR: associated type `Assoc` not found for `Self`
        //~^ ERROR: no method named `foo` found for reference `&()` in the current scope
            [(); 3]
        }
    }

    impl () { //~ ERROR: cannot define inherent `impl` for primitive types
        reuse Trait::*;
    }
}

mod test_4 {
    trait Trait<T> {
        fn foo<U>(&self, _: dyn Trait) {}
        //~^ ERROR: missing generics for trait `test_4::Trait`
    }

    reuse Trait::<_>::foo::<i32> as x;
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
}

fn main() {}
