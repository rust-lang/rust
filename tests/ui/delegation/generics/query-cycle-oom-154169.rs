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

fn main() {}
