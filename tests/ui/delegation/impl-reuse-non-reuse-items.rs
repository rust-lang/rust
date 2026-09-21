#![feature(fn_delegation)]

mod non_delegatable_items {
    trait Trait {
        fn method(&self);
        const CONST: u8;
        type Type;
        #[allow(non_camel_case_types)]
        type method;
    }

    struct F;
    impl Trait for F {
        fn method(&self) {}
        const CONST: u8 = 0;
        type Type = u8;
        type method = u8;
    }

    struct S(F);

    reuse impl Trait for S { &self.0 }
    //~^ ERROR item `CONST` is an associated method, which doesn't match its trait `Trait`
    //~| ERROR item `Type` is an associated method, which doesn't match its trait `Trait`
    //~| ERROR duplicate definitions with name `method`
    //~| ERROR not all trait items implemented, missing: `CONST`, `Type`, `method`
    //~| ERROR expected function, found associated constant `Trait::CONST`
    //~| WARN trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| ERROR the trait `non_delegatable_items::Trait` is not dyn compatible
}

fn main() {}
