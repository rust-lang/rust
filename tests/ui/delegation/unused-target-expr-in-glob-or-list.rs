#![feature(fn_delegation)]

pub trait Trait: Sized {
    fn static_self() -> F { F }
    fn static_self2() -> F { F }

    fn static_value(_: F) -> i32 { 1 }
    fn static_mut_ref(_: &mut F) -> i32 { 2 }
    fn static_ref(_: &F) -> i32 { 3 }
}

#[derive(Default, Eq, PartialEq, Debug)]
pub struct F;
impl Trait for F {}

struct S(F);
impl Trait for S {
    reuse <F as Trait>::* { self.0 }
    //~^ ERROR: unused target expression is specified for glob or list delegation
}

struct S1(F);
impl S1 {
    reuse <F as Trait>::* { self.0 }
    //~^ ERROR: unused target expression is specified for glob or list delegation
}

struct S2(F);
impl Trait for S2 {
    reuse <F as Trait>::{static_self} { self.0 }
    //~^ ERROR: unused target expression is specified for glob or list delegation
}

struct S3(F);
impl Trait for S3 {
    reuse <F as Trait>::{static_self, static_value} { self.0 }
    //~^ ERROR: unused target expression is specified for glob or list delegation
}

struct S4(F);
impl Trait for S4 {
    reuse <F as Trait>::{static_self, static_value, static_mut_ref, static_ref} { self.0 }
    //~^ ERROR: unused target expression is specified for glob or list delegation
}

struct S5(F);
impl Trait for S5 {
    reuse <F as Trait>::{static_self, static_value, static_mut_ref, static_ref} { }
    //~^ ERROR: unused target expression is specified for glob or list delegation
}

struct S6(F);
impl Trait for S6 {
    // Error about unused target expression is not emitted when error delegation is generated.
    reuse UnresolvedTrait::* { self.0 }
    //~^ ERROR: cannot find type `UnresolvedTrait` in this scope
}

struct S7(F);
impl Trait for S7 {
    reuse <F as Trait>::*;
}

struct S8(F);
impl Trait for S8 {
    reuse <F as Trait>::{static_self, static_self2} { { F } }
    //~^ ERROR: unused target expression is specified for glob or list delegation
}

struct S9;
impl S9 {
    reuse <F as Trait>::{static_self, static_self2} { { F } }
    //~^ ERROR: unused target expression is specified for glob or list delegation

    reuse <F as Trait>::{static_value, static_mut_ref, static_ref} { }
    //~^ ERROR: unused target expression is specified for glob or list delegation
}

trait Trait2 {
    reuse <F as Trait>::{static_self, static_self2} { { F } }
    //~^ ERROR: unused target expression is specified for glob or list delegation

    reuse <F as Trait>::{static_value, static_mut_ref} { }
    //~^ ERROR: unused target expression is specified for glob or list delegation

    reuse <F as Trait>::{static_ref};
}

mod free_to_trait1 {
    use super::{F, Trait};

    reuse <F as Trait>::{static_self, static_self2} { { F } }
    //~^ ERROR: unused target expression is specified for glob or list delegation

    reuse <F as Trait>::{static_value, static_mut_ref} { }
    //~^ ERROR: unused target expression is specified for glob or list delegation

    reuse <F as Trait>::{static_ref};
}

mod macros {
    use super::*;

    macro_rules! delegation {
        () => {
            impl Trait for S {
                reuse <F as Trait>::static_self { self.0 }
                //~^ ERROR: delegation's target expression is specified for function with no params
                //~| ERROR: mismatched types
                //~| ERROR: this function takes 0 arguments but 1 argument was supplied
                //~| ERROR: method `static_self` has an incompatible type for trait
                reuse <F as Trait>::static_value { self.0 }
                //~^ ERROR: no field `0` on type `F`
                reuse <F as Trait>::static_mut_ref { self.0 }
                //~^ ERROR: no field `0` on type `&mut F`
                reuse <F as Trait>::static_ref { self.0 }
                //~^ ERROR: no field `0` on type `&F`
            }
        };
    }

    struct S(F);
    delegation!();

    macro_rules! delegation2 {
        () => {
            reuse <F as Trait>::static_self { self.0 }
            //~^ ERROR: delegation's target expression is specified for function with no params
            //~| ERROR: mismatched types
            //~| ERROR: this function takes 0 arguments but 1 argument was supplied
            //~| ERROR: method `static_self` has an incompatible type for trait
            reuse <F as Trait>::static_value { self.0 }
            //~^ ERROR: no field `0` on type `F`
            reuse <F as Trait>::static_mut_ref { self.0 }
            //~^ ERROR: no field `0` on type `&mut F`
            reuse <F as Trait>::static_ref { self.0 }
            //~^ ERROR: no field `0` on type `&F`
        };
    }

    struct S1(F);
    impl Trait for S1 {
        delegation2!();
    }
}

mod free_list {
    mod to_reuse {
        pub fn value() -> i32 { 1 }
        pub fn mut_ref() -> i32 { 2 }
        pub fn r#ref() -> i32 { 3 }
    }

    reuse to_reuse::{value, mut_ref, r#ref} { () }
    //~^ ERROR: this function takes 0 arguments but 1 argument was supplied
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
    //~| ERROR: this function takes 0 arguments but 1 argument was supplied
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: delegation's target expression is specified for function with no params
    //~| ERROR: delegation's target expression is specified for function with no params
    //~| ERROR: delegation's target expression is specified for function with no params

    reuse to_reuse::{value as value2} { }
    //~^ ERROR: this function takes 0 arguments but 1 argument was supplied
    //~| ERROR: mismatched types
    //~| ERROR: delegation's target expression is specified for function with no params
}

mod to_free {
    trait Trait {
        fn value() -> i32 { 1 }
        fn mut_ref() -> i32 { 2 }
        fn r#ref() -> i32 { 3 }
    }

    mod to_reuse {
        pub fn value() -> i32 { 1 }
        pub fn mut_ref() -> i32 { 2 }
        pub fn r#ref() -> i32 { 3 }
    }

    struct F;
    impl Trait for F {}

    struct S(F);
    impl Trait for S {
        reuse to_reuse::{value, mut_ref, r#ref} { () }
        //~^ ERROR: unused target expression is specified for glob or list delegation
    }

    struct S2;
    impl S2 {
        reuse to_reuse::{value, mut_ref, r#ref} { 1 + 1 }
        //~^ ERROR: this function takes 0 arguments but 1 argument was supplied
        //~| ERROR: this function takes 0 arguments but 1 argument was supplied
        //~| ERROR: this function takes 0 arguments but 1 argument was supplied
        //~| ERROR: mismatched types
        //~| ERROR: mismatched types
        //~| ERROR: mismatched types
        //~| ERROR: delegation's target expression is specified for function with no params
        //~| ERROR: delegation's target expression is specified for function with no params
        //~| ERROR: delegation's target expression is specified for function with no params
    }
}

fn main() {}
