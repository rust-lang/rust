// Testing gating of `#[stable]` in "weird" places.
//
// This file sits on its own because these signal errors, making
// this test incompatible with the "warnings only" nature of
// issue-43106-gating-of-builtin-attrs.rs

#![stable()]
//~^ ERROR stability attributes may not be used outside of the standard library
//~| ERROR missing 'since'
//~| ERROR missing 'feature'

#[stable()]
//~^ ERROR stability attributes may not be used outside of the standard library
//~| ERROR missing 'since'
//~| ERROR missing 'feature'
mod stable {
    mod inner {
        #![stable()]
        //~^ ERROR stability attributes may not be used outside of the standard library
        //~| ERROR missing 'since'
        //~| ERROR missing 'feature'
    }

    #[stable()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'since'
    //~| ERROR missing 'feature'
    fn f() {}

    #[stable()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'since'
    //~| ERROR missing 'feature'
    struct S;

    #[stable()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'since'
    //~| ERROR missing 'feature'
    type T = S;

    #[stable()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'since'
    //~| ERROR missing 'feature'
    impl S {}
}

fn main() {}
