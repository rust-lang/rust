// Testing gating of `#[unstable]` in "weird" places.
//
// This file sits on its own because these signal errors, making
// this test incompatible with the "warnings only" nature of
// issue-43106-gating-of-builtin-attrs.rs

#![unstable()]
//~^ ERROR stability attributes may not be used outside of the standard library
//~| ERROR missing 'issue'
//~| ERROR missing 'feature'

#[unstable()]
//~^ ERROR stability attributes may not be used outside of the standard library
//~| ERROR missing 'issue'
//~| ERROR missing 'feature'
mod unstable {
    mod inner {
        #![unstable()]
        //~^ ERROR stability attributes may not be used outside of the standard library
        //~| ERROR missing 'issue'
        //~| ERROR missing 'feature'
    }

    #[unstable()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'issue'
    //~| ERROR missing 'feature'
    fn f() {}

    #[unstable()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'issue'
    //~| ERROR missing 'feature'
    struct S;

    #[unstable()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'issue'
    //~| ERROR missing 'feature'
    type T = S;

    #[unstable()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'issue'
    //~| ERROR missing 'feature'
    impl S {}
}

fn main() {}
