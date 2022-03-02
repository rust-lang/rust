// Testing gating of `#[rustc_deprecated]` in "weird" places.
//
// This file sits on its own because these signal errors, making
// this test incompatible with the "warnings only" nature of
// issue-43106-gating-of-builtin-attrs.rs

#![rustc_deprecated()]
//~^ ERROR stability attributes may not be used outside of the standard library
//~| ERROR missing 'since' [E0542]

#[rustc_deprecated()]
//~^ ERROR stability attributes may not be used outside of the standard library
//~| ERROR missing 'since' [E0542]
mod rustc_deprecated {
    mod inner {
        #![rustc_deprecated()]
        //~^ ERROR stability attributes may not be used outside of the standard library
        //~| ERROR missing 'since' [E0542]
    }

    #[rustc_deprecated()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'since' [E0542]
    fn f() {}

    #[rustc_deprecated()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'since' [E0542]
    //~| ERROR missing 'since' [E0542]
    struct S;

    #[rustc_deprecated()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'since' [E0542]
    type T = S;

    #[rustc_deprecated()]
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR missing 'since' [E0542]
    impl S {}
}

fn main() {}
