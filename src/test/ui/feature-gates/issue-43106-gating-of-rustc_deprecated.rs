// Testing gating of `#[rustc_deprecated]` in "weird" places.
//
// This file sits on its own because these signal errors, making
// this test incompatible with the "warnings only" nature of
// issue-43106-gating-of-builtin-attrs.rs

#![rustc_deprecated()]
//~^ ERROR stability attributes may not be used outside of the standard library

#[rustc_deprecated()]
//~^ ERROR stability attributes may not be used outside of the standard library
mod rustc_deprecated {
    mod inner { #![rustc_deprecated()] }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[rustc_deprecated()] fn f() { }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[rustc_deprecated()] struct S;
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR stability attributes may not be used outside of the standard library

    #[rustc_deprecated()] type T = S;
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[rustc_deprecated()] impl S { }
    //~^ ERROR stability attributes may not be used outside of the standard library
}

fn main() {}
