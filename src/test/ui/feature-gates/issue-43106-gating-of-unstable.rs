// Testing gating of `#[unstable]` in "weird" places.
//
// This file sits on its own because these signal errors, making
// this test incompatible with the "warnings only" nature of
// issue-43106-gating-of-builtin-attrs.rs

#![unstable()]
//~^ ERROR stability attributes may not be used outside of the standard library

#[unstable()]
//~^ ERROR stability attributes may not be used outside of the standard library
mod unstable {
    mod inner { #![unstable()] }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[unstable()] fn f() { }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[unstable()] struct S;
    //~^ ERROR stability attributes may not be used outside of the standard library
    //~| ERROR stability attributes may not be used outside of the standard library

    #[unstable()] type T = S;
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[unstable()] impl S { }
    //~^ ERROR stability attributes may not be used outside of the standard library
}

fn main() {}
