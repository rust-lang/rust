// Testing gating of `#[stable]` in "weird" places.
//
// This file sits on its own because these signal errors, making
// this test incompatible with the "warnings only" nature of
// issue-43106-gating-of-builtin-attrs.rs

#![stable                     = "1300"]
//~^ ERROR stability attributes may not be used outside of the standard library

#[stable = "1300"]
//~^ ERROR stability attributes may not be used outside of the standard library
mod stable {
    mod inner { #![stable="1300"] }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[stable = "1300"] fn f() { }
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[stable = "1300"] struct S;
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[stable = "1300"] type T = S;
    //~^ ERROR stability attributes may not be used outside of the standard library

    #[stable = "1300"] impl S { }
    //~^ ERROR stability attributes may not be used outside of the standard library
}

fn main() {}
