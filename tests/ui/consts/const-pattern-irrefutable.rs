//@ dont-require-annotations: NOTE

mod foo {
    pub const b: u8 = 2;
    //~^ NOTE missing patterns are not covered because `b` is interpreted as a constant pattern, not a new variable
    pub const d: (u8, u8) = (2, 1);
    //~^ NOTE missing patterns are not covered because `d` is interpreted as a constant pattern, not a new variable
}

use foo::b as c;
use foo::d;

const a: u8 = 2;
//~^ NOTE missing patterns are not covered because `a` is interpreted as a constant pattern, not a new variable

#[derive(PartialEq)]
struct S {
    foo: u8,
}

const e: S = S {
    foo: 0,
};

fn main() {
    let a = 4;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE patterns `0_u8..=1_u8` and `3_u8..=u8::MAX` not covered
    //~| HELP introduce a variable instead
    let c = 4;
    //~^ ERROR refutable pattern in local binding
    //~| NOTE patterns `0_u8..=1_u8` and `3_u8..=u8::MAX` not covered
    //~| HELP introduce a variable instead
    let d = (4, 4);
    //~^ ERROR refutable pattern in local binding
    //~| NOTE patterns `(0_u8..=1_u8, _)` and `(3_u8..=u8::MAX, _)` not covered
    //~| HELP introduce a variable instead
    let e = S {
    //~^ ERROR refutable pattern in local binding
    //~| NOTE pattern `S { foo: 1_u8..=u8::MAX }` not covered
    //~| HELP introduce a variable instead
        foo: 1,
    };
    fn f() {} // Check that the `NOTE`s still work with an item here (cf. issue #35115).
}
