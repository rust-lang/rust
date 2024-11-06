mod foo {
    pub const b: u8 = 2;
    //~^ missing patterns are not covered because `c` is interpreted as a constant pattern, not a new variable
    pub const d: u8 = 2;
    //~^ missing patterns are not covered because `d` is interpreted as a constant pattern, not a new variable
}

use foo::b as c;
use foo::d;

const a: u8 = 2;
//~^ missing patterns are not covered because `a` is interpreted as a constant pattern, not a new variable

fn main() {
    let a = 4;
    //~^ ERROR refutable pattern in local binding
    //~| patterns `0_u8..=1_u8` and `3_u8..=u8::MAX` not covered
    //~| HELP introduce a variable instead
    let c = 4;
    //~^ ERROR refutable pattern in local binding
    //~| patterns `0_u8..=1_u8` and `3_u8..=u8::MAX` not covered
    //~| HELP introduce a variable instead
    let d = 4;
    //~^ ERROR refutable pattern in local binding
    //~| patterns `0_u8..=1_u8` and `3_u8..=u8::MAX` not covered
    //~| HELP introduce a variable instead
    fn f() {} // Check that the `NOTE`s still work with an item here (cf. issue #35115).
}
