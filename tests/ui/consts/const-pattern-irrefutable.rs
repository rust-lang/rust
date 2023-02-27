mod foo {
    pub const b: u8 = 2;
    pub const d: u8 = 2;
}

use foo::b as c;
use foo::d;

const a: u8 = 2;

fn main() {
    let a = 4;
    //~^ ERROR refutable pattern in local binding
    //~| patterns `0_u8..=1_u8` and `3_u8..=u8::MAX` not covered
    //~| HELP you might want to use `if let` to ignore the variants that aren't matched
    let c = 4;
    //~^ ERROR refutable pattern in local binding
    //~| patterns `0_u8..=1_u8` and `3_u8..=u8::MAX` not covered
    //~| HELP you might want to use `if let` to ignore the variants that aren't matched
    let d = 4;
    //~^ ERROR refutable pattern in local binding
    //~| patterns `0_u8..=1_u8` and `3_u8..=u8::MAX` not covered
    //~| HELP you might want to use `if let` to ignore the variants that aren't matched
    fn f() {} // Check that the `NOTE`s still work with an item here (cf. issue #35115).
}
