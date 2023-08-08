// Semantically, we do not allow e.g., `static X: u8 = 0;` as an associated item.

#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

fn main() {}

struct S;
impl S {
    static IA: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    static IB: u8;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR associated constant in `impl` without body
    default static IC: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR a static item cannot be `default`
    pub(crate) default static ID: u8;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR associated constant in `impl` without body
    //~| ERROR a static item cannot be `default`
}

trait T {
    static TA: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    static TB: u8;
    //~^ ERROR associated `static` items are not allowed
    default static TC: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR a static item cannot be `default`
    pub(crate) default static TD: u8;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR visibility qualifiers are not permitted here
    //~| ERROR a static item cannot be `default`
}

impl T for S {
    static TA: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    static TB: u8;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR associated constant in `impl` without body
    default static TC: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR a static item cannot be `default`
    pub default static TD: u8;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR associated constant in `impl` without body
    //~| ERROR visibility qualifiers are not permitted here
    //~| ERROR a static item cannot be `default`
}
