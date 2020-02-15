// Semantically, we do not allow e.g., `static X: u8 = 0;` as an associated item.

#![feature(specialization)]

fn main() {}

struct S;
impl S {
    static IA: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    static IB: u8;
    //~^ ERROR associated `static` items are not allowed
    default static IC: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    pub(crate) default static ID: u8;
    //~^ ERROR associated `static` items are not allowed
}

trait T {
    static TA: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    static TB: u8;
    //~^ ERROR associated `static` items are not allowed
    default static TC: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR `default` is only allowed on items in
    pub(crate) default static TD: u8;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR `default` is only allowed on items in
    //~| ERROR unnecessary visibility qualifier
}

impl T for S {
    static TA: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    static TB: u8;
    //~^ ERROR associated `static` items are not allowed
    default static TC: u8 = 0;
    //~^ ERROR associated `static` items are not allowed
    pub default static TD: u8;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR unnecessary visibility qualifier
}
