// Syntactically, we do allow e.g., `static X: u8 = 0;` as an associated item.

fn main() {}

#[cfg(false)]
impl S {
    static IA: u8 = 0; //~ ERROR associated `static` items are not allowed
    static IB: u8; //~ ERROR associated `static` items are not allowed
    default static IC: u8 = 0; //~ ERROR associated `static` items are not allowed
    //~^ ERROR a static item cannot be `default`
    pub(crate) default static ID: u8; //~ ERROR associated `static` items are not allowed
    //~^ ERROR a static item cannot be `default`
}

#[cfg(false)]
trait T {
    static TA: u8 = 0; //~ ERROR associated `static` items are not allowed
    static TB: u8; //~ ERROR associated `static` items are not allowed
    default static TC: u8 = 0; //~ ERROR associated `static` items are not allowed
    //~^ ERROR a static item cannot be `default`
    pub(crate) default static TD: u8; //~ ERROR associated `static` items are not allowed
    //~^ ERROR a static item cannot be `default`
}

#[cfg(false)]
impl T for S {
    static TA: u8 = 0; //~ ERROR associated `static` items are not allowed
    static TB: u8; //~ ERROR associated `static` items are not allowed
    default static TC: u8 = 0; //~ ERROR associated `static` items are not allowed
    //~^ ERROR a static item cannot be `default`
    pub default static TD: u8; //~ ERROR associated `static` items are not allowed
    //~^ ERROR a static item cannot be `default`
}
