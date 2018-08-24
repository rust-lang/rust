mod foo {
    pub const b: u8 = 2;
    pub const d: u8 = 2;
}

use foo::b as c;
use foo::d;

const a: u8 = 2;

fn main() {
    let a = 4; //~ ERROR refutable pattern in local binding: `_` not covered
    let c = 4; //~ ERROR refutable pattern in local binding: `_` not covered
    let d = 4; //~ ERROR refutable pattern in local binding: `_` not covered
    fn f() {} // Check that the `NOTE`s still work with an item here (c.f. issue #35115).
}
