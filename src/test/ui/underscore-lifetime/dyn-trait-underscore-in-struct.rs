// Check that the `'_` in `dyn Trait + '_` acts like ordinary elision,
// and not like an object lifetime default.
//
// cc #48468

use std::fmt::Debug;

struct Foo {
    x: Box<dyn Debug + '_>, //~ ERROR missing lifetime specifier
    //~^ ERROR E0228
}

fn main() { }
