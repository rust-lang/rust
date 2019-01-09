// aux-build:xcrate.rs
// compile-flags:--extern xcrate
// edition:2018

use crate; //~ ERROR crate root imports need to be explicitly named: `use crate as name;`
use *; //~ ERROR cannot glob-import all possible crates

fn main() {
    let s = ::xcrate; //~ ERROR expected value, found module `xcrate`
                      //~^ NOTE not a value
}
