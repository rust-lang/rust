// aux-build:xcrate.rs
// compile-flags:--extern xcrate

#![feature(extern_in_paths)]

use extern; //~ ERROR unresolved import `extern`
            //~^ NOTE no `extern` in the root
use extern::*; //~ ERROR cannot glob-import all possible crates

fn main() {
    let s = extern::xcrate; //~ ERROR expected value, found module `extern::xcrate`
                            //~^ NOTE not a value
}
