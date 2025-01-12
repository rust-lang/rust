#![feature(never_patterns)]
#![allow(incomplete_features)]

fn main() {
    match () {
        (!|
        //~^ ERROR: mismatched types
        !) if true => {} //~ ERROR a never pattern is always unreachable
        //~^ ERROR: mismatched types
        (!|!) if true => {} //~ ERROR a never pattern is always unreachable
    }
}
