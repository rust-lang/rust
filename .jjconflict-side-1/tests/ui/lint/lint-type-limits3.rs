#![allow(dead_code)]
#![warn(overflowing_literals)]

//@ compile-flags: -D unused-comparisons
fn main() { }

fn qux() {
    let mut i = 1i8;
    while 200 != i { //~ ERROR comparison is useless due to type limits
                     //~| WARN literal out of range for `i8`
        i += 1;
    }
}
