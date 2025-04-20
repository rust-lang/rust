#![feature(never_patterns, never_type)]
#![allow(incomplete_features)]

enum Void {}
fn main() {}

macro_rules! never {
    () => { ! }
}

fn no_arms_or_guards(x: Void) {
    match x {
        never!() => {}
        //~^ ERROR a never pattern is always unreachable
    }
}
