#![feature(exhaustive_patterns)]
#![feature(never_type, never_type_fallback)]
#![allow(unreachable_code)]
#![deny(unreachable_patterns)]

enum Void {}

impl Iterator for Void {
    type Item = Void;

    fn next(&mut self) -> Option<Void> {
        None
    }
}

fn main() {
    for _ in unimplemented!() as Void {}
    //~^ ERROR unreachable pattern
}
