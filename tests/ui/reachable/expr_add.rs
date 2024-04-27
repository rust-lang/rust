#![feature(never_type)]
#![allow(unused_variables)]
#![deny(unreachable_code)]

use std::ops;

struct Foo;

impl ops::Add<!> for Foo {
    type Output = !;
    fn add(self, rhs: !) -> ! {
        unimplemented!()
    }
}

fn main() {
    let x = Foo + return; //~ ERROR unreachable
}
