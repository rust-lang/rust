//@ check-pass
#![allow(unused_imports)]
#![deny(unused_qualifications)]

use self::A::B;

#[derive(PartialEq)]
pub enum A {
    B,
}

fn main() {}
