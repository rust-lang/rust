#![deny(unused_qualifications)]

use self::A::B;

#[derive(PartialEq)]
pub enum A {
    B,
}

fn main() {}
