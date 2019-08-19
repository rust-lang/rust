// build-pass (FIXME(62277): could be check-pass?)
#![allow(unused_imports)]
#![deny(unused_qualifications)]

use self::A::B;

#[derive(PartialEq)]
pub enum A {
    B,
}

fn main() {}
