#![allow(unused)]

macro_rules! m {
    ($i) => {}; //~ ERROR missing fragment specifier
}

fn main() {}
