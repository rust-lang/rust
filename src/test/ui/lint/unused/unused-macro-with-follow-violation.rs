#![allow(unused_macros)]

macro_rules! test {
    ($e:expr +) => () //~ ERROR not allowed for `expr` fragments
}

fn main() { }
