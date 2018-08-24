#![feature(rustc_attrs)]
#![allow(dead_code)]

macro_rules! foo {
    ($x:tt) => (type Alias = $x<i32>;)
}

foo!(Box);

#[rustc_error]
fn main() {} //~ ERROR compilation successful
