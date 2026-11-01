//@ build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]

macro_rules! foo {
    ($x:tt) => (type Alias = $x<i32>;)
}

foo!(Box);


fn main() {}
