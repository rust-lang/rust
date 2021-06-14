// check-pass
// run-rustfix
// compile-flags: -Z unstable-options --edition 2018

#![warn(reserved_prefix)]

macro_rules! m2 {
    ($a:tt $b:tt) => {};
}

macro_rules! m3 {
    ($a:tt $b:tt $c:tt) => {};
}

fn main() {
    m2!(z"hey");
    //~^ WARNING prefix `z` is unknown [reserved_prefix]
    //~| WARNING become a hard error
    m2!(prefix"hey");
    //~^ WARNING prefix `prefix` is unknown [reserved_prefix]
    //~| WARNING become a hard error
    m3!(hey#123);
    //~^ WARNING prefix `hey` is unknown [reserved_prefix]
    //~| WARNING become a hard error
    m3!(hey#hey);
    //~^ WARNING prefix `hey` is unknown [reserved_prefix]
    //~| WARNING become a hard error
}
