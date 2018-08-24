#![allow(unused_macros)]

macro_rules! assign {
    (($($a:tt)*) = ($($b:tt))*) => { //~ ERROR expected `*` or `+`
        $($a)* = $($b)*
    }
}

fn main() {}
