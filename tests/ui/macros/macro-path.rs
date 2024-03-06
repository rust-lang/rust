//@ run-pass
#![allow(non_camel_case_types)]


mod m {
    pub type t = isize;
}

macro_rules! foo {
    ($p:path) => ({
        fn f() -> $p { 10 }
        f()
    })
}

pub fn main() {
    assert_eq!(foo!(m::t), 10);
}
