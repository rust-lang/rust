// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// Passing enums by value

// pretty-expanded FIXME #23616

pub enum void {}

mod bindgen {
    use super::void;

    extern "C" {
        pub fn printf(v: void);
    }
}

pub fn main() {}
