//@ run-pass
#![allow(non_upper_case_globals)]


pub fn main() {
    static _x: isize = 1<<2;
}
