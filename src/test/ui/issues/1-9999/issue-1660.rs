// run-pass
#![allow(non_upper_case_globals)]

// pretty-expanded FIXME #23616

pub fn main() {
    static _x: isize = 1<<2;
}
