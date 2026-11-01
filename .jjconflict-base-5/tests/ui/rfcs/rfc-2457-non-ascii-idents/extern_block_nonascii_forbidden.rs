#![feature(extern_types)]

extern "C" {
    type 一; //~ ERROR items in `extern` blocks cannot use non-ascii identifiers
    fn 二(); //~ ERROR items in `extern` blocks cannot use non-ascii identifiers
    static 三: usize; //~ ERROR items in `extern` blocks cannot use non-ascii identifiers
}

fn main() {}
