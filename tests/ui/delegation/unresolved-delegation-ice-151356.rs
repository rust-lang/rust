#![allow(incomplete_features)]
#![feature(fn_delegation)]

extern "C" {
    fn a() {
        //~^ ERROR incorrect function inside `extern` block
        reuse foo {}
    }
}

pub fn main() {}
