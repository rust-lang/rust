//@ edition: 2021

// https://github.com/rust-lang/rust/pull/111761#issuecomment-1557777314
macro_rules! m {
    () => {
        extern crate core as std;
        //~^ ERROR macro-expanded `extern crate` items cannot shadow names passed with `--extern`
    }
}

m!();

use std::mem; //~ ERROR `std` is ambiguous
use ::std::mem as _; //~ ERROR `std` is ambiguous

fn main() {}
