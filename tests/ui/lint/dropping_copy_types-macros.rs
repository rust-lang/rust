//@ check-fail
//@ run-rustfix

#![deny(dropping_copy_types)]

use std::fmt::Write;

fn main() {
    let mut msg = String::new();
    drop(writeln!(&mut msg, "test"));
    //~^ ERROR calls to `std::mem::drop`

    drop(format_args!("a"));
    //~^ ERROR calls to `std::mem::drop`
}
