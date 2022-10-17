// run-fail
// check-run-results

#![allow(unused_imports)]
#![feature(core_panic)]
#![feature(error_in_core)]

extern crate core;

use core::panicking::panic_source;
use core::error;

#[derive (Debug)]
struct MyErr;

use std::fmt;
impl fmt::Display for MyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "my source error message")
    }
}

impl error::Error for MyErr {

}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "full");
    let source = MyErr;
    //FIXME make the function do the Some wrapping for us
    panic_source(format_args!("here's my panic error message"), Some(&source));

}
