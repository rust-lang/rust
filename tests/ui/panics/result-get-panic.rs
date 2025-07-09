//@ run-fail
//@ check-run-results
//@ needs-subprocess

use std::result::Result::Err;

fn main() {
    println!("{}", Err::<isize, String>("kitty".to_string()).unwrap());
}
