//@ run-fail
//@ error-pattern:called `Result::unwrap()` on an `Err` value
//@ ignore-emscripten no processes

use std::result::Result::Err;

fn main() {
    println!("{}", Err::<isize, String>("kitty".to_string()).unwrap());
}
