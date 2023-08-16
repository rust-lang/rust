// run-fail
//@error-in-other-file:called `Result::unwrap()` on an `Err` value
//@ignore-target-emscripten no processes

use std::result::Result::Err;

fn main() {
    println!("{}", Err::<isize, String>("kitty".to_string()).unwrap());
}
