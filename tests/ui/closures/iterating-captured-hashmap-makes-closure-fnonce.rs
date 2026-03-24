//! Regression test for <https://github.com/rust-lang/rust/issues/42065>.
//!
//! Check that iterating a captured `HashMap` by value consumes the capture, so
//! the closure only implements `FnOnce` and cannot be called twice.

use std::collections::HashMap;

fn main() {
    let dict: HashMap<i32, i32> = HashMap::new();
    let debug_dump_dict = || {
        for (key, value) in dict {
            println!("{:?} - {:?}", key, value);
        }
    };
    debug_dump_dict();
    debug_dump_dict();
    //~^ ERROR use of moved value: `debug_dump_dict`
}
