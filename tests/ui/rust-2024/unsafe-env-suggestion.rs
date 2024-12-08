//@ run-rustfix

#![deny(deprecated_safe_2024)]

use std::env;

#[deny(unused_unsafe)]
fn main() {
    env::set_var("FOO", "BAR");
    //~^ ERROR call to deprecated safe function
    //~| WARN this is accepted in the current edition
    env::remove_var("FOO");
    //~^ ERROR call to deprecated safe function
    //~| WARN this is accepted in the current edition

    unsafe {
        env::set_var("FOO", "BAR");
        env::remove_var("FOO");
    }
}
