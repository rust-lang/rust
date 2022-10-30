use std::env::{remove_var, set_var};

fn main() {
    set_var("FOO", "bar");
    remove_var("FOO");

    unsafe {
        set_var("FOO", "bar"); //~ ERROR mutation of environment inside unsafe context
        remove_var("FOO"); //~ ERROR mutation of environment inside unsafe context
    }
}
