#![allow(unused)]
#![warn(clippy::manual_find)]
//@no-rustfix
fn vec_string(strings: Vec<String>) -> Option<String> {
    for s in strings {
        //~^ ERROR: manual implementation of `Iterator::find`
        //~| NOTE: you may need to dereference some variables
        if s == String::new() {
            return Some(s);
        }
    }
    None
}

fn tuple(arr: Vec<(String, i32)>) -> Option<String> {
    for (s, _) in arr {
        //~^ ERROR: manual implementation of `Iterator::find`
        //~| NOTE: you may need to dereference some variables
        if s == String::new() {
            return Some(s);
        }
    }
    None
}

fn main() {}
