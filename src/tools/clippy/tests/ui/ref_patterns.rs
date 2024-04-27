#![allow(unused)]
#![warn(clippy::ref_patterns)]

fn use_in_pattern() {
    let opt = Some(5);
    match opt {
        None => {},
        Some(ref opt) => {},
        //~^ ERROR: usage of ref pattern
    }
}

fn use_in_binding() {
    let x = 5;
    let ref y = x;
    //~^ ERROR: usage of ref pattern
}

fn use_in_parameter(ref x: i32) {}
//~^ ERROR: usage of ref pattern

fn main() {}
