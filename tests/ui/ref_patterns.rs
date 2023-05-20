#![allow(unused)]
#![warn(clippy::ref_patterns)]

fn use_in_pattern() {
    let opt = Some(5);
    match opt {
        None => {},
        Some(ref opt) => {},
    }
}

fn use_in_binding() {
    let x = 5;
    let ref y = x;
}

fn use_in_parameter(ref x: i32) {}

fn main() {}
