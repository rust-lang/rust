#![allow(unused_variables)]
use std::fs::File;

fn main() {
    if Err(err) = File::open("hello.txt") {
        //~^ ERROR: cannot find value `err`
        //~| ERROR: mismatched types
    }
}
