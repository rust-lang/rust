#![allow(stable_features)]
#![feature(no_core, core)]
#![no_core]

extern crate std;
extern crate core;

use std::option::Option::Some;

fn main() {
    let a = Some("foo");
    a.unwrap();
}
