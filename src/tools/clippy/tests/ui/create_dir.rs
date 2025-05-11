#![allow(unused_must_use)]
#![warn(clippy::create_dir)]

use std::fs::create_dir_all;

fn create_dir() {}

fn main() {
    // Should be warned
    std::fs::create_dir("foo");
    //~^ create_dir
    std::fs::create_dir("bar").unwrap();
    //~^ create_dir

    // Shouldn't be warned
    create_dir();
    std::fs::create_dir_all("foobar");
}
