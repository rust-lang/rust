// run-rustfix
#![allow(unused_must_use)]
#![warn(clippy::create_dir)]

fn create_dir() {}

fn main() {
    // Should be warned
    std::fs::create_dir("foo");
    std::fs::create_dir("bar").unwrap();

    // Shouldn't be warned
    create_dir();
    std::fs::create_dir_all("foobar");
}
