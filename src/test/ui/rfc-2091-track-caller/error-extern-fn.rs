#![feature(track_caller)]
#![allow(dead_code)]

extern "Rust" {
    #[track_caller] //~ ERROR: `#[track_caller]` is not supported on foreign functions
    fn bar();
}

fn main() {}
