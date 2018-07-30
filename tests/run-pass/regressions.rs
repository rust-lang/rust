#![feature(tool_lints)]

#![allow(clippy::blacklisted_name)]

pub fn foo(bar: *const u8) {
    println!("{:#p}", bar);
}

fn main() {}
