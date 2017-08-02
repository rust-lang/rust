#![feature(plugin)]
#![plugin(clippy)]
#![allow(blacklisted_name)]

pub fn foo(bar: *const u8) {
    println!("{:#p}", bar);
}

fn main() {}
