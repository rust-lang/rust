#![feature(plugin)]
#![plugin(clippy)]

pub fn foo(bar: *const u8) {
    println!("{:#p}", bar);
}

fn main() {}
