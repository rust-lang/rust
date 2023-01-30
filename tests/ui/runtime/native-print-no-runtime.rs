// run-pass

#![feature(start)]

#[start]
pub fn main(_: isize, _: *const *const u8) -> isize {
    println!("hello");
    0
}
