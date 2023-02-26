#![feature(no_core)]
#![feature(lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "add"]
trait Add<T> {
    const add: u32 = 1u32;
}

impl Add<u32> for u32 {}

fn main() {
    1u32 + 1u32;
    //~^ ERROR cannot add `u32` to `u32`
}
