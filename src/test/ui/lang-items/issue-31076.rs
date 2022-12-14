#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized {}

#[lang="add"]
trait Add<T> {}

impl Add<i32> for i32 {}

fn main() {
    let x = 5 + 6;
    //~^ ERROR cannot add `i32` to `{integer}`
    let y = 5i32 + 6i32;
    //~^ ERROR cannot add `i32` to `i32`
}
