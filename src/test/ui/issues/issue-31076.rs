#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized {}

#[lang="add"]
trait Add<T> {}

impl Add<i32> for i32 {}

fn main() {
    let x = 5 + 6;
    //~^ ERROR binary operation `+` cannot be applied to type `{integer}`
    let y = 5i32 + 6i32;
    //~^ ERROR binary operation `+` cannot be applied to type `i32`
}
