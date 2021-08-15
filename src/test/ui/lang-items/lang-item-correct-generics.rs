// build-pass

#![feature(lang_items,no_core)]
#![no_core]
#![crate_type="lib"]

#[lang = "sized"]
trait MySized {}

#[lang = "copy"]
trait MyCopy {}

#[lang = "drop"]
trait MyDrop<T> {}

struct S;

impl<T> MyDrop<T> for S {}

#[lang = "i32"]
impl<'a> i32 {
    fn foo() {}
}

fn bar() {
    i32::foo();
    S;
}
