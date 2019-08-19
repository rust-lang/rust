// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(unused_variables)]
trait Trait<'a> {
    type A;
    type B;
}

fn foo<'a, T: Trait<'a>>(value: T::A) {
    let new: T::B = unsafe { std::mem::transmute_copy(&value) };
}

fn main() { }
