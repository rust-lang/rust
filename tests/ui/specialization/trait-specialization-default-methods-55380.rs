// https://github.com/rust-lang/rust/issues/55380
//@ run-pass
#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

pub trait Foo {
    fn abc() -> u32;
    fn def() -> u32;
}

pub trait Marker {}

impl Marker for () {}

impl<T> Foo for T {
    default fn abc() -> u32 { 16 }
    default fn def() -> u32 { 42 }
}

impl<T: Marker> Foo for T {
    fn def() -> u32 {
        Self::abc()
    }
}

fn main() {
   assert_eq!(<()>::def(), 16);
   assert_eq!(<i32>::def(), 42);
}
