//@ run-pass

#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

trait Base {
    fn value(&self) -> u32 {
        0
    }
}
trait Sub<T>: Base {
    fn value(&self) -> T;
}
impl Base for () {}
impl Sub<u8> for () {
    fn value(&self) -> u8 {
        8
    }
}
impl Sub<u16> for () {
    fn value(&self) -> u16 {
        16
    }
}

// Supertrait shadowing keeps both Sub::value candidates. Same-trait collapse
// then recognizes that all candidates are of the same trait. The final impl is
// selected by type inference.
fn method<T: Sub<u8> + Sub<u16>>(x: &T) -> u16 {
    x.value()
}
fn path<T: Sub<u8> + Sub<u16>>(x: &T) -> u8 {
    T::value(x)
}

fn main() {
    assert_eq!(method(&()), 16);
    assert_eq!(path(&()), 8);
}
