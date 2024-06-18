//@ run-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait : Sized {
    fn by_value(self) -> i32 { 1 }
    fn by_mut_ref(&mut self) -> i32 { 2 }
    fn by_ref(&self) -> i32 { 3 }
}

struct F;
impl Trait for F {}

struct S(F);

impl Trait for S {
    reuse Trait::{by_value, by_mut_ref, by_ref} { self.0 }
}

fn main() {
    let mut s = S(F);
    assert_eq!(s.by_ref(), 3);
    assert_eq!(s.by_mut_ref(), 2);
    assert_eq!(s.by_value(), 1);
}
