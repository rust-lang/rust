//@ run-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    use crate::S;

    pub fn foo<'a>(#[cfg(false)] a: u8, _b: &'a S) -> u32 {
        1
    }
}

reuse to_reuse::foo;

trait Trait {
    fn foo(&self) -> u32 { 0 }
    fn bar(self: Box<Self>) -> u32 { 2 }
    fn baz(a: (i32, i32)) -> i32 { a.0 + a.1 }
}

struct F;
impl Trait for F {}

struct S(F);

impl Trait for S {
    reuse to_reuse::foo { self }
    reuse Trait::bar { Box::new(self.0) }
    reuse <F as Trait>::baz;
}

fn main() {
    let s = S(F);
    assert_eq!(1, foo(&s));
    assert_eq!(1, s.foo());
    assert_eq!(2, Box::new(s).bar());
    assert_eq!(4, S::baz((2, 2)));
}
