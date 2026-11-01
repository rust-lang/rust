//@ run-pass

#![feature(fn_delegation)]

trait Trait: Sized {
    fn by_value(self) -> i32 { 1 }
    fn by_mut_ref(&mut self) -> i32 { 2 }
    fn by_ref(&self) -> i32 { 3 }

    fn static_self() -> F { F }

    fn static_value(_: F) -> i32 { 1 }
    fn static_mut_ref(_: &mut F) -> i32 { 2 }
    fn static_ref(_: &F) -> i32 { 3 }
}

#[derive(Default, Eq, PartialEq, Debug)]
struct F;
impl Trait for F {}

struct S(F);

impl Trait for S {
    // Delegation's expression is removed from static functions.
    reuse <F as Trait>::* { self.0 }
}

fn main() {
    let mut s = S(F);
    assert_eq!(s.by_mut_ref(), 2);
    assert_eq!(s.by_ref(), 3);
    assert_eq!(s.by_value(), 1);

    assert_eq!(S::static_self(), F);

    assert_eq!(S::static_value(F), 1);
    assert_eq!(S::static_mut_ref(&mut F), 2);
    assert_eq!(S::static_ref(&F), 3);
}
