//@ check-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn foo() -> impl Clone { 0 }
}

reuse to_reuse::foo;

trait Trait {
    fn bar() -> impl Clone { 1 }
}

struct S;
impl Trait for S {}

impl S {
    reuse to_reuse::foo;
    reuse <S as Trait>::bar;
}

fn main() {
    foo().clone();
    <S>::bar().clone();
}
