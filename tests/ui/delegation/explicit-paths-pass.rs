//@ run-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn bar(&self, x: i32) -> i32 { x }
    fn description(&self) -> &str {
        "hello world!"
    }
    fn static_method(x: i32) -> i32 { x }
    fn static_method2(x: i32, y: i32) -> i32 { x + y }
}

struct F;
impl Trait for F {}

mod to_reuse {
    pub fn foo(x: i32) -> i32 { x + 1 }
    pub fn zero_args() -> i32 { 15 }
}

reuse to_reuse::zero_args { self }

struct S(F);
impl Trait for S {
    reuse Trait::bar { self.0 }
    reuse Trait::description { self.0 }
    reuse <F as Trait>::static_method;
    reuse <F as Trait>::static_method2 { S::static_method(self) }
}

impl S {
    reuse <F as Trait>::static_method { to_reuse::foo(self) }
}

impl std::fmt::Display for S {
    reuse <str as std::fmt::Display>::fmt { self.description() }
}

fn main() {
    let s = S(F);
    assert_eq!(42, s.bar(42));
    assert_eq!("hello world!", format!("{s}"));
    assert_eq!(43, S::static_method(42));
    assert_eq!(42, <S as Trait>::static_method(42));
    assert_eq!(21, S::static_method2(10, 10));

    #[inline]
    reuse to_reuse::foo;
    assert_eq!(43, foo(42));
    assert_eq!(15, zero_args());
}
