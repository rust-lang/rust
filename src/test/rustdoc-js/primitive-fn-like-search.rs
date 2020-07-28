pub struct Foo;

pub trait Bar {}

pub fn foo<T: Bar, D: ::std::fmt::Debug>(a: Foo, b: u32, c: T, d: D) -> u32 {0}
