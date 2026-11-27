#![feature(generic_const_exprs)]

pub struct Foo<const N: usize>;

pub fn foo<const N: usize>() -> Foo<{ N + 1 }> {
    Foo
}
