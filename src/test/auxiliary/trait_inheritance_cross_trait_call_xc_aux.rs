
pub trait Foo {
    fn f() -> int;
}

pub struct A {
    x: int
}

impl A : Foo {
    fn f() -> int { 10 }
}
