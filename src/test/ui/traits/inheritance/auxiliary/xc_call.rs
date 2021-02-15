pub trait Foo {
    fn f(&self) -> isize;
}

pub struct A {
    pub x: isize
}

impl Foo for A {
    fn f(&self) -> isize { 10 }
}
