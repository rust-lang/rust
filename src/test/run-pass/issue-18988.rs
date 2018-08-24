pub trait Foo : Send { }

pub struct MyFoo {
    children: Vec<Box<Foo>>,
}

impl Foo for MyFoo { }

pub fn main() { }
