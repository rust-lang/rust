pub struct Foo;

pub trait Bar{}

pub fn bar() -> Box<Bar> {
    unimplemented!()
}


pub fn try_foo(x: Foo){}
pub fn try_bar(x: Box<Bar>){}
