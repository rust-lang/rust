pub struct Foo;

pub trait Bar {}

pub fn bar() -> Box<dyn Bar> {
    unimplemented!()
}
