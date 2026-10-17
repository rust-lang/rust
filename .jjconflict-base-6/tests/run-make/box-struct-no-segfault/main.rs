#![crate_type = "lib"]
extern crate foo;
use foo::Foo;

pub fn crash() -> Box<Foo> {
    Box::new(Foo::new())
}
