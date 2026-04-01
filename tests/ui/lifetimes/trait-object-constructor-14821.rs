//! Regression test for https://github.com/rust-lang/rust/issues/14821

//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
trait SomeTrait {}
struct Meow;
impl SomeTrait for Meow {}

struct Foo<'a> {
    x: &'a dyn SomeTrait,
    y: &'a dyn SomeTrait,
}

impl<'a> Foo<'a> {
    pub fn new<'b>(x: &'b dyn SomeTrait, y: &'b dyn SomeTrait) -> Foo<'b> { Foo { x: x, y: y } }
}

fn main() {
    let r = Meow;
    let s = Meow;
    let q = Foo::new(&r as &dyn SomeTrait, &s as &dyn SomeTrait);
}
