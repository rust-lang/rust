// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
trait SomeTrait {}
struct Meow;
impl SomeTrait for Meow {}

struct Foo<'a> {
    x: &'a SomeTrait,
    y: &'a SomeTrait,
}

impl<'a> Foo<'a> {
    pub fn new<'b>(x: &'b SomeTrait, y: &'b SomeTrait) -> Foo<'b> { Foo { x: x, y: y } }
}

fn main() {
    let r = Meow;
    let s = Meow;
    let q = Foo::new(&r as &SomeTrait, &s as &SomeTrait);
}
