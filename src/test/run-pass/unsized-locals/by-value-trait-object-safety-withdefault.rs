#![feature(unsized_locals)]

pub trait Foo {
    fn foo(self) -> String {
        format!("hello")
    }
}

struct A;

impl Foo for A {}


fn main() {
    let x = *(Box::new(A) as Box<dyn Foo>);
    assert_eq!(x.foo(), format!("hello"));

    // I'm not sure whether we want this to work
    let x = Box::new(A) as Box<dyn Foo>;
    assert_eq!(x.foo(), format!("hello"));
}
