#![feature(unsized_locals)]

pub trait Foo {
    fn foo(self) -> String;
}

struct A;

impl Foo for A {
    fn foo(self) -> String {
        format!("hello")
    }
}

struct B(i32);

impl Foo for B {
    fn foo(self) -> String {
        format!("{}", self.0)
    }
}

struct C(String);

impl Foo for C {
    fn foo(self) -> String {
        self.0
    }
}

struct D(Box<String>);

impl Foo for D {
    fn foo(self) -> String {
        *self.0
    }
}


fn main() {
    let x = *(Box::new(A) as Box<dyn Foo>);
    assert_eq!(x.foo(), format!("hello"));
    let x = *(Box::new(B(42)) as Box<dyn Foo>);
    assert_eq!(x.foo(), format!("42"));
    let x = *(Box::new(C(format!("jumping fox"))) as Box<dyn Foo>);
    assert_eq!(x.foo(), format!("jumping fox"));
    let x = *(Box::new(D(Box::new(format!("lazy dog")))) as Box<dyn Foo>);
    assert_eq!(x.foo(), format!("lazy dog"));
}
