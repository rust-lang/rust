#![feature(unsized_locals)]
#![allow(incomplete_features)]

fn main() {
    pub trait Foo {
        fn foo(self) -> String;
    }

    struct A;

    impl Foo for A {
        fn foo(self) -> String {
            format!("hello")
        }
    }

    let x = *(Box::new(A) as Box<dyn Foo>); //~ERROR: unsized locals are not supported
    assert_eq!(x.foo(), format!("hello"));

    // I'm not sure whether we want this to work
    let x = Box::new(A) as Box<dyn Foo>;
    assert_eq!(x.foo(), format!("hello"));
}
