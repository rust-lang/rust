#![allow(dead_code)]

struct Foo;

impl PartialEq for Foo {
    fn eq(&self, _: &Foo) -> bool {
        true
    }
    fn ne(&self, _: &Foo) -> bool {
        //~^ partialeq_ne_impl

        false
    }
}

struct Bar;

impl PartialEq for Bar {
    fn eq(&self, _: &Bar) -> bool {
        true
    }
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, _: &Bar) -> bool {
        false
    }
}

fn main() {}
