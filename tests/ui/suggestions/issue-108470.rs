//@ run-rustfix
#![allow(dead_code)]

struct Foo {
    t: Thing
}

#[derive(Clone)]
struct Thing;

fn test_clone() {
    let t = &Thing;
    let _f = Foo {
        t //~ ERROR mismatched types
    };
}

struct Bar {
    t: bool
}

fn test_is_some() {
    let t = Option::<i32>::Some(1);
    let _f = Bar {
        t //~ ERROR mismatched types
    };
}

fn main() {}
