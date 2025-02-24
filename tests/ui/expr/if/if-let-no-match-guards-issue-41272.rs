//@ check-pass
#![allow(dead_code)]
struct Foo;

impl Foo {
    fn bar(&mut self) -> bool { true }
}

fn error(foo: &mut Foo) {
    if let Some(_) = Some(true) {
    } else if foo.bar() {}
}

fn ok(foo: &mut Foo) {
    if let Some(_) = Some(true) {
    } else {
        if foo.bar() {}
    }
}

fn main() {}
