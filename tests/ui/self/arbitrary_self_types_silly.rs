//@ run-pass
#![feature(arbitrary_self_types)]

struct Foo;
struct Bar;

impl std::ops::Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &Foo
    }
}

impl Foo {
    fn bar(self: Bar) -> i32 { 3 }
}

fn main() {
    assert_eq!(3, Bar.bar());
}
