//@ check-pass

#![deny(dead_code)]

#[allow(dead_code)]
trait Foo {
    const FOO: u32;
    type Baz;
    fn foobar();
}

const fn bar(x: u32) -> u32 {
    x
}

struct Qux;

struct FooBar;

impl Foo for u32 {
    const FOO: u32 = bar(0);
    type Baz = Qux;

    fn foobar() {
        let _ = FooBar;
    }
}

fn main() {}
