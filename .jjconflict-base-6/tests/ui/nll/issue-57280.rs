//@ check-pass

trait Foo {
    const BLAH: &'static str;
}

struct Placeholder;

impl Foo for Placeholder {
    const BLAH: &'static str = "hi";
}

fn foo(x: &str) {
    match x {
        <Placeholder as Foo>::BLAH => { }
        _ => { }
    }
}

fn main() {}
