//@ check-pass

trait Foo {
    type T;
    fn foo() -> Box<<Self as Foo>::T>;
}

fn main() {}
