// run-pass
// pretty-expanded FIXME #23616

trait Foo {
    type T;
    fn foo() -> Box<<Self as Foo>::T>;
}

fn main() {}
