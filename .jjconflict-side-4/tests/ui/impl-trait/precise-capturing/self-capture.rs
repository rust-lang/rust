//@ check-pass

trait Foo {
    fn bar<'a>() -> impl Sized + use<Self>;
}

fn main() {}
