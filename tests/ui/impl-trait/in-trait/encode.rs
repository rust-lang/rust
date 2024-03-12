//@ build-pass
//@ compile-flags: --crate-type=lib

trait Foo {
    fn bar() -> impl Sized;
}
