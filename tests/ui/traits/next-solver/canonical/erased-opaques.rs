//@ compile-flags: -Znext-solver
//@ build-pass
//@ edition: 2021
//@ compile-flags: -C debuginfo=1 --crate-type=lib

pub(crate) struct Foo;

impl From<()> for Foo {
    fn from(_: ()) -> Foo {
        String::new().extend('a'.to_uppercase());
        Foo
    }
}
