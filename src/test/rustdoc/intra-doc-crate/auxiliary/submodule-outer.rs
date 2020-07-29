#![crate_name = "bar"]
#![deny(intra_doc_resolution_failures)]

pub trait Foo {
    /// [`Bar`] [`Baz`]
    fn foo();
}

pub trait Bar {
}

pub trait Baz {
}
