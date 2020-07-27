#![crate_name = "bar"]
#![deny(intra_doc_link_resolution_failure)]

pub trait Foo {
    /// [`Bar`] [`Baz`]
    fn foo();
}

pub trait Bar {
}

pub trait Baz {
}
