// Testing the output when an invalid builtin attribute is passed as value
// to `doc(attribute = "...")`.

#![feature(rustdoc_internals)]

#[doc(attribute = "foo df")] //~ ERROR
mod foo {}

#[doc(attribute = "fooyi")] //~ ERROR
mod foo2 {}
