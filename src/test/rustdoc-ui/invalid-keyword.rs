#![feature(rustdoc_internals)]

#[doc(keyword = "foo df")] //~ ERROR
mod foo {}
