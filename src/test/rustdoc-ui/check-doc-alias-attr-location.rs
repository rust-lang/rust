#![feature(doc_alias)]

pub struct Bar;
pub trait Foo {}

#[doc(alias = "foo")] //~ ERROR
extern {}

#[doc(alias = "bar")] //~ ERROR
impl Bar {}

#[doc(alias = "foobar")] //~ ERROR
impl Foo for Bar {}
