#![stable(feature = "bar", since = "3.3.3")]
#![crate_name = "foo"]

#![feature(staged_api)]

#[stable(feature = "bar", since = "3.3.3")]
pub trait Bar {}

#[stable(feature = "baz", since = "3.3.3")]
pub trait Baz {}

#[stable(feature = "baz", since = "3.3.3")]
pub struct Foo;

//@ has foo/trait.Bar.html '//div[@id="implementors-list"]//span[@class="since"]' '4.4.4'
#[stable(feature = "foobar", since = "4.4.4")]
impl Bar for Foo {}

//@ has foo/trait.Baz.html '//div[@id="implementors-list"]//span[@class="since"]' '3.3.3'
#[stable(feature = "foobaz", since = "3.3.3")]
impl Baz for Foo {}
