#![crate_name = "foo"]

#![feature(staged_api)]

#[stable(feature = "bar", since = "1.0.0")]
pub trait Bar {}

#[stable(feature = "baz", since = "1.0.0")]
pub trait Baz {}

pub struct Foo;

// @has foo/trait.Bar.html '//div[@id="implementors-list"]//span[@class="since"]' '2.0.0'
#[stable(feature = "foobar", since = "2.0.0")]
impl Bar for Foo {}

// @!has foo/trait.Baz.html '//div[@id="implementors-list"]//span[@class="since"]' '1.0.0'
#[stable(feature = "foobaz", since = "1.0.0")]
impl Baz for Foo {}
