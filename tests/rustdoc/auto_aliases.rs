#![feature(rustc_attrs)]

// @has auto_aliases/trait.Bar.html '//*[@data-aliases="auto_aliases::Foo"]' 'impl Bar for Foo'
pub struct Foo;

#[rustc_auto_trait]
pub trait Bar {}
