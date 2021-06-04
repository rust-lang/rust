#![feature(auto_traits)]

// @has auto_aliases/trait.Bar.html '//div[@data-aliases="auto_aliases::Foo"]' 'impl Bar for Foo'
pub struct Foo;

pub auto trait Bar {}
