// This test ensures that all const trait bounds are displayed, in particular `Destruct`.

#![feature(const_trait_impl)]
#![crate_name = "foo"]

use std::marker::Destruct;

pub const trait Foo {}

//@ has 'foo/fn.f.html'
//@ matches - '//*[@class="rust item-decl"]//*[@class="where"]' '^where\s+T: Foo,$'
pub const fn f<T>(_: &T)
where T: [const] Foo
{}

//@ has 'foo/fn.f2.html'
//@ matches - '//*[@class="rust item-decl"]//*[@class="where"]' '^where\s+T: Destruct,$'
pub const fn f2<T>(_: &T)
where T: [const] Destruct
{}
