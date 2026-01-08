// https://github.com/rust-lang/rust/issues/81141
#![crate_name = "foo"]

use crate::bar::Foo as Alias;

pub mod bar {
    pub struct Foo<'a, T>(&'a T);
}

//@ has "foo/fn.foo.html"
//@ has - '//*[@class="rust item-decl"]/code' "pub fn foo<'a, T>(f: Foo<'a, T>) -> Foo<'a, usize>"
pub fn foo<'a, T>(f: Alias<'a, T>) -> Alias<'a, usize> {
    Alias(&0)
}
