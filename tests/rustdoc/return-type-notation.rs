//@ edition: 2021

#![crate_type = "lib"]
#![feature(return_type_notation)]

pub trait Foo {
    async fn bar();
}

//@ has "return_type_notation/fn.foo.html"
//@ has - '//pre[@class="rust item-decl"]' "pub fn foo<T: Foo<bar(..): Send>>()"
//@ has - '//pre[@class="rust item-decl"]' "where <T as Foo>::bar(..): 'static, T::bar(..): Sync"
pub fn foo<T: Foo<bar(..): Send>>()
where
    <T as Foo>::bar(..): 'static,
    T::bar(..): Sync,
{
}
