//@ edition:2021

#![allow(incomplete_features)]

pub trait Foo {
    //@ has async_trait_sig/trait.Foo.html '//h4[@class="code-header"]' "async fn bar() -> i32"
    async fn bar() -> i32;

    //@ has async_trait_sig/trait.Foo.html '//h4[@class="code-header"]' "async fn baz() -> i32"
    async fn baz() -> i32 {
        1
    }
}
