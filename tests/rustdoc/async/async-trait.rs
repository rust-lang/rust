//@ aux-build:async-trait-dep.rs
//@ edition:2021

#![allow(incomplete_features)]

extern crate async_trait_dep;

pub struct Oink {}

//@ has 'async_trait/struct.Oink.html' '//h4[@class="code-header"]' "async fn woof()"
impl async_trait_dep::Meow for Oink {
    async fn woof() {
        todo!()
    }
}
