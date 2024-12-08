//@ edition:2021

#![feature(async_closure)]

pub fn closure() -> impl AsyncFn() {
    async || { /* Don't really need to do anything here. */ }
}
