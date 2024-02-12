// edition:2021

#![feature(async_closure)]

pub fn closure() -> impl async Fn() {
    async || { /* Don't really need to do anything here. */ }
}
