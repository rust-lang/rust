//@ edition:2021

pub fn closure() -> impl AsyncFn() {
    async || { /* Don't really need to do anything here. */ }
}
