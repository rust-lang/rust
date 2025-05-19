//@ known-bug: #140500

#![feature(async_drop)]
use std::future::AsyncDrop;
struct a;
impl Drop for a {
    fn b() {}
}
impl AsyncDrop for a {
    fn c(d: impl Sized) {}
}
async fn bar() {
    a;
}
