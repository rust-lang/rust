//@ known-bug: #140484
//@edition:2024
#![feature(async_drop)]
use std::future::AsyncDrop;
struct a;
impl Drop for a {
    fn b() {}
}
impl AsyncDrop for a {
    type c;
}
async fn bar() {
    a;
}
