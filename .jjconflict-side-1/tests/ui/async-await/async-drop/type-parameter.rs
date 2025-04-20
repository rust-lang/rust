//@ edition: 2024
// ex-ice: #140500
#![crate_type = "lib"]
#![feature(async_drop)]
#![expect(incomplete_features)]
use std::future::AsyncDrop;
struct A;
impl Drop for A {
    fn drop(&mut self) {}
}
impl AsyncDrop for A {
    fn drop(_wrong: impl Sized) {} //~ ERROR: method `drop` has a `self: Pin<&mut Self>` declaration in the trait, but not in the impl
}
async fn bar() {
    A;
}
