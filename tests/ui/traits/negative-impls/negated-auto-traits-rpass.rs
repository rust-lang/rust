//@ run-pass
#![allow(unused_variables)]
#![feature(negative_impls)]

pub struct WaitToken;
impl !Send for WaitToken {}

pub struct Test<T>(#[allow(dead_code)] T);
unsafe impl<T: 'static> Send for Test<T> {}

pub fn spawn<F>(_: F) -> () where F: FnOnce(), F: Send + 'static {}

fn main() {
    let wt = Test(WaitToken);
    spawn(move || {
        let x = wt;
        println!("Hello, World!");
    });
}
