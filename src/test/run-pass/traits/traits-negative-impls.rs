// run-pass
#![allow(unused_variables)]
#![feature(optin_builtin_traits)]

use std::marker::Send;

pub struct WaitToken;
impl !Send for WaitToken {}

pub struct Test<T>(T);
unsafe impl<T: 'static> Send for Test<T> {}

pub fn spawn<F>(_: F) -> () where F: FnOnce(), F: Send + 'static {}

fn main() {
    let wt = Test(WaitToken);
    spawn(move || {
        let x = wt;
        println!("Hello, World!");
    });
}
