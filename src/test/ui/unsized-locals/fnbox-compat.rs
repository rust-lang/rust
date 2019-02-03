#![feature(fnbox)]

use std::boxed::FnBox;

fn call_it<T>(f: Box<dyn FnBox(&i32) -> T>) -> T {
    f(&42)
    //~^ERROR implementation of `std::ops::FnOnce` is not general enough
}

fn main() {
    let s = "hello".to_owned();
    assert_eq!(&call_it(Box::new(|_| s)) as &str, "hello");
}
