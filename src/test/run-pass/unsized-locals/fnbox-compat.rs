#![feature(fnbox)]

use std::boxed::FnBox;

fn call_it<T>(f: Box<dyn FnBox() -> T>) -> T {
    f()
}

fn main() {
    let s = "hello".to_owned();
    assert_eq!(&call_it(Box::new(|| s)) as &str, "hello");
}
