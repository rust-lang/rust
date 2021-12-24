// run-fail
// error-pattern:panicked at 'test-assert-owned'
// ignore-emscripten no processes

#![allow(non_fmt_panics)]

use std::any::TypeId;

fn main() {
    println!("String:       {:?}", TypeId::of::<String>());
    println!("&'static str: {:?}", TypeId::of::<&'static str>());
    assert!(false, "test-assert-owned".to_string());
}
